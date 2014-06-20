/*
elm-scala: an implementation of ELM in Scala using MTJ
Copyright (C) 2014 Davi Pereira dos Santos

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

package ml.classifiers

import ml.Pattern
import ml.mtj.DenseMatrix2
import ml.neural.elm.Data._
import ml.neural.elm.Math._
import no.uib.cipr.matrix.{DenseVector, DenseMatrix}

import scala.util.Random
import ml.models.{ELMOnlineModel, Model}
import ml.neural.elm.{ELMUtils, ELM}

case class OSELM(L: Int, seed: Int = 0) extends ELM {
  override val toString = "OSELM"

  protected def cast(model: Model) = model match {
    case m: ELMOnlineModel => m
    case _ => println("ELM and variants require ELMModel.")
      sys.exit(0)
  }

  /**
   * This is not a problem for I-ELM and variants, but is needed for OS-ELM (and EM-ELM?).
   * At the moment the test is fast, i.e. it doesn't really check the rank.
   * @param trSet
   * @param ninsts
   * @return
   */
  protected def checkFullRankness(trSet: Seq[Pattern], ninsts: Int) {
    if (ninsts < L) {
      println("ERROR: Training set size (" + ninsts + ") is lesser than L (" + L + ")!")
      sys.exit(0)
    }
  }

  def build(trSet: Seq[Pattern]) = {
    val rnd = new Random(seed)

    val ninsts = checkEmptyness(trSet)
    checkFullRankness(trSet, ninsts)
    val natts = trSet.head.nattributes
    val nclasses = trSet.head.nclasses
    val t = patterns2matrices(trSet,ninsts)
    val Xt = t._1
    val Y = t._2
    val biasesArray = new Array[Double](L)
    val Alfat = new DenseMatrix(L, natts)
    initializeWeights(Alfat, biasesArray, rnd)

    val H = feedHiddent(Xt, Alfat, biasesArray)
    val Ht = new DenseMatrix(L, ninsts)
    H.transpose(Ht)
    val HtH = new DenseMatrix(L, L)
    Ht.mult(H, HtH)
    val P = inv(HtH)
    val pinvH = new DenseMatrix(L, ninsts)
    P.mult(Ht, pinvH)
    val Beta = new DenseMatrix(L, nclasses)
    pinvH.mult(Y, Beta)
    ELMOnlineModel(rnd, Alfat, biasesArray, H, P, Beta)
  }

  def update(model: Model, fast_mutable: Boolean = false)(pattern: Pattern) = {
    val m = cast(model)
    val Alfat = m.Alfat
    val biases = m.biases
    val P0 = m.P
    val H = m.H
    val rnd = m.rnd
    val Beta0 = m.Beta //LxO
    val (h, hm) = ELMUtils.feedHiddenv(pattern.arraymtj, Alfat, biases) //h: Lx1; H: NxL

    val L = h.size()
    val O = Beta0.numColumns()
    val y = pattern.weighted_label_array //y: Ox1; Y: LxO
    val ym = new DenseMatrix2(y)
    ym.resize(1, O)

    //P1
    val tmpLx1 = new DenseVector(P0.numRows())
    val tmpLx1m = new DenseMatrix(tmpLx1, false)
    P0.mult(h, tmpLx1) //Lx1
    val tmp = h.dot(tmpLx1)
    val factor = -1 / (1 + tmp)
    val P0hht = new DenseMatrix(L, L)
    tmpLx1m.mult(hm, P0hht) //LxL
    val deltaP = new DenseMatrix(L, L)
    P0hht.mult(P0, deltaP) //LxL
    deltaP.scale(factor)
    val P1 = if (fast_mutable) {
      P0.add(deltaP)
      P0
    } else {
      deltaP.add(P0)
      deltaP
    }

    //Beta1
    val parens = new DenseMatrix(1, O)
    hm.mult(Beta0, parens) //1xO
    parens.scale(-1)
    parens.add(ym)
    deltaP.mult(h, tmpLx1)
    val tmpLxO = new DenseMatrix(L, O)
    tmpLx1m.mult(parens, tmpLxO)
    val Beta1 = if (fast_mutable) {
      Beta0.add(tmpLxO)
      Beta0
    } else {
      tmpLxO.add(Beta0)
      tmpLxO
    }

    //todo: atualizar H? H fica mais comprido a cada update!
    ELMOnlineModel(rnd, Alfat, biases, H, P1, Beta1)
  }

  def updateAll(model: Model, fast_mutable: Boolean)(patterns: Seq[Pattern]) = ???
}