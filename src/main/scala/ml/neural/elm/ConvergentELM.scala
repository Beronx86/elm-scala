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
package ml.neural.elm

import ml.classifiers.Learner
import ml.models.{ELMGenericModel, Model}
import no.uib.cipr.matrix.{Matrices, DenseVector, DenseMatrix}
import ml.Pattern
import ml.neural.elm.Math._
import util.XSRandom
import scala.util.Random
import ml.mtj.{ResizableDenseMatrix, DenseMatrix2}
import ml.neural.elm.Data._

/**
 * Created by davi on 21/05/14.
 */
trait ConvergentELM extends ELM {
  val Lbuild: Int

  def build(trSet: Seq[Pattern]) = {
    val rnd = new XSRandom(seed)

    val ninsts = checkEmptyness(trSet)
    checkFullRankness(trSet, ninsts)
    val natts = trSet.head.nattributes
    val nclasses = trSet.head.nclasses
    val t = patterns2matrices(trSet, ninsts)
    val Xt = t._1
    val Y = t._2
    val biasesArray = new Array[Double](Lbuild)
//    val Alfa = new DenseMatrix(Lbuild, natts)
    val Alfat = new DenseMatrix(Lbuild, natts)
    initializeWeights(Alfat, biasesArray, rnd)

    val H = feedHiddent(Xt, Alfat, biasesArray)
    val Ht = new DenseMatrix(Lbuild, ninsts)
    H.transpose(Ht)
    val HtH = new DenseMatrix(Lbuild, Lbuild)
    Ht.mult(H, HtH)
    val P = inv(HtH)
    val pinvH = new DenseMatrix(Lbuild, ninsts)
    P.mult(Ht, pinvH)
    val Beta = new DenseMatrix(Lbuild, nclasses)
    pinvH.mult(Y, Beta)

    //todo:X and HHinv are calculated even when it is useless (e.g. for OS-ELM)! lazy is ineffective in call-by-value
    lazy val X = new DenseMatrix(Xt.numColumns(), Xt.numRows())
    Xt.transpose(X)

    lazy val HHinv = {
      val r = new DenseMatrix(H.numRows(), H.numRows())
      H.mult(pinvH, r)
      r
    }

    ELMGenericModel(rnd, Alfat, biasesArray, H, P, Beta, X, Y, pinvH, HHinv)
  }

  /**
   * This is not a problem for I-ELM and variants, but is needed for OS-ELM (and EM-ELM?).
   * At the moment the test is fast, i.e. it doesn't really check the rank.
   * @param trSet
   * @param ninsts
   * @return
   */
  protected def checkFullRankness(trSet: Seq[Pattern], ninsts: Int) {
    if (ninsts < Lbuild) {
      println("ERROR: Training set size (" + ninsts + ") is lesser than L (" + Lbuild + ")!")
      sys.exit(0)
    }
  }

  protected def PRESS
}

//rnd ok