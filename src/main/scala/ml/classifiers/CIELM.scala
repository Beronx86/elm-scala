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
import ml.models.{ELMSimpleModel, Model}
import ml.mtj.ResizableDenseMatrix
import ml.neural.elm.{Data, ConvexIELMTrait}
import ml.neural.elm.Data._
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.{Tempo, XSRandom}

/**
 * CI-ELM
 * Created by davi on 19/05/14.
 */
case class CIELM(seed: Int = 42, notes: String = "", callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ()) extends ConvexIELMTrait {
  override val toString = "CIELM_" + notes
  val Lbuild = -1

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val m = cast(model)
    val newE = m.e.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
    val newX = Data.appendRowToMatrix(m.X, pattern.array)
    val newTmp = new DenseVector(newX.numRows())

    val (weights, bias, newRnd) = newNode(m.Alfat.numColumns(), m.rnd)
    val newAlfat = Data.appendRowToMatrix(m.Alfat, weights)
    val newBiases = Data.appendToArray(m.biases, bias)
    //    val (h, beta) = addNodeForConvexUpdate(weights, bias, newX, newE, newTmp)
    //    val newBeta = Data.appendRowToMatrix(m.Beta, beta)
    //
    //    ELMSimpleModel(newRnd, newAlfat, newBiases, newBeta, newX, newE)
    ???
  }

  def build(trSet: Seq[Pattern]): Model = {
    Tempo.start
    val rnd = new XSRandom(seed)
    val ninsts = checkEmptyness(trSet)
    val L = if (Lbuild == -1) ninsts else Lbuild
    val natts = trSet.head.nattributes
    val nclasses = trSet.head.nclasses
    val X = patterns2matrix(trSet, ninsts)
    val biases = Array.fill(L)(0d)
    val Alfat = new ResizableDenseMatrix(L, natts)
    val Beta = new ResizableDenseMatrix(L, nclasses)
    val (t, e) = patterns2te(trSet, ninsts)
    var l = 0
    while (l < L) {
      Alfat.resizeRows(l + 1) //needed to call f()
      Beta.resizeRows(l + 1)
      val (weights, b, newRnd) = newNode(natts, rnd)
      rnd.setSeed(newRnd.getSeed)
      val (h, beta) = addNodeForConvexUpdate(weights, b, X, t, e)
      biases(l) = b
      updateNetwork(l, weights, beta, Beta, Alfat)
      l += 1
      val te = Tempo.stop
      Tempo.start
      f(ELMSimpleModel(newRnd, Alfat, biases, Beta, null, null), te)
    }
    //    Alfat.resizeRows(l)
    //    Beta.resizeRows(l)
    val model = ELMSimpleModel(rnd, Alfat, biases, Beta, null, null)
    model
  }

  /**
   * Mutate e
   * @return
   */
  def addNodeForConvexUpdate(weights: Array[Double], bias: Double, X: DenseMatrix, t: Array[DenseVector], e: Array[DenseVector]) = {
    val nclasses = t.size
    //Generate node and calculate h.
    val alfa = new DenseVector(weights, false)
    val beta = new Array[Double](nclasses)
    val h = feedHidden(X, alfa, bias)
    val hneg = h.copy()
    hneg.scale(-1)
    var o = 0
    while (o < nclasses) {
      //Calculate new weight.
      val t_h = t(o).copy()
      t_h.add(hneg)
      val a = t_h.copy()
      a.scale(-1)
      a.add(e(o))
      val nume = e(o).dot(a)
      val deno = a.dot(a)
      val b = nume / deno
      beta(o) = b

      //Recalculate residual error.
      e(o).scale(1 - b)
      t_h.scale(b)
      e(o).add(t_h)

      o += 1
    }
    (h, beta)
  }
}

