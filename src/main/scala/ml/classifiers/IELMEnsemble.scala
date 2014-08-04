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
import ml.neural.elm.{IELMTraitEnsemble, Data, IELMTrait}
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.{Tempo, Datasets, XSRandom}

import scala.util.Random

case class IELMEnsemble(M: Int = 10, seed: Int = 42, notes: String = "", callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ())
  extends IELMTraitEnsemble {
  override val toString = "IELMEnsemble_" + notes
  val Lbuild = -1

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val m = cast(model)
    //    if (math.sqrt(m.N + 1).toInt > math.sqrt(m.N).toInt) build(?) //<- not possible to rebuild, since we don't have Y nor all patterns anymore.
    val newE = m.e.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
    val newX = Data.appendRowToMatrix(m.X, pattern.array)
    val newTmp = new DenseVector(newX.numRows())

    val (weights, bias, newRnd) = newNode(m.Alfat.numColumns(), m.rnd)
    val newAlfat = Data.appendRowToMatrix(m.Alfat, weights)
    val newBiases = Data.appendToArray(m.biases, bias)
    val (h, beta) = addNode(weights, bias, newX, newE, newTmp)
    val newBeta = Data.appendRowToMatrix(m.Beta, beta)

    ELMSimpleModel(newRnd, newAlfat, newBiases, newBeta, newX, newE, null)
  }

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector) = {
    val (weights, bias, newRnd) = newNode(X.numColumns(), rnd)
    rnd.setSeed(newRnd.getSeed)
    val (h, beta) = addNode(weights, bias, X, e, tmp)
    (weights, bias, h, beta)
  }

  def addNode(weights: Array[Double], bias: Double, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector) = {
    val nclasses = e.size
    //Generate node and calculate h.
    val alfa = new DenseVector(weights, false)
    val beta = new Array[Double](nclasses)
    val h = feedHidden(X, alfa, bias)
    var o = 0
    while (o < nclasses) {
      tmp.set(h)
      //Calculate new weight.
      val nume = e(o).dot(h)
      val deno = h.dot(h)
      val b = nume / deno
      beta(o) = b

      //Recalculate residual error.
      tmp.scale(-b)
      e(o).add(tmp)

      o += 1
    }
    (h, beta)
  }
}


