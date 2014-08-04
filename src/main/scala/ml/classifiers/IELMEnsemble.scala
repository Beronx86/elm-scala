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
import ml.models.{ELMSimpleEnsembleModel, Model}
import ml.neural.elm.{Data, IELMTraitEnsemble}
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.XSRandom

case class IELMEnsemble(M: Int = 10, seed: Int = 42, notes: String = "", callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ())
  extends IELMTraitEnsemble {
  override val toString = "IELMEnsemble_" + notes
  val Lbuild = -1

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val m = ensCast(model)
    val nclasses = m.BetaS.head.numColumns()
    val L = nclasses + 1
    val natts = m.X.numColumns()
    val newBiasesS = new Array[Array[Double]](M)
    val newAlfatS = new Array[DenseMatrix](M)
    val newBetaS = new Array[DenseMatrix](M)
    val newES = new Array[Vector[DenseVector]](M)

    val newTmp = new DenseVector(m.X.numRows() + 1)
    val newX = Data.appendRowToMatrix(m.X, pattern.array)
    var me = 0
    var newRndG = m.rnd
    while (me < M) {
      newES(me) = m.eS(me).zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
      val (weights, bias, newRnd) = newNode(m.AlfatS(me).numColumns(), newRndG)
      newRndG = newRnd
      newAlfatS(me) = Data.appendRowToMatrix(m.AlfatS(me), weights)
      newBiasesS(me) = Data.appendToArray(m.biasesS(me), bias)
      val (h, beta) = addNode(weights, bias, newX, newES(me), newTmp)
      newBetaS(me) = Data.appendRowToMatrix(m.BetaS(me), beta)
      me += 1
    }


    ELMSimpleEnsembleModel(newRndG, newAlfatS, newBiasesS, newBetaS, newX, newES, null)
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

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector) = {
    val (weights, bias, newRnd) = newNode(X.numColumns(), rnd)
    rnd.setSeed(newRnd.getSeed)
    val (h, beta) = addNode(weights, bias, X, e, tmp)
    (weights, bias, h, beta)
  }
}


