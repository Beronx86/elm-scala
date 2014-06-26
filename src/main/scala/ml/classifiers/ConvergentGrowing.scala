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
import ml.models.{ELMGenericModel, Model}
import ml.neural.elm.ConvergentELM
import ml.neural.elm.Math._
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.XSRandom

trait ConvergentGrowing extends ConvergentELM {

  def growByOne(model: Model, fast_mutable: Boolean=false) = {
    if (fast_mutable) {
      val bla = ???
    } else Unit
    val m = cast(model)

    //immutable fields (if no more instances are added!)
    val xm = m.X
    val ym = m.Y

    //mutable fields
    val rnd = m.rnd
    val Alfat = m.Alfat
//    val Alfa = m.Alfa
    val biases = m.biases
    val hminv = m.Hinv
    val H = m.H
    val I = m.I

    //useless fields
//    lazy val P0 = m.P
//    lazy val Beta0 = m.Beta

    val (newAlfat, newBiases, newH, newHinv, newBeta, newRnd, hHinv) = grow(I, rnd, H, xm, ym, hminv, Alfat, biases, m.HHinv)
    ELMGenericModel(newRnd, newAlfat, newBiases.getData, newH, null, newBeta, xm, ym, newHinv, hHinv)
  }

  def growTo(desiredL: Int, model: Model, fast_mutable: Boolean = false) = {
    //todo: foldLeft is way slower than a good while
    (2 to desiredL).foldLeft(model)((m, p) => growByOne(m, fast_mutable))
  }

  protected def grow(I:DenseMatrix, rnd: XSRandom, H: DenseMatrix, X: DenseMatrix, Y: DenseMatrix, Hinv: DenseMatrix, Alfat: DenseMatrix, biases: Array[Double], HHinv:DenseMatrix) = {
    val (newAlfat, newNeuron, newBiases, newRnd) = addNeuron(rnd, Alfat, biases)
    val (newH, newh) = resizeH(H, X, newNeuron, newBiases)

//    val I = Matrices.identity(HHinv.numRows())
    val I_HHinv = HHinv.add(-1, I)
    val tmp = new DenseMatrix(newh, false)
    val tmpt = new DenseMatrix(1, tmp.numRows())
    tmp.transpose(tmpt)

    val num = new DenseMatrix(1, H.numRows())
    tmpt.mult(I_HHinv, num)
    val tmp2 = new DenseVector(1)
    num.mult(newh, tmp2)
    val factor = 1 / tmp2.get(0)
    num.scale(factor)
    val D = num //1xN

    val Hinvh = new DenseVector(Alfat.numRows()) //Lx1
    Hinv.mult(newh, Hinvh)
    val Hinvhm = new DenseMatrix(Hinvh, false)
    val HinvhD = new DenseMatrix(Alfat.numRows(), H.numRows()) //LxN
    Hinvhm.mult(D, HinvhD)
    val tmp3 = Hinv.copy()
    tmp3.add(-1, HinvhD)
    val U = tmp3 //LxN

    val newHinv = stackUD(U, D)
    val newBeta = updateBeta(Y, newHinv)
    val newHHinv = new DenseMatrix(H.numRows(), Hinv.numColumns())
    H.mult(Hinv, newHHinv)

    (newAlfat, newBiases, newH, newHinv, newBeta, newRnd, newHHinv)
  }

  protected def addNeuron(rnd: XSRandom, alfat: DenseMatrix, biases: Array[Double]) = {
    val newRnd = rnd.clone()
    val newAlfat = new DenseMatrix(alfat.numRows + 1, alfat.numColumns())
    val newBiases = new DenseVector(biases.size + 1)
    val newNeuron = new DenseVector(alfat.numColumns())
    var i = 0
    var j = 0
    while (i < alfat.numRows) {
      j = 0
      while (j < alfat.numColumns()) {
        newAlfat.set(i, j, alfat.get(i, j))
        j += 1
      }
      newBiases.set(i, biases(i))
      i += 1
    }
    j = 0
    while (j < alfat.numColumns()) {
      val v = newRnd.nextDouble() * 2 - 1
      newAlfat.set(i, j, v)
      newNeuron.set(j, v)
      j += 1
    }
    newBiases.set(i, newRnd.nextDouble() * 2 - 1)
    (newAlfat, newNeuron, newBiases, newRnd)
  }

  protected def resizeH(H: DenseMatrix, X: DenseMatrix, lastNeuron: DenseVector, biases: DenseVector) = {
    val newH = new DenseMatrix(H.numRows(), H.numColumns + 1)
    //    val newHt = new DenseMatrix(H.numColumns + 1, H.numRows())
    val h = new DenseVector(H.numRows())
    var i = 0
    var j = 0
    while (i < H.numRows()) {
      j = 0
      while (j < H.numColumns()) {
        val v = H.get(i, j)
        newH.set(i, j, v)
        //        newHt.set(j, i, v)
        j += 1
      }
      i += 1
    }
    X.mult(lastNeuron, h)
    i = 0
    while (i < H.numRows()) {
      val v = sigm2(h.get(i) + biases.get(j))
      newH.set(i, j, v)
      //      newHt.set(j, i, v)
      h.set(i, v)
      i += 1
    }
    (newH, h)
  }

  private def stackUD(U: DenseMatrix, D: DenseMatrix) = {
    val UD = new DenseMatrix(U.numRows + 1, U.numColumns())
    var i = 0
    var j = 0
    while (i < U.numRows) {
      j = 0
      while (j < U.numColumns()) {
        UD.set(i, j, U.get(i, j))
        j += 1
      }
      i += 1
    }
    j = 0
    while (j < U.numColumns()) {
      UD.set(i, j, D.get(0, j))
      j += 1
    }
    UD
  }

  protected def updateBeta(Y: DenseMatrix, Hinv: DenseMatrix) = {
    val Beta = new DenseMatrix(Hinv.numRows(), Y.numColumns()) //LxO
//    val Betat = new DenseMatrix(Y.numColumns(), Hinv.numRows()) //OxL
    Hinv.mult(Y, Beta)
//    Beta.transpose(Betat)
    Beta
  }
}

//rnd ok
