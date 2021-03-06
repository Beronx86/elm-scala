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

import ml.models.{ELMGroModel, Model}
import ml.mtj.DenseMatrix2
import ml.neural.elm.ConvergentELM
import ml.neural.elm.Math._
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.XSRandom

trait ConvergentGrowing extends ConvergentELM {
  def growByOne(model: Model, fast_mutable: Boolean = false) = {
    //    if (fast_mutable) {      println("fastmut"); val bla = ???    } else Unit
    val m = cast(model)
    val (newAlfat, newBiases, newH, newHinv, newBeta, newRnd) = grow(m.rnd, m.H, m.Xt, m.Y, m.Hinv, m.Alfat, m.biases, m.HHinv)
    ELMGroModel(newRnd, newAlfat, newBiases.getData, newBeta, m.Xt, m.Y, newH, newHinv)
  }

  def growTo(desiredL: Int, model: Model, fast_mutable: Boolean = false) = {
    //todo: foldLeft is way slower than a good while
    (2 to desiredL).foldLeft(cast(model))((m, p) => growByOne(m, fast_mutable))
  }

  /**
   * Fast and cheap.
   * @param m
   */
  def identMinusM(m: DenseMatrix) {
    val d = m.getData
    //    m.zero()
    var gap = m.numRows() + 1
    val l = d.size - 1
    var c = 0
    var cc = 0
    while (c < l) {
      d(c) = 1 - d(c)
      cc = 1
      while (cc < gap - 1) {
        d(c + cc) = 0 - d(c + cc)
        cc += 1
      }
      c += gap
    }
    d(l) = 1 - d(l)
    //    etln(m)
  }


  /**
   * Fast and cheap.
   * @param m
   */
  def mPlusIdentity(m: DenseMatrix) {
    val d = m.getData
    val l = d.size
    var c = 0
    var gap = m.numRows() + 1
    while (c < l) {
      d(c) += 1
      c += gap
    }
  }

  /**
   * Fast and cheap.
   * @param m
   */
  def mMinusIdentity(m: DenseMatrix) {
    val d = m.getData
    val l = d.size
    var c = 0
    var gap = m.numRows() + 1
    while (c < l) {
      d(c) -= 1
      c += gap
    }
  }

  /**
   * This is not thread-safe since HHinv is changed for a few milisseconds. todo
   * @param rnd
   * @param H
   * @param Xt
   * @param Y
   * @param Hinv
   * @param Alfat
   * @param biases
   * @param HHinv
   * @return
   */
  protected def grow(rnd: XSRandom, H: DenseMatrix, Xt: DenseMatrix, Y: DenseMatrix, Hinv: DenseMatrix, Alfat: DenseMatrix, biases: Array[Double], HHinv: DenseMatrix) = {
    val (newAlfat, newNeuron, newBiases, newRnd) = addNeuron(rnd, Alfat, biases)
    val (newH, newh, newhm1XN) = resizeH(H, Xt, newNeuron, newBiases)

    //    val I = Matrices.identity(HHinv.numRows())
    val I_HHinv = HHinv //.add(-1, I)
    mMinusIdentity(I_HHinv)

    val num = new DenseMatrix(1, H.numRows())
    newhm1XN.mult(I_HHinv, num)
    mPlusIdentity(I_HHinv) //recovers original value

    val deno = new DenseVector(1)
    num.mult(newh, deno)
    val factor = 1 / deno.get(0)
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

    (newAlfat, newBiases, newH, newHinv, newBeta, newRnd)
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

  protected def resizeH(H: DenseMatrix, Xt: DenseMatrix, lastNeuron: DenseVector, biases: DenseVector) = {
    val newH = new DenseMatrix(H.numRows(), H.numColumns + 1)
    //    val newHt = new DenseMatrix(H.numColumns + 1, H.numRows())
    val h = new DenseVector(H.numRows())
    var i = 0
    var j = 0
    //todo: arraycopy
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

    val lastNeuronm = new DenseMatrix2(lastNeuron.getData)
    lastNeuronm.setAsRowVector()
    val hm = new DenseMatrix2(h.getData)
    hm.setAsRowVector()
    lastNeuronm.mult(Xt, hm)

    i = 0
    while (i < H.numRows()) {
      val v = sigm2(h.get(i) + biases.get(j))
      newH.set(i, j, v)
      //      newHt.set(j, i, v)
      h.set(i, v)
      i += 1
    }
    (newH, h, hm)
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
