import ml.Pattern
import ml.models.Model
import ml.neural.elm.ELM

///*
//elm-scala: an implementation of ELM in Scala using MTJ
//Copyright (C) 2014 Davi Pereira dos Santos
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//*/
//package ml.classifiers
//
//import ml.neural.elm.Data._
//import ml.neural.elm.Math._
//import ml.Pattern
//import scala.util.Random
//import ml.models.Model
//import no.uib.cipr.matrix.{Matrices, DenseVector, DenseMatrix}
//import ml.mtj.DenseMatrix2
//import ml.neural.elm.ELM
//import util.Tempo
//
///**
// *
// */
case class interaELM(Lmax: Int, seed: Int = 0) extends ELM {
  ??? //todo:cuidado com a mutabilidade de rnd no update()!
  /**
   * Every call to build generates a model from scratch
   * (and reinstanciate all needed internal mutable objects, if any).
   * @param pool
   * @return
   */
  def build(pool: Seq[Pattern]) = ???

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = ???

  //  def updateAll(model: Model, fast_mutable: Boolean = false)(patterns: Seq[Pattern]): Model
  def updateAll(model: Model, fast_mutable: Boolean)(patterns: Seq[Pattern]) = ???
}
//  override val toString = "interaELM"
////
////  def build(trSet: Seq[Pattern]) = {
//////    val m = super.build(trSet)
////    //calcula PRESS
////    //cresce até PRESS parar de aumentar
////
////    //retorna modelo atualizado (recalcular P somente se cresceu).
////    ???
////  }
////
////   def update(model: Model, fast_mutable: Boolean = false)(pattern: Pattern) = {
////    val aelmModel = cast(model)
////    val Alfat = aelmModel.Alfat
////    val biases = aelmModel.biases
////    val P0 = aelmModel.P
////    val H = aelmModel.H
////    val rnd = aelmModel.rnd
////    val Beta0 = aelmModel.Beta //LxO
////    val (h, hm) = feedHiddenv(pattern.arraymtj, Alfat, biases) //h: Lx1; H: NxL
////
////    ???
////    val m = super.update(model, fast_mutable)(pattern)
////    ???
////    m
////  }
////
////  def grow(rnd: Random, H: DenseMatrix, X: DenseMatrix, Y: DenseMatrix, Hinv: DenseMatrix, Alfat: DenseMatrix, biases: Array[Double]) = {
////    val HHinv = new DenseMatrix(H.numRows(), Hinv.numColumns())
////    H.mult(Hinv, HHinv)
////    val (newAlfat, newNeuron, newBiases) = addNeuron(rnd, Alfat, new DenseVector(biases, false))
////    val (newH, newh) = resizeH(H, X, newNeuron, newBiases)
////
////    val I = Matrices.identity(HHinv.numRows())
////    val I_HHinv = HHinv.add(-1, I)
////    val tmp = new DenseMatrix(newh, true)
////    tmp.transpose()
////    val num = new DenseMatrix(1, H.numRows())
////    tmp.mult(I_HHinv, num)
////    val tmp2 = new DenseVector(newh.size())
////    num.mult(newh, tmp2)
////    val factor = 1 / tmp2.get(0)
////    num.scale(factor)
////    val D = num //1xN
////
////    val Hinvh = new DenseVector(Alfat.numRows()) //Lx1
////    Hinv.mult(newh, Hinvh)
////    val Hinvhm = new DenseMatrix(Hinvh, false)
////    val HinvhD = new DenseMatrix(Alfat.numRows(), H.numRows()) //LxN
////    Hinvhm.mult(D, HinvhD)
////    val tmp3 = Hinv.copy()
////    tmp3.add(-1, HinvhD)
////    val U = tmp3 //LxN
////
////    val newHinv = stackUD(U, D)
////    val newBetat = updateBeta(newAlfat, Y, newHinv)
////
////    (newAlfat, newBiases, newH, newHinv, newBetat)
////  }
////
////  protected def addNeuron(rnd: Random, Alfat: DenseMatrix, biases: DenseVector) = {
////    val newHiddenLayer = new DenseMatrix(Alfat.numRows + 1, Alfat.numColumns())
////    val newBiases = new DenseVector(biases.size + 1)
////    val newNeuron = new DenseVector(Alfat.numColumns())
////    var i = 0
////    var j = 0
////    while (i < Alfat.numRows) {
////      j = 0
////      while (j < Alfat.numColumns()) {
////        newHiddenLayer.set(i, j, Alfat.get(i, j))
////        j += 1
////      }
////      newBiases.set(i, biases.get(i))
////      i += 1
////    }
////    j = 0
////    while (j < Alfat.numColumns()) {
////      val v = rnd.nextDouble() * 2 - 1
////      newHiddenLayer.set(i, j, v)
////      newNeuron.set(j, v)
////      j += 1
////    }
////    newBiases.set(i, rnd.nextDouble() * 2 - 1)
////    (newHiddenLayer, newNeuron, newBiases)
////  }
////
////  protected def resizeH(H: DenseMatrix, X: DenseMatrix, lastNeuron: DenseVector, biases: DenseVector) = {
////    val newH = new DenseMatrix(H.numRows(), H.numColumns + 1)
////    val h = new DenseVector(H.numRows())
////    var i = 0
////    var j = 0
////    while (i < H.numRows()) {
////      j = 0
////      while (j < H.numColumns()) {
////        val v = H.get(i, j)
////        newH.set(i, j, v)
////        j += 1
////      }
////      i += 1
////    }
////    X.mult(lastNeuron, h)
////    i = 0
////    while (i < H.numRows()) {
////      val v = sigm2(h.get(i) + biases.get(j))
////      newH.set(i, j, v)
////      h.set(i, v)
////      i += 1
////    }
////    (newH, h)
////  }
////
////  private def stackUD(U: DenseMatrix, D: DenseMatrix) = {
////    val UD = new DenseMatrix(U.numRows + 1, U.numColumns())
////    var i = 0
////    var j = 0
////    while (i < U.numRows) {
////      j = 0
////      while (j < U.numColumns()) {
////        UD.set(i, j, U.get(i, j))
////        j += 1
////      }
////      i += 1
////    }
////    j = 0
////    while (j < U.numColumns()) {
////      UD.set(i, j, D.get(0, j))
////      j += 1
////    }
////    UD
////  }
////
////  protected def updateBeta(Alfat: DenseMatrix, Y: DenseMatrix, Hinv: DenseMatrix) = {
////    val outputLayer = new DenseMatrix(Alfat.numRows(), Y.numColumns()) //LxO
////    val outputLayert = new DenseMatrix(Y.numColumns(), Alfat.numRows()) //OxL
////    Hinv.mult(Y, outputLayer)
////    outputLayer.transpose(outputLayert)
////    outputLayert
////  }
//}
//
