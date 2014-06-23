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
import scala.util.Random
import ml.mtj.{ResizableDenseMatrix, DenseMatrix2}
import ml.neural.elm.Data._

/**
 * Created by davi on 21/05/14.
 */
trait ELM extends Learner {
  val seed: Int
  if (seed == 0) {
    println("Seed cannot be 0 because the fast random algorithm (XS) would produce only 0s.")
    sys.exit(0)
  }

  def EMC(model: Model)(patterns: Seq[Pattern]) = ???

  def expected_change(model: Model)(pattern: Pattern) = ???

  protected def feedHidden(X: DenseMatrix, alfa: DenseVector, bias: Double) = {
    val h = new DenseVector(X.numRows())
    X.mult(alfa, h)
    addAndApplyOnVector(h, bias, sigm2)
    h
  }

  protected def feedHiddent(Xt: DenseMatrix, Alfat: DenseMatrix, biases: Array[Double]) = {
    val Ht = new DenseMatrix(Alfat.numRows(), Xt.numColumns())
    val H = new DenseMatrix(Ht.numColumns(), Ht.numRows())
    Alfat.mult(Xt, Ht)
    Ht.transpose(H)
    addToEachLineOnMatrixAndApplyf(H, biases, sigm2)
    H
  }

  protected def checkEmptyness(trSet: Seq[Pattern]) = {
    val ninsts = trSet.length
    if (ninsts == 0) {
      println("ERROR: Empty training set.")
      sys.exit(0)
    }
    ninsts
  }

  /**
   * mutates rnd!
   * @param Alfat
   * @param biases
   * @param rnd
   */
  protected def initializeWeights(Alfat: DenseMatrix, biases: Array[Double], rnd: Random) {
    var i = 0
    var j = 0
    while (i < Alfat.numRows()) {
      j = 0
      while (j < Alfat.numColumns()) {
        val v = rnd.nextDouble() * 2 - 1
//        Alfa.set(j, i, v)
        Alfat.set(i, j, v)
        j += 1
      }
      biases(i) = rnd.nextDouble() * 2 - 1
      i += 1
    }
  }

  //
  //
  //
  //  protected def feedHiddent(Xt: DenseMatrix, Alfat: DenseMatrix, biases: Array[Double]) = {
  //    val Ht = new DenseMatrix(Alfat.numRows(), Xt.numColumns())
  //    val H = new DenseMatrix(Ht.numColumns(), Ht.numRows())
  //    Alfat.mult(Xt, Ht)
  //    Ht.transpose(H)
  //    addToEachLineOnMatrixAndApplyf(H, biases, sigm2)
  //    H
  //  }

  //
  //
  //  def PRESSaccuracy(Yt: DenseMatrix, Beta: DenseMatrix, H: DenseMatrix, Ht: DenseMatrix, P: DenseMatrix, label: Seq[Double]) = {
  //    val Pred = feedOutput(H, Beta)
  //    val PredT = new DenseMatrix(Pred.numColumns(), Pred.numRows())
  //    Pred.transpose(PredT)
  //    val Preddata = PredT.getData
  //    val Ydata = Yt.getData
  //    val Hdata = Ht.getData
  //    val size = Hdata.size //cagada?
  //    val L = Ht.numRows()
  //    val ninsts = Ht.numColumns()
  //    val nclasses = Yt.numRows()
  //    var iH = 0
  //    var iY = 0
  //    val harray = new Array[Double](L)
  //    val yarray = new Array[Double](nclasses)
  //    val parray = new Array[Double](nclasses)
  //    val hm = new DenseMatrix2(harray)
  //    hm.resize(1, L)
  //    val y = new DenseVector(yarray, false)
  //    val p = new DenseVector(parray, false)
  //    val h = new DenseVector(harray, false)
  //    var hits = 0
  //    var i = 0
  //    while (iH < size) {
  //      System.arraycopy(Hdata, iH, harray, 0, L)
  //      val hmPt = new DenseMatrix(1, L)
  //      hm.mult(P, hmPt)
  //      val hP = new DenseVector(hmPt.getData, false)
  //      val hPh = hP.dot(h)
  //      val den = 1 - hPh
  //
  //      System.arraycopy(Ydata, iY, yarray, 0, nclasses)
  //      System.arraycopy(Preddata, iY, parray, 0, nclasses)
  //      p.scale(-1)
  //      p.add(y)
  //      val diffs = p //esperado - predito
  //      diffs.scale(-1 / den)
  //      y.add(diffs)
  //      if (y.getData.zipWithIndex.max._2 == label(i)) hits += 1 //cagada?
  //      iH += L
  //      iY += nclasses
  //      i += 1
  //    }
  //    hits / i.toDouble
  //  }
  //

  //
  //
  //  def EMC(model: Model)(patterns: Seq[Pattern]): Pattern = ???
  //
  //  def expected_change(model: Model)(pattern: Pattern): Double = ???
  //
  //  def updateAll(model: Model, fast_mutable: Boolean)(patterns: Seq[Pattern]): Model = ???
  //
  //
  //
  //
  //
  //  def build(trSet: Seq[Pattern]): ELMModel
}

//rnd ok