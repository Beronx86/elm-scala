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

import ml.models.{ELMOnlineModel, Model}
import no.uib.cipr.matrix.{Matrices, DenseVector, DenseMatrix}
import ml.Pattern
import ml.neural.elm.Math._
import scala.util.Random
import ml.mtj.{ResizableDenseMatrix, DenseMatrix2}
import ml.neural.elm.Data._

/**
 * Created by davi on 21/05/14.
 */
object ELMUtils {
  def distribution(output: DenseMatrix) = {
    val data = output.getData
    val Tinc = new DenseVector(data, false)
    val sum = data.sum
    Tinc.scale(1 / sum)
    data
  }


  def test(patt: Pattern, Alfat: DenseMatrix, biases: Array[Double], Beta: DenseMatrix) = {
    val (h, hm) = feedHiddenv(new DenseVector(patt.array, false), Alfat, biases)
    val O = feedOutput(hm, Beta)
    applyOnMatrix(O, sigm)
    O
  }

  protected def feedOutput(H: DenseMatrix, Beta: DenseMatrix) = {
    val O = new DenseMatrix(H.numRows(), Beta.numColumns())
    H.mult(Beta, O)
    O
  }

  def calculateP(H: DenseMatrix) = {
    println("Recalculating P; perhaps part of this can be avoided...")
    val Ht = new DenseMatrix(H.numColumns(), H.numRows())
    H.transpose(Ht)
    val HtH = new DenseMatrix(Ht.numRows(), H.numColumns())
    Ht.mult(H, HtH) //15.5%
    inv(HtH) //9.1%
  }

  def feedHiddenv(x: DenseVector, Alfat: DenseMatrix, biases: Array[Double]) = {
    val h = new DenseVector(Alfat.numRows())
    Alfat.mult(x, h)
    //    h.add(new DenseVector(biases))
    //    applyOnVector(h, sigm2)
    val data = h.getData
    var j = 0
    while (j < h.size) {
      data(j) = sigm2(data(j) + biases(j))
      j += 1
    }
    val H = new DenseMatrix2(data)
    H.resize(1, h.size)
    (h, H)
  }

  //  protected def cast(model: Model) = model match {
  //    case m: ELMModel => m
  //    case _ => println("ELM requires ELMModel.")
  //      sys.exit(0)
  //  }
  //
  //

  //
  //  protected def feedOutput(H: DenseMatrix, Beta: DenseMatrix) = {
  //    val O = new DenseMatrix(H.numRows(), Beta.numColumns())
  //    H.mult(Beta, O)
  //    O
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

}