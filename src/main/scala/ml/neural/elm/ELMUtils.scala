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

  def feedOutput(H: DenseMatrix, Beta: DenseMatrix): DenseMatrix = {
    val Y = new DenseMatrix(H.numRows(), Beta.numColumns())
    feedOutput(H, Beta, Y)
  }

  def feedOutput(H: DenseMatrix, Beta: DenseMatrix, Y: DenseMatrix): DenseMatrix = {
    H.mult(Beta, Y)
    Y
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

  //  protected def feedHiddent(Xt: DenseMatrix, Alfat: DenseMatrix, biases: Array[Double]) = {
  //    val Ht = new DenseMatrix(Alfat.numRows(), Xt.numColumns())
  //    val H = new DenseMatrix(Ht.numColumns(), Ht.numRows())
  //    Alfat.mult(Xt, Ht)
  //    Ht.transpose(H)
  //    addToEachLineOnMatrixAndApplyf(H, biases, sigm2)
  //    H
  //  }
  //
  //  protected def feedHidden(X: DenseMatrix, alfa: DenseVector, bias: Double) = {
  //    val h = new DenseVector(X.numRows())
  //    X.mult(alfa, h)
  //    addAndApplyOnVector(h, bias, sigm2)
  //    h
  //  }


  //  protected def cast(model: Model) = model match {
  //    case m: ELMModel => m
  //    case _ => println("ELM requires ELMModel.")
  //      sys.exit(0)
  //  }
  //
  //
}
