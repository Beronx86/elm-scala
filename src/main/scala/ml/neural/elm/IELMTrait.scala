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

import ml.Pattern
import ml.models.ELMSimpleModel
import ml.neural.elm.Data._
import no.uib.cipr.matrix.{ResizableDenseMatrix, DenseMatrix, DenseVector}
import util.XSRandom

/**
 * Created by davi on 21/05/14.
 */
trait IELMTrait extends IteratedBuildELM {

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector): (Array[Double], Double, DenseVector, Array[Double])

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

  def bareBuild(ninsts: Int, natts: Int, nclasses: Int, X: DenseMatrix, e: Vector[DenseVector]) = {
    val L = ninsts
    val biases = Array.fill(L)(0d)
    val Alfat = new ResizableDenseMatrix(L, natts)
    val Beta = new ResizableDenseMatrix(L, nclasses)
    val rnd = new XSRandom(seed)
    var l = 0
    val tmp = new DenseVector(ninsts)
    if (callf) {
      Alfat.resizeRows(0)
      Beta.resizeRows(0)
      while (l < L) {
        val (weights, bias, h, beta) = buildCore(rnd, X, e, tmp)
        biases(l) = bias
        l += 1
        Alfat.addRow(weights)
        Beta.addRow(beta)
        f(ELMSimpleModel(rnd.clone(), Alfat, biases, Beta, X, e, null), -1) //todo: measure time, instead of -1
      }
    } else {
      while (l < L) {
        val (weights, bias, h, beta) = buildCore(rnd, X, e, tmp)
        //        H.addCol(h) //vantagem do I-ELM: H vem pronto e não muda
        biases(l) = bias
        Alfat.setRow(l, weights)
        Beta.setRow(l, beta)
        l += 1
      }
    }
    ELMSimpleModel(rnd, Alfat, biases, Beta, X, e, null) //todo: se nao crescer, manter P e H anteriores?
  }

  def build(trSet: Seq[Pattern]) = {
    val nclasses = trSet.head.nclasses
    if (trSet.size < nclasses) {
      println("At least |Y| instances required.")
      sys.exit(1)
    }
    val initialTrSet = trSet.take(nclasses)
    val natts = initialTrSet.head.nattributes
    val X = patterns2matrix(initialTrSet, nclasses)
    val e = patterns2t(initialTrSet, nclasses)
    val firstModel = bareBuild(nclasses, natts, nclasses, X, e)
    trSet.drop(nclasses).foldLeft(firstModel)((m, p) => cast(update(m, fast_mutable = true)(p)))
  }

}