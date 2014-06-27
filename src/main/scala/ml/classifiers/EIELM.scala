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

import ml.neural.elm.{IELMTrait, ELM}
import ml.Pattern
import ml.models.{ELMGenericModel, Model}
import ml.neural.elm.Data._
import ml.mtj.ResizableDenseMatrix
import no.uib.cipr.matrix.{DenseVector, DenseMatrix}
import util.XSRandom
import scala.util.Random

/**
 * Created by davi on 24/05/14.
 */
case class EIELM(Lbuild: Int, seed: Int = 42, size: Int = 1, callf: Boolean = false, f: (ELMGenericModel, Double) => Unit = (tmp: Model, tmpt: Double) => ())
  extends IELMTrait {
  override val toString = "EIELM"
  val CANDIDATES = 10

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Array[DenseVector], tmp: DenseVector) = createNodeAmongCandidates(rnd, X, e, tmp)

  /**
   * Mutate e, tmp, tmp2 and rnd
   * @return
   */
  def createNodeAmongCandidates(rnd: XSRandom, X: DenseMatrix, e: Array[DenseVector], tmp: DenseVector) = {
    val nclasses = e.size
    val natts = X.numColumns()
    val ninsts = X.numRows()
    val candidates = 0 until CANDIDATES map { idx =>
      val newe = Array.fill(nclasses)(new DenseVector(ninsts))
      val (weights, bias, newRnd) = newNode(natts, rnd)
      rnd.setSeed(newRnd.getSeed)
      val alfa = new DenseVector(weights, false)
      val beta = new Array[Double](nclasses)
      val h = feedHidden(X, alfa, bias)
      var o = 0
      while (o < nclasses) {
        tmp.set(h)
        val neweo = newe(o)
        neweo.set(e(o))

        //Calculate new weight.
        val nume = neweo.dot(h)
        val deno = h.dot(h)
        val b = nume / deno
        beta(o) = b

        //Recalculate residual error.
        tmp.scale(-b)
        neweo.add(tmp)

        o += 1
      }
      //todo: check if this is the correct way to aggregate multiclass errors and propagate to CIELM or others
      val err = newe.map { x =>
        val sq = x.dot(x)
        sq * sq
      }.sum
      (err, weights, bias, h, beta, newe)
    }
    val (_, weights, bias, h, beta, newe) = candidates.minBy(_._1)
    e.zip(newe).foreach { case (a, b) => a.set(b)}
    (weights, bias, h, beta)
  }
}//rnd ok
