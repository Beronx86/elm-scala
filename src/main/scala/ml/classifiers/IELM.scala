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

import ml.neural.elm.{IteratedBuildELM, IELMTrait, ELM}
import ml.Pattern
import ml.models.Model
import ml.neural.elm.Data._
import ml.mtj.ResizableDenseMatrix
import no.uib.cipr.matrix.{DenseVector, DenseMatrix}
import util.XSRandom
import scala.util.Random

/**
 * Created by davi on 24/05/14.
 */
case class IELM(Lbuild: Int, seed: Int = 42, callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ())
  extends IELMTrait {
  override val toString = "IELM"

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Array[DenseVector], tmp: DenseVector) = {
    val (weights, bias, newRnd) = newNode(X.numColumns(), rnd)
    rnd.setSeed(newRnd.getSeed)
    val (h, beta) = addNode(weights, bias, X, e, tmp)
    (weights, bias, h, beta)
  }

  def addNode(weights: Array[Double], bias: Double, X: DenseMatrix, e: Array[DenseVector], tmp: DenseVector) = {
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
}//rnd ok
