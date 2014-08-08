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
import ml.models.{ELMSimpleModel, Model}
import ml.neural.elm.ConvexIELMTrait
import ml.neural.elm.Data._
import no.uib.cipr.matrix.{ResizableDenseMatrix, DenseMatrix, DenseVector}
import util.{Tempo, XSRandom}

/**
 * CI-ELM com seleção de nós candidatos
 * @param seed
 * @param notes
 * @param callf
 * @param f
 */
case class ECIELM(seed: Int = 42, notes: String = "", callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ()) extends ConvexIELMTrait {
  override val toString = "ECIELM_" + notes
  val CANDIDATES = 10
  val Lbuild = -1

  def grow(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], t: Vector[DenseVector]) = {
    val tmp = new DenseVector(X.numRows())
    val tmp2 = new DenseVector(X.numRows())
    val (newRnd, weights, b, h, beta) = createNodeForConvexUpdateAmongCandidates(rnd, X, t, e, tmp, tmp2)
    (weights, b, h, beta, newRnd)
  }

  def createNodeForConvexUpdateAmongCandidates(rnd: XSRandom, X: DenseMatrix, t: Vector[DenseVector], e: Vector[DenseVector], tmp: DenseVector, tmp2: DenseVector) = {
    val newRnd = rnd.clone()
    val nclasses = t.size
    val natts = X.numColumns()
    val ninsts = X.numRows()
    val candidates = 0 until CANDIDATES map { idx =>
      val newe = Array.fill(nclasses)(new DenseVector(ninsts))
      val (weights, bias, tmpRnd) = newNode(natts, newRnd)
      newRnd.setSeed(tmpRnd.getSeed)

      val alfa = new DenseVector(weights, false)
      val beta = new Array[Double](nclasses)
      val h = feedHidden(X, alfa, bias)
      val hneg = h.copy()
      hneg.scale(-1)
      var o = 0
      while (o < nclasses) {
        val neweo = newe(o)
        neweo.set(e(o))

        //Calculate new weight.
        tmp.set(t(o))
        tmp.add(hneg)
        tmp2.set(tmp)
        tmp2.scale(-1)
        tmp2.add(neweo)
        val nume = neweo.dot(tmp2)
        val deno = tmp2.dot(tmp2)
        val b = nume / deno
        beta(o) = b

        //Recalculate residual error.
        neweo.scale(1 - b)
        tmp.scale(b)
        neweo.add(tmp)

        o += 1
      }
      val err = newe.map(x => math.sqrt(x.dot(x))).sum
      (err, weights, bias, h, beta, newe)
    }
    val (_, weights, bias, h, beta, newe) = candidates.minBy(_._1)
    //    println("-------xxxxx--------" + candidates.map(_._1))
    e.zip(newe).foreach { case (a, b) => a.set(b)}
    (newRnd, weights, bias, h, beta)
  }
}