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
import ml.models.{Model, ELMSimpleModel}
import ml.neural.elm.Data._
import no.uib.cipr.matrix.{DenseVector, DenseMatrix, ResizableDenseMatrix}
import util.XSRandom

/**
 * Created by davi on 21/05/14.
 */
trait ConvexIELMTrait extends IteratedBuildELM {
   def updateNetwork(l: Int, weights: Array[Double], beta: Array[Double], Beta: ResizableDenseMatrix, Alfat: ResizableDenseMatrix) {
      val nclasses = Beta.numColumns()
      val natts = Alfat.numColumns()
      var o = 0
      while (o < nclasses) {
         var lesserThanl = 0
         while (lesserThanl < l) {
            Beta.set(lesserThanl, o, Beta.get(lesserThanl, o) * (1 - beta(o)))
            lesserThanl += 1
         }
         Beta.set(lesserThanl, o, beta(o))
         o += 1
      }

      var i = 0
      while (i < natts) {
         Alfat.set(l, i, weights(i))
         i += 1
      }
   }

   def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
      val m = cast(model)
      val newE = m.e.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
      val newT = m.t.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
      val newX = Data.appendRowToMatrix(m.X, pattern.array)

      val (weights, bias, h, beta, newRnd) = grow(m.rnd, newX, newE, newT)

      val newAlfat = Data.appendRowToMatrix(m.Alfat, weights)
      val newBiases = Data.appendToArray(m.biases, bias)
      val newBeta = Data.appendRowToMatrix(m.Beta, beta)
      ELMSimpleModel(newRnd, newAlfat, newBiases, newBeta, newX, newE, newT)
   }

   def build(trSet: Seq[Pattern]): Model = {
      val nclasses = trSet.head.nclasses
      if (trSet.size < nclasses) {
         println("At least |Y| instances required.")
         sys.exit(1)
      }
      val initialTrSet = trSet.take(nclasses)
      val natts = initialTrSet.head.nattributes
      val X = patterns2matrix(initialTrSet, nclasses)
      val (t, e) = patterns2te(initialTrSet, nclasses)
      val firstModel = bareBuild(nclasses, nclasses, natts, nclasses, X, e, t)
      trSet.drop(nclasses).foldLeft(firstModel)((m, p) => cast(update(m, fast_mutable = true)(p)))
   }

   def grow(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], t: Vector[DenseVector]): (Array[Double], Double, DenseVector, Array[Double], XSRandom)

   def bareBuild(L: Int, ninsts: Int, natts: Int, nclasses: Int, X: DenseMatrix, e: Vector[DenseVector], t: Vector[DenseVector]) = {
      val biases = Array.fill(L)(0d)
      val Alfat = new ResizableDenseMatrix(L, natts)
      val Beta = new ResizableDenseMatrix(L, nclasses)
      var l = 0
      var rnd = new XSRandom(seed)
      while (l < L) {
         val (weights, b, h, beta, newRnd) = grow(rnd, X, e, t)
         rnd = newRnd
         biases(l) = b
         updateNetwork(l, weights, beta, Beta, Alfat)
         l += 1
      }
      ELMSimpleModel(rnd, Alfat, biases, Beta, X, e, t)
   }

   def addNodeForConvexUpdate(weights: Array[Double], bias: Double, X: DenseMatrix, t: Vector[DenseVector], e: Vector[DenseVector]) = {
      val nclasses = t.size
      //Generate node and calculate h.
      val alfa = new DenseVector(weights, false)
      val beta = new Array[Double](nclasses)
      val h = feedHidden(X, alfa, bias)
      val hneg = h.copy()
      hneg.scale(-1)
      var o = 0
      while (o < nclasses) {
         //Calculate new weight.
         val t_h = t(o).copy()
         t_h.add(hneg)
         val a = t_h.copy()
         a.scale(-1)
         a.add(e(o))
         val nume = e(o).dot(a)
         val deno = a.dot(a)
         val b = nume / deno
         beta(o) = b

         //Recalculate residual error.
         e(o).scale(1 - b)
         t_h.scale(b)
         e(o).add(t_h)

         o += 1
      }
      (h, beta)
   }
}