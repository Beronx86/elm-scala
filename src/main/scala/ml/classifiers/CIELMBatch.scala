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
import ml.neural.elm.{Data, ConvexIELMTrait}
import no.uib.cipr.matrix.{DenseVector, DenseMatrix}
import util.XSRandom
import ml.neural.elm.Data._

/**
 * CIELMBatch
 */
case class CIELMBatch(seed: Int = 42, callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ()) extends ConvexIELMTrait {
   override val toString = "CIELMBatch"
   val id = 8001
   val Lbuild = -1
   val abr = "CIELM"

   def grow(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], t: Vector[DenseVector]) = {
      val natts = X.numColumns()
      val (weights, bias, newRnd) = newNode(natts, rnd)
      val (h, beta) = addNodeForConvexUpdate(weights, bias, X, t, e)
      (weights, bias, h, beta, newRnd)
   }

   override def build(trSet: Seq[Pattern]): Model = {
      val nclasses = trSet.head.nclasses
      val n = trSet.size
      if (trSet.size < nclasses) {
         println("At least |Y| instances required.")
         sys.exit(1)
      }
      val natts = trSet.head.nattributes
      val X = patterns2matrix(trSet, n)
      val (t, e) = patterns2te(trSet, n)
      bareBuild(n, n, natts, nclasses, X, e, t)
   }

   override def update(model: Model, fast_mutable: Boolean = false)(pattern: Pattern) = {
      val m = cast(model)
      val newE = m.e.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
      val newT = m.t.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
      val newX = Data.appendRowToMatrix(m.X, pattern.array)
      bareBuild(m.N + 1, m.N + 1, newX.numColumns(), pattern.nclasses, newX, newE, newT)
   }
}
