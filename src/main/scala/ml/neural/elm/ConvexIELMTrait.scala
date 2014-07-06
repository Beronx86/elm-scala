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
import ml.models.Model
import ml.mtj.ResizableDenseMatrix

/**
 * Created by davi on 21/05/14.
 */
trait ConvexIELMTrait extends IteratedBuildELM {

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = ???

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
}