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

import ml.models.ELMModel

/**
 * Grows network from 1.
 * L changes monotonically.
 * @param deltaL
 * @param seed
 */
case class interawfELM(deltaL: Int, seed: Int = 42, notes: String = "") extends interaTrait {
  override val toString = s"interawfELM d${deltaL}_" + notes

  protected def modelSelection(model: ELMModel) = {
    //todo: analyse which matrices can be reused along all growing (i.e. they don't change size and need not be kept intact as candidate for the final model)
    val previousE = errorMatrix(model.H, model.Beta, model.Y)
    val previousError = LOOError(model.Y)(previousE)(model.HHinv) //PRESS(previousE)(HHinv)

    var m = model
    val min = m.L
    val max = math.min(m.L + deltaL, m.N)
    val (_, best) = (previousError, model) +: ((min to max) map { L =>
      if (L > min) m = growByOne(m)
      val E = errorMatrix(m.H, m.Beta, m.Y)
      val press = LOOError(m.Y)(E)(m.HHinv) //PRESS(E)(HHinv)
      (press, m)
    }) minBy (_._1)
    best
  }
}
