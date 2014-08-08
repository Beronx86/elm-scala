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
import ml.models.{ELMModel, ELMIncModel, Model}
import util.XSRandom

/**
 * Grows network from 1.
 * L changes monotonically.
 * Attempts at each new instance.
 * @param deltaL
 * @param seed
 */
case class interawfAlwaysELM(deltaL: Int, seed: Int = 42, notes: String = "") extends interaTrait {
  override val toString = s"interawfAlwaysELM d${deltaL}_" + notes

  override def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val m = super.update(model)(pattern)
    val gm = modelSelection(m)
    ELMIncModel(gm.rnd, gm.Alfat, gm.biases, gm.Beta, gm.P, gm.N, gm.Xt, gm.Y)
  }

  protected def modelSelection(model: ELMModel) = {
    //todo: analyse which matrices can be reused along all growing (i.e. they don't change size and need not be kept intact as candidate for the final model)
    var m = model
    val min = m.L
    val max = math.min(m.L + deltaL, m.N)
    val (_, best) = (min to max) map { L =>
      if (L > min) m = growByOne(m)
      val E = errorMatrix(m.H, m.Beta, m.Y)
      val press = LOOError(m.Y)(E)(m.HHinv) //PRESS(E)(HHinv)
      (press, m)
    } minBy (_._1)
    best
  }
}
