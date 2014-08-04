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
 * Grows network from 1 to Lmax according to arriving instances.
 * @param Lmax
 * @param seed
 */
case class interaNewSeedELM(Lmax: Int, seed: Int = 42, notes: String = "") extends interaTrait {
  override val toString = "interaELMNewSeed_" + notes

  protected def modelSelection(model: ELMModel) = {
    val previousE = errorMatrix(model.H, model.Beta, model.Y)
    val previousError = LOOError(model.Y)(previousE)(model.HHinv) //PRESS(previousE)(HHinv)

    val rnd = model.rnd.clone()
    var m = cast(buildCore(1, model.Xt, model.Y, rnd))
    val (_, best) = (previousError, model) +: (1 to math.min(m.N / 2, Lmax) map { L =>
      if (L > 1) m = growByOne(m)
      val E = errorMatrix(m.H, m.Beta, m.Y)
      val press = LOOError(m.Y)(E)(m.HHinv) //PRESS(E)(HHinv)
      (press, m)
    }) minBy (_._1)
    best
  }
}