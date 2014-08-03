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
import util.XSRandom

/**
 * Grows network from 1 to Lmax according to arriving instances.
 * Rebuilds from scratch. Should be equal interaELM in accuracy.
 * @param Lmax
 * @param seed
 */
case class interaELMNoEM(Lmax: Int, seed: Int = 42, notes: String = "") extends interaTrait {
  override val toString = "interaELMNoEM_" + notes

  protected def modelSelection(model: ELMModel) = {
    //todo: analyse which matrices can be reused along all growing (i.e. they don't change size and need not be kept intact as candidates for the final model)
    var m = model
    val (_, best) = (1 to math.min(m.N / 2, Lmax) map { L =>
      if (L > 1) m = buildCore(L, m.Xt, m.Y, new XSRandom(seed))
      val E = errorMatrix(m.H, m.Beta, m.Y)
      val press = LOOError(m.Y)(E)(m.HHinv)
      //      val press = PRESS(E)(m.HHinv)
      (press, m)
    }) minBy (_._1)
    best
  }
}