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
 * @param Lmax
 * @param seed
 */
case class interaELMreg(Lmax: Int, seed: Int = 42, notes: String = "") extends interaTrait {
  override val toString = "interaELMreg_" + notes

  protected def modelSelection(model: ELMModel) = {
    //todo: analyse which matrices can be reused along all growing (i.e. they don't change size and need not be kept intact as candidate for the final model)

    var m = cast(buildCore(1, model.Xt, model.Y, new XSRandom(seed)))
    val oldL = m.L
    val Lfim = math.min(m.N / 2, Lmax)
    val (_, best) = (1 to Lfim map { L =>
      if (L > 1) m = growByOne(m)
      val E = errorMatrix(m.H, m.Beta, m.Y)
      val press = PRESS(E)(m.HHinv)
      //      val press = LOOError(m.Y)(E)(m.HHinv)
      val l = (L - oldL).abs / m.N.toDouble
      (press + l, m)
    }).sortBy(_._1).head //.take(2).minBy(_._2.L)
    best
  }
}