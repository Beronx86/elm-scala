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
 * Grows network from 1 to Lmax (or N) according to arriving instances.
 * @param Lmax
 * @param seed
 */
case class interaELM(Lmax: Int, take: Double = 0, seed: Int = 42, notes: String = "") extends interaTrait {
  override val toString = s"interaELM (${take * 100}pct)_" + notes

  protected def modelSelection(model: ELMModel) = {
    //todo: analyse which matrices can be reused along all growing (i.e. they don't change size and need not be kept intact as candidate for the final model)

    var m = cast(buildCore(1, model.Xt, model.Y, new XSRandom(seed)))
    val Lfim = math.min(m.N, Lmax)
    val (_, best) = (1 to Lfim map { L =>
      if (L > 1) m = growByOne(m)
      val E = errorMatrix(m.H, m.Beta, m.Y)
      //      val press = PRESS(E)(m.HHinv)
      val press = LOOError(m.Y)(E)(m.HHinv)
      //      println(press + " " + L)
      (press, m)
    }).sortBy(_._1).apply((take * m.N).toInt) //take(66).sortBy(_._2.L).apply(33)
    best
  }
}