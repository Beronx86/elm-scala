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
 * Selects best network from 1 to 1 + deltaL (or N / 2, the lesser) at build.
 * Selects best network from L - deltaL to L + deltaL (or N / 2, the lesser) at update.
 * The N / 2 limit is to avoid NaNs in LOOError calculation during model selection and numerical instability.
 * @param seed
 */
case class interaELM(deltaL: Int = 1000, takePct: Double = 0, seed: Int = 42, notes: String = "") extends interaTrait {
  override val toString = s"interaELM (${takePct * 100}pct)_" + notes

  def modelSelection(model: ELMModel) = {
    //todo: analyse which matrices can be reused along all growing (i.e. they don't change size and need not be kept intact as candidate for the final model)
    val Lini = math.max(Lbuild, model.L - deltaL)
    val Lfim = math.min(model.N / 2, model.L + deltaL)

    var m = cast(buildCore(Lini, model.Xt, model.Y, new XSRandom(seed)))
    val (_, l) = (Lini to Lfim map { L =>
      if (L > Lini) m = growByOne(m)
      val E = errorMatrix(m.H, m.Beta, m.Y)
      //      val press = PRESS(E)(m.HHinv)
      val press = LOOError(m.Y)(E)(m.HHinv)
      //      println(press + " " + L)
      (press, L)
    }).sortBy(_._1).take((takePct * deltaL * 2 + 0.001).ceil.toInt).minBy(_._2)
    val best = cast(if (l != model.L) buildCore(l, model.Xt, model.Y, new XSRandom(seed)) else model)
    best
  }
}