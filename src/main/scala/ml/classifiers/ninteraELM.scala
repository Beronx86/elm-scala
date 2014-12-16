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
import ml.models.{ELMIncModel, ELMModel, Model}
import util.XSRandom

case class ninteraELM(seed: Int = 42, deltaL: Int = 10) extends interaTrait {
   override val toString = s"ninteraELM (+-$deltaL)"
   val id = if (deltaL == 10) 11 else throw new Error("Parametros inesperados para interaELM.")
   val abr = "nintera"

   override def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
      val m = super.update(model)(pattern)
      if (m.N % 4 == 0) {
         val gm = modelSelection(m)
         ELMIncModel(gm.rnd, gm.Alfat, gm.biases, gm.Beta, gm.P, gm.N, gm.Xt, gm.Y)
      } else m
   }

   def modelSelection(model: ELMModel) = {
      //todo: analyse which matrices can be reused along all growing (i.e. they don't change size and need not be kept intact as candidate for the final model)
      val Lini = math.max(Lbuild, model.L - deltaL)
      val Lfim = math.min(model.N / 2, model.L + deltaL)

      var m = cast(buildCore(Lini, model.Xt, model.Y, new XSRandom(seed)))
      val (_, l) = (Lini to Lfim map { L =>
         if (L > Lini) m = growByOne(m)
         val E = errorMatrix(m.H, m.Beta, m.Y)
         val press = PRESS(E)(m.HHinv)
         //      val press = LOOError(m.Y)(E)(m.HHinv)
         //      println(press + " " + L)
         (press, L)
      }).sortBy(_._1).head
      val best = cast(if (l != model.L) buildCore(l, model.Xt, model.Y, new XSRandom(seed)) else model)
      best
   }
}