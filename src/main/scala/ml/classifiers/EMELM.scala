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
import ml.models.Model

/**
 * Grows network from 1 to Lmax. *
 * build() não é continuável, isto é, ele não simula internamente um modelo incremental.
 * @param seed
 */
// todo: It performs batch learning, i.e. retrains from scratch to accomodate new instances.
case class EMELM(Lbuild: Int, seed: Int = 42) extends ConvergentGrowing {
   override val toString = "EMELM"
   val id = -8
   val abr = toString

   override def build(trSet: Seq[Pattern]): Model = batchBuild(trSet)

   def update(model: Model, fast_mutable: Boolean, semcrescer: Boolean = false)(pattern: Pattern) = {
      println("EM-ELM does not accept update() for now. It could perform an entire rebuild.")
      sys.exit(1)
   }
}