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
import ml.models.{ELMGenericModel, Model}
import ml.neural.elm.ConvergentELM
import ml.neural.elm.Math._
import no.uib.cipr.matrix.{DenseMatrix, DenseVector, Matrices}
import util.XSRandom

/**
 * Grows network from 1 to Lmax.
 * It performs batch learning, i.e. retrains from scratch to accomodate new instances.
 * @param Lmax
 * @param seed
 */
case class EMELM(Lmax: Int, seed: Int = 42) extends ConvergentGrowing {
  override val toString = "EMELM"
  val Lbuild = 1

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = ???

  def updateAll(model: Model, fast_mutable: Boolean)(patterns: Seq[Pattern]) = ???
}