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
package ml.neural.elm

import ml.Pattern
import ml.models.{BatchModel, Model}

/**
 * A call to update performs total rebuild.
 * Created by davi on 21/05/14.
 */
trait BatchELM extends ELM {
  private def cast2batmodel(model: Model) = model match {
    case m: BatchModel => m
    case _ => throw new Exception("BatchLearner requires BatchModel.")
  }

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = build(pattern +: cast2batmodel(model).training_set)

  def updateAll(model: Model, fast_mutable: Boolean)(patterns: Seq[Pattern]) = build(patterns ++ cast2batmodel(model).training_set)
}