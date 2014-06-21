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
package ml.models

import ml.Pattern
import no.uib.cipr.matrix.{DenseVector, DenseMatrix}
import ml.neural.elm.{ELMUtils, ELM}
import scala.util.Random

trait ELMModel extends Model {
  val rnd: Random
  val Alfat: DenseMatrix
  val biases: Array[Double]
  val H: DenseMatrix
  val PReady: DenseMatrix
  val Beta: DenseMatrix
  lazy val P = if (PReady == null) ELMUtils.calculateP(H) else PReady

  def distribution(pattern: Pattern) = ELMUtils.distribution(ELMUtils.test(pattern, Alfat, biases, Beta))
}

case class ELMOnlineModel(rnd: Random, Alfat: DenseMatrix, biases: Array[Double], H: DenseMatrix, PReady: DenseMatrix, Beta: DenseMatrix)
  extends ELMModel

case class ELMBatchModel(rnd: Random, Alfat: DenseMatrix, biases: Array[Double], H: DenseMatrix, PReady: DenseMatrix, Beta: DenseMatrix, training_set: Vector[Pattern])
  extends ELMModel