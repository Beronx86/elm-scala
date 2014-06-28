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
import ml.neural.elm.{ELMUtils, Math}
import no.uib.cipr.matrix.DenseMatrix
import util.XSRandom

trait ELMModel extends Model {
  val rnd: XSRandom
  //needed by test()
  val biases: Array[Double]
  val Alfat: DenseMatrix
  val Beta: DenseMatrix

  val H: DenseMatrix

  //needed by addNeuron()
  val Xt: DenseMatrix

  val N: Int
  lazy val L = Alfat.numRows()
  lazy val I = Math.identity(N)

  def distribution(pattern: Pattern) = ELMUtils.distribution(ELMUtils.test(pattern, Alfat, biases, Beta))
}

case class ELMSimpleModel(rnd: XSRandom, Alfat: DenseMatrix, biases: Array[Double], Beta: DenseMatrix, N: Int) extends ELMModel

case class ELMIncModel(rnd: XSRandom, Alfat: DenseMatrix, biases: Array[Double], Beta: DenseMatrix,
                       P: DenseMatrix, N: Int) extends ELMModel {
  lazy val H = {
    val m = new DenseMatrix(N, Alfat.numRows())
    X.

  }

}

case class ELMGroModel(rnd: XSRandom, Alfat: DenseMatrix, biases: Array[Double], Beta: DenseMatrix,
                       H: DenseMatrix, X: DenseMatrix, Y: DenseMatrix, Hinv: DenseMatrix) extends ELMModel {
  val N = H.numRows()
  lazy val HHinv = {
    val r = new DenseMatrix(H.numRows(), H.numRows())
    H.mult(Hinv, r)
    r
  }
}
