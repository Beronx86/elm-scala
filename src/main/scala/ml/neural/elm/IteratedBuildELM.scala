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

import ml.classifiers.Learner
import ml.models.{ELMOnlineModel, Model}
import no.uib.cipr.matrix.{Matrices, DenseVector, DenseMatrix}
import ml.Pattern
import ml.neural.elm.Math._
import scala.util.Random
import ml.mtj.{ResizableDenseMatrix, DenseMatrix2}
import ml.neural.elm.Data._

/**
 * Created by davi on 21/05/14.
 */
trait IteratedBuildELM extends BatchELM {//todo: correct this, extending BatchELM is very inefficient!
  val f: (ELMOnlineModel, Double) => Unit
  val initialL: Int
  val callf: Boolean

  /**
   * mutates rnd!
   * @param natts
   * @param rnd
   * @return
   */
  protected def newNode(natts: Int, rnd: Random) = {
    val w = Array.fill(natts)(rnd.nextDouble() * 2 - 1)
    (w, rnd.nextDouble() * 2 - 1)
  }
}