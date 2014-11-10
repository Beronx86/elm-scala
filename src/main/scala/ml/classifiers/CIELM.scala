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
import ml.models.{ELMSimpleModel, Model}
import ml.neural.elm.Data._
import ml.neural.elm.{ConvexIELMTrait, Data}
import no.uib.cipr.matrix.{ResizableDenseMatrix, DenseMatrix, DenseVector}
import util.{Datasets, Tempo, XSRandom}

import scala.util.Random

/**
 * CI-ELM
 * Created by davi on 19/05/14.
 */
case class CIELM(seed: Int = 42, callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ()) extends ConvexIELMTrait {
  override val toString = "CIELM"
  val id = 8
  val Lbuild = -1
  val abr = toString

  /**
   * Não altera rnd.
   * @param rnd
   * @param X
   * @param e
   * @param t
   * @return
   */
  def grow(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], t: Vector[DenseVector]) = {
    val natts = X.numColumns()

    val (weights, bias, newRnd) = newNode(natts, rnd)
    val (h, beta) = addNodeForConvexUpdate(weights, bias, X, t, e)

    (weights, bias, h, beta, newRnd)
  }
}
