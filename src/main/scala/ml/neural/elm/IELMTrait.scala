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

import ml.models.{ELMOnlineModel, Model}
import no.uib.cipr.matrix.{DenseVector, DenseMatrix}
import ml.Pattern
import ml.neural.elm.Math._
import scala.util.Random
import ml.mtj.{ResizableDenseMatrix, DenseMatrix2}
import ml.neural.elm.Data._
import util.Tempo

/**
 * Created by davi on 21/05/14.
 */
trait IELMTrait extends IteratedBuildELM {
//  val f: (ELMModel, Double) => Unit
//  val L: Int
//  val callf: Boolean
//
  def build(trSet: Seq[Pattern]) = {
    val rnd = new Random(seed)
    Tempo.start
    val ninsts = checkEmptyness(trSet: Seq[Pattern])
    val natts = trSet.head.nattributes
    val nclasses = trSet.head.nclasses
    val X = patterns2matrix(trSet,ninsts)
    val biases = Array.fill(initialL)(0d)
    val Alfat = new ResizableDenseMatrix(initialL, natts)
    val Beta = new ResizableDenseMatrix(initialL, nclasses)
    val H = new ResizableDenseMatrix(ninsts, initialL)
    H.resizeCols(0)
    val e = patterns2t(trSet,ninsts)
    var l = 0
    val tmp = new DenseVector(ninsts)
    if (callf) {
      Alfat.resizeRows(0)
      Beta.resizeRows(0)
      while (l < initialL) {
        val (weights, bias, h, beta) = buildCore(rnd, X, e, tmp)
        biases(l) = bias
        l += 1
        Alfat.addRow(weights)
        Beta.addRow(beta)
        val te = Tempo.stop
        Tempo.start
        f(ELMOnlineModel(rnd, Alfat, biases, null, null, Beta), te)
      }
    } else {
      while (l < initialL) {
        val (weights, bias, h, beta) = buildCore(rnd, X, e, tmp)
        H.addCol(h) //vantagem do I-ELM: H vem pronto e não muda
        biases(l) = bias
        Alfat.setRow(l, weights)
        Beta.setRow(l, beta)
        l += 1
      }
      Alfat.resizeRows(l)
      Beta.resizeRows(l)
    }
    ELMOnlineModel(rnd, Alfat, biases, H, null, Beta) //todo: se nao crescer, manter P e H anteriores?
  }

  protected def buildCore(rnd: Random, X: DenseMatrix, e: Array[DenseVector], tmp: DenseVector): (Array[Double], Double, DenseVector, Array[Double])

//
//
//

}