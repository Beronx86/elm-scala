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
import ml.models.ELMSimpleModel
import ml.mtj.ResizableDenseMatrix
import ml.neural.elm.Data._
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.{Tempo, XSRandom}

/**
 * Created by davi on 21/05/14.
 */
trait IELMTrait extends IteratedBuildELM {

  def build(trSet: Seq[Pattern]) = {
    val rnd = new XSRandom(seed)
    Tempo.start
    val ninsts = checkEmptyness(trSet: Seq[Pattern])
    val natts = trSet.head.nattributes
    val nclasses = trSet.head.nclasses
    val X = patterns2matrix(trSet, ninsts)
    val biases = Array.fill(Lbuild)(0d)
    val Alfat = new ResizableDenseMatrix(Lbuild, natts)
    val Beta = new ResizableDenseMatrix(Lbuild, nclasses)
    val H = new ResizableDenseMatrix(ninsts, Lbuild)
    H.resizeCols(0)
    val e = patterns2t(trSet, ninsts)
    var l = 0
    val tmp = new DenseVector(ninsts)
    if (callf) {
      Alfat.resizeRows(0)
      Beta.resizeRows(0)
      while (l < Lbuild) {
        val (weights, bias, h, beta) = buildCore(rnd, X, e, tmp)
        biases(l) = bias
        l += 1
        Alfat.addRow(weights)
        Beta.addRow(beta)
        val te = Tempo.stop
        Tempo.start
        f(ELMSimpleModel(rnd.clone(), Alfat, biases, Beta, ninsts), te)
      }
    } else {
      while (l < Lbuild) {
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
    ELMSimpleModel(rnd, Alfat, biases, Beta, ninsts) //todo: se nao crescer, manter P e H anteriores?
  }

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Array[DenseVector], tmp: DenseVector): (Array[Double], Double, DenseVector, Array[Double])

  //
  //
  //

}

//rnd ok