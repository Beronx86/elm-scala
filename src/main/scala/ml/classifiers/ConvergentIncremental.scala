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
import ml.models.{ELMIncModel, Model}
import ml.mtj.DenseMatrix2
import ml.neural.elm.{ConvergentELM, ELMUtils}
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}

trait ConvergentIncremental extends ConvergentELM {
  def update(model: Model, fastAndCheap: Boolean = false)(pattern: Pattern) = {
    val m = cast(model)
    val Alfat = m.Alfat
    val biases = m.biases
    val P0 = m.P
    val rnd = m.rnd
    val Beta0 = m.Beta //LxO
    val x = pattern.arraymtj
    val (h, hm) = ELMUtils.feedHiddenv(x, Alfat, biases) //h: Lx1; H: NxL

    val L = h.size()
    val O = Beta0.numColumns()
    val y = pattern.weighted_label_array //y: Ox1; Y: LxO
    val ym = new DenseMatrix2(y)
    ym.resize(1, O)

    //P1
    val tmpLx1 = new DenseVector(P0.numRows())
    val tmpLx1m = new DenseMatrix(tmpLx1, false)
    P0.mult(h, tmpLx1) //Lx1
    val tmp = h.dot(tmpLx1)
    val factor = -1 / (1 + tmp)
    val P0hht = new DenseMatrix(L, L)
    tmpLx1m.mult(hm, P0hht) //LxL
    val deltaP = new DenseMatrix(L, L)
    P0hht.mult(P0, deltaP) //LxL
    deltaP.scale(factor)
    val P1 = if (fastAndCheap) {
      P0.add(deltaP)
      P0
    } else {
      deltaP.add(P0)
      deltaP
    }

    //Beta1
    val parens = new DenseMatrix(1, O)
    hm.mult(Beta0, parens) //1xO
    parens.scale(-1)
    parens.add(ym)
    deltaP.mult(h, tmpLx1)
    val tmpLxO = new DenseMatrix(L, O)
    tmpLx1m.mult(parens, tmpLxO)
    val Beta1 = if (fastAndCheap) {
      Beta0.add(tmpLxO)
      Beta0
    } else {
      tmpLxO.add(Beta0)
      tmpLxO
    }

    //All of this is useless for OS-only ELM
    val newXt = new DenseMatrix(m.Xt.numRows(), m.Xt.numColumns() + 1)
    System.arraycopy(m.Xt.getData, 0, newXt.getData, 0, m.Xt.getData.size)
    System.arraycopy(x.getData, 0, newXt.getData, m.Xt.getData.size, x.getData.size)
    val newY = new DenseMatrix(m.Y.numRows() + 1, m.Y.numColumns())
    var i = 0
    while (i < m.Y.numRows()) {
      var j = 0
      while (j < m.Y.numColumns()) {
        newY.set(i, j, m.Y.get(i, j))
        j += 1
      }
      i += 1
    }
    var j = 0
    while (j < m.Y.numColumns()) {
      newY.set(i, j, y(j))
      j += 1
    }
    ELMIncModel(rnd, Alfat, biases, Beta1, P1, m.N + 1, newXt, newY)
  }

  //  def updateAll(model: Model, fastAndCheap: Boolean = false)(patterns: Seq[Pattern]) =    patterns.foldLeft(model)((m, p) => update(m, fastAndCheap = true)(p))
}