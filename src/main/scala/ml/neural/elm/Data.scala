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
import no.uib.cipr.matrix.{DenseVector, DenseMatrix}

/**
 * Created by davi on 23/05/14.
 */
object Data {

  def patterns2matrices(insts: Seq[Pattern], ninsts: Int) = {
    val natts = insts.head.nattributes
    val nclasses = insts.head.nclasses
    val Xt = new DenseMatrix(natts, ninsts)
    val Y = new DenseMatrix(ninsts, nclasses)
    var i = 0
    insts foreach {
      inst =>
        val arr = inst.array
        val larr = inst.weighted_label_array
        var j = 0
        while (j < natts) {
          Xt.set(j, i, arr(j))
          j += 1
        }
        j = 0
        while (j < nclasses) {
          Y.set(i, j, larr(j))
          j += 1
        }
        i += 1
    }
    (Xt, Y)
  }

  def patterns2matricest(insts: Seq[Pattern], ninsts: Int) = {
    val natts = insts.head.nattributes
    val nclasses = insts.head.nclasses
    val X = new DenseMatrix(ninsts, natts)
    val Yt = new DenseMatrix(nclasses, ninsts)
    var i = 0
    insts foreach {
      inst =>
        val arr = inst.array
        val larr = inst.weighted_label_array
        var j = 0
        while (j < natts) {
          X.set(i, j, arr(j))
          j += 1
        }
        j = 0
        while (j < nclasses) {
          Yt.set(j, i, larr(j))
          j += 1
        }
        i += 1
    }
    (X, Yt)
  }

  def patterns2matrix(insts: Seq[Pattern], ninsts: Int) = {
    val natts = insts.head.nattributes
    val X = new DenseMatrix(ninsts, natts)
    var i = 0
    insts foreach {
      inst =>
        val arr = inst.array
        //        val larr = inst.weighted_label_array
        var j = 0
        while (j < natts) {
          X.set(i, j, arr(j))
          j += 1
        }
        i += 1
    }
    X
  }

  def patterns2te(insts: Seq[Pattern], ninsts: Int) = {
    val nclasses = insts.head.nclasses
    val t = Vector.fill(nclasses)(new DenseVector(ninsts))
    val e = Vector.fill(nclasses)(new DenseVector(ninsts))
    0 until nclasses foreach { o =>
      var i = 0
      while (i < ninsts) {
        t(o).set(i, insts(i).weighted_label_array(o))
        i += 1
      }
      System.arraycopy(t(o).getData, 0, e(o).getData, 0, ninsts)
    }
    (t, e)
  }

  def patterns2t(insts: Seq[Pattern], ninsts: Int) = {
    val nclasses = insts.head.nclasses
    val t = Vector.fill(nclasses)(new DenseVector(ninsts))
    0 until nclasses foreach { o =>
      var i = 0
      while (i < ninsts) {
        t(o).set(i, insts(i).weighted_label_array(o))
        i += 1
      }
    }
    t
  }

  def appendToVector(vec: DenseVector, v: Double) = {
    val s = vec.size()
    val newVec = new DenseVector(s + 1)
    System.arraycopy(vec.getData, 0, newVec.getData, 0, s)
    newVec.getData()(s) = v
    newVec
  }

  def appendToArray(a: Array[Double], v: Double) = {
    val s = a.size
    val newA = new Array[Double](s + 1)
    System.arraycopy(a, 0, newA, 0, s)
    newA(s) = v
    newA
  }

  def appendRowToMatrix(m: DenseMatrix, vec: Array[Double]) = {
    val w = m.numColumns()
    val h = m.numRows()
    val newM = new DenseMatrix(h + 1, w)
    val d0 = m.getData
    val d1 = newM.getData
    var c = 0
    while (c < w) {
      val d = c * (h + 1)
      System.arraycopy(d0, c * h, d1, d, h)
      d1(d + h) = vec(c)
      c += 1
    }
    newM
  }

}
