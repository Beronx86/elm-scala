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

import no.uib.cipr.matrix.{DenseMatrix, DenseVector, Matrices}
import util.Lock

import scala.collection.mutable

/**
 * Created by davi on 23/05/14.
 */
object Math extends Lock {
  val readOnly = true
  val hardClose = ()
  val IMap = mutable.LinkedHashMap[Int, DenseMatrix]()
  var fileLocked = false

  def close() = Unit

  def isOpen() = false

  /**
   * The identity matrix is immutable by definition.
   * This function is thread-safe (with global lock).
   * @param n
   */
  def identity(n: Int) = {
    acquire()
    val r = IMap.getOrElse(n, Matrices.identity(n))
    if (IMap.size > 20) IMap.drop(5)
    release()
    r
  }

  def sigm(x: Double) = 1.0 / (1 + math.exp(-x))

  def sigm2(x: Double) = (1.0 / (1 + math.exp(-x)) - 0.5) * 2d //better to avoid numerical instability

  def inv(A: DenseMatrix) = {
    val I = Matrices.identity(A.numRows())
    val tmp = A.copy()
    A.solve(I, tmp)
    tmp
  }

  def applyOnMatrix(M: DenseMatrix, f: Double => Double) {
    val data = M.getData
    var i = 0
    while (i < data.size) {
      data(i) = f(data(i))
      i += 1
    }
  }

  def applyOnVector(V: DenseVector, f: Double => Double) {
    val data = V.getData
    var i = 0
    while (i < data.size) {
      data(i) = f(data(i))
      i += 1
    }
  }

  def addAndApplyOnVector(V: DenseVector, v: Double, f: Double => Double) {
    val data = V.getData
    var i = 0
    while (i < data.size) {
      data(i) = f(data(i) + v)
      i += 1
    }
  }

  def addToEachRowOnMatrixAndApplyf(M: DenseMatrix, a: Array[Double], f: Double => Double) {
    var i = 0
    var j = 0
    while (j < M.numColumns()) {
      val v = a(j)
      i = 0
      while (i < M.numRows()) {
        M.set(i, j, f(M.get(i, j) + v))
        i += 1
      }
      j += 1
    }
  }

  def addToEachColumnOnMatrixAndApplyf(M: DenseMatrix, a: Array[Double], f: Double => Double) {
    var i = 0
    while (i < M.numRows()) {
      var j = 0
      val v = a(i)
      while (j < M.numColumns()) {
        M.set(i, j, f(M.get(i, j) + v))
        j += 1
      }
      i += 1
    }
  }

  def copyToEachColumnOnMatrixAndApplyf(M: DenseMatrix, a: Array[Double], f: Double => Double) {
    var c = 0
    val al = a.size
    val d = M.getData
    val ml = d.size
    while (c < ml) {
      System.arraycopy(a, 0, d, c, al)
      c += al
    }
  }

}

//rnd ok
