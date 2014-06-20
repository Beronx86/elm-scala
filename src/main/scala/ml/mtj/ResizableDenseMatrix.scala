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
package ml.mtj

import no.uib.cipr.matrix.{DenseVector, DenseMatrix}

class ResizableDenseMatrix(maxRowns: Int, maxColumns: Int) extends DenseMatrix(maxRowns, maxColumns) {
  def setRow(i: Int, a: Array[Double]) {
    var j = 0
    while (j < numColumns) {
      set(i, j, a(j))
      j += 1
    }
  }

  //  def setData(data:Array[Double]) {
  //    this.data
  //  }
  def addCol(col: Array[Double]) {
    numColumns += 1
    var i = 0
    while (i < numRows) {
      set(i, numColumns - 1, col(i))
      i += 1
    }
  }

  def copyTo(m: DenseMatrix, rows: Int = numRows) {
    var i = 0
    while (i < rows) {
      var j = 0
      while (j < numColumns) {
        m.set(i, j, this.get(i, j))
        j += 1
      }
      i += 1
    }
  }

  def addRow(row: Array[Double]) {
    val m = new DenseMatrix(numRows + 1, numColumns)
    copyTo(m)
    var j = 0
    while (j < numColumns) {
      m.set(numRows, j, row(j))
      j += 1
    }

    numRows += 1
    System.arraycopy(m.getData, 0, this.getData, 0, math.min(m.getData.size, getData.size))
  }

  def addExtraRow(row: Array[Double]) = {
    val m = new DenseMatrix(numRows + 1, numColumns)
    copyTo(m)
    var j = 0
    while (j < numColumns) {
      m.set(numRows, j, row(j))
      j += 1
    }
    m
  }

  def addCol(col: DenseVector) {
    this.addCol(col.getData)
  }

  def addRow(row: DenseVector) {
    this.addRow(row.getData)
  }

  def resizeRows(rows: Int) {
    val m = new DenseMatrix(rows, numColumns)
    copyTo(m, math.min(rows, numRows))
    numRows = rows
    System.arraycopy(m.getData, 0, this.getData, 0, math.min(m.getData.size, getData.size))
  }

  def resizeCols(columns: Int) {
    numColumns = columns
  }


  def resize(rows: Int, columns: Int) {
    numRows = rows
    numColumns = columns
  }
}
