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

class DenseMatrix2(data: Array[Double]) extends DenseMatrix(new DenseVector(data, false), false) {
  //  def setData(data:Array[Double]) {
  //    this.data
  //  }
  def addCol(col: DenseVector) = {
    var i = 0
    while (i < numRows) {
      set(i, numColumns - 1, col.get(i))
      i += 1
    }
  }

  def addRow(row: DenseVector) = {
    ??? //isso deve estar errado...
    resize(numRows + 1, numColumns)
    var j = 0
    while (j < numColumns) {
      set(numRows - 1, j, row.get(j))
      j += 1
    }
  }

  def resize(rows: Int, columns: Int) {
    numRows = rows
    numColumns = columns
  }

//  override def forEach(action: Consumer[_ >: MatrixEntry]): Unit = ???
//
//  override def spliterator(): Spliterator[MatrixEntry] = ???
}
