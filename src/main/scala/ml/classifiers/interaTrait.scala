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
import ml.models.{ELMIncModel, ELMModel, Model}

/**
 * Growing + OS-ELM.
 */
trait interaTrait extends ConvergentIncremental with ConvergentGrowing {
  val Lbuild = 1

  override def build(trSet: Seq[Pattern]) = {
    val model = cast(super.build(trSet))
    modelSelection(model)
  }

  override def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val m = super.update(model)(pattern)
    if (math.sqrt(m.N + 1).toInt > math.sqrt(m.N).toInt) {
      val gm = modelSelection(m)
      ELMIncModel(gm.rnd, gm.Alfat, gm.biases, gm.Beta, gm.P, gm.N, gm.Xt, gm.Y)
    } else m
  }

  protected def modelSelection(model: ELMModel): ELMModel
}


/*
====
  def rank = {
    ??? //how to define which value is "zero enough"?
    val svd = new SVD(H.numRows(), H.numColumns())
    val s = svd.factor(H)
    //    val U = s.getU()
    val S = s.getS()
    //    val Vt = s.getVt()
    var i = 0
    var r = 0
    while (i < H.numColumns) {
      val v = S(i).abs
      if (v >= 0.000000001) {
        r += 1
      }
      i += 1
    }
    r
  }

  def incChunk(patternschunk: Seq[Pattern]) {
    val newninsts = patternschunk.length

    //resize matrices according to new N
    P = new DenseMatrix(nattributes, newninsts)
    Tt = new DenseMatrix(nclasses, newninsts)
    tempH = new DenseMatrix(hiddenLayerSize, newninsts)
    Ht = new DenseMatrix(hiddenLayerSize, newninsts)
    H = new DenseMatrix(newninsts, hiddenLayerSize)

    //resize BiasMatrix
    BiasMatrix = new DenseMatrix(hiddenLayerSize, newninsts)
    i = 0
    while (i < hiddenLayerSize) {
      val v = BiasMatrixinc.get(i)
      j = 0
      while (j < newninsts) {
        BiasMatrix.set(i, j, v)
        j += 1
      }
      i += 1
    }

    transP.transpose(P)
    T.transpose(Tt)
    hiddenLayer.mult(P, tempH)
    tempH.add(BiasMatrix)

    //recalculate H and Ht
    i = 0
    while (i < hiddenLayerSize) {
      j = 0
      while (j < newninsts) {
        Ht.set(i, j, sigm2(tempH.get(i, j)))
        j += 1
      }
      i += 1
    }
    Ht.transpose(H)

    //OS-ELM part
    //P update
    val tmpLxN = new DenseMatrix(hiddenLayerSize, newninsts)
    Pk.mult(Ht, tmpLxN)
    val tmp_NxN = new DenseMatrix(newninsts, newninsts)
    H.mult(tmpLxN, tmp_NxN)
    val I = Matrices.identity(newninsts)
    tmp_NxN.add(I)
    val Binv = inv(tmp_NxN) //matlab implementation use inv() here
    val tmp2LxN = new DenseMatrix(hiddenLayerSize, newninsts)
    tmpLxN.mult(Binv, tmp2LxN)
    val D = new DenseMatrix(newninsts, hiddenLayerSize)
    H.mult(Pk, D)
    tmp2LxN.mult(D, H0tH0_LxL)
    H0tH0_LxL.scale(-1)
    H0tH0_LxL.add(Pk)
    Pk.set(H0tH0_LxL)

    //Beta update
    val tmp_NxO = new DenseMatrix(newninsts, nclasses)
    H.mult(outputLayer, tmp_NxO)
    tmp_NxO.scale(-1)
    tmp_NxO.add(T)
    Pk.mult(Ht, tmp2LxN) //reuses tmp2LxN
    tmp2LxN.mult(tmp_NxO, tmpLxO)
    tmpLxO.add(outputLayer)
    outputLayer.set(tmpLxO)
    outputLayer.transpose(outputLayert)
  }

  def incrementChunk(patternschunk: Seq[Pattern]) {
    //patterns to matrices
    val newninsts = patternschunk.length
    T = new DenseMatrix(newninsts, nclasses)
    transP = new DenseMatrix(newninsts, nattributes)
    i = 0
    patternschunk foreach {
      inst =>
        val arr = inst.array
        val larr = inst.weighted_label_array
        j = 0
        while (j < nattributes) {
          transP.set(i, j, arr(j))
          j += 1
        }
        j = 0
        while (j < nclasses) {
          T.set(i, j, larr(j))
          j += 1
        }
        i += 1
    }
    incChunk(patternschunk)
  }

  def decrementChunk(patternschunk: Seq[Pattern]) {
    //patterns to matrices
    val newninsts = patternschunk.length
    T = new DenseMatrix(newninsts, nclasses)
    transP = new DenseMatrix(newninsts, nattributes)
    i = 0
    patternschunk foreach {
      inst =>
        val arr = inst.array
        val larr = inst.reversed_weighted_label_array
        j = 0
        while (j < nattributes) {
          transP.set(i, j, arr(j))
          j += 1
        }
        j = 0
        while (j < nclasses) {
          T.set(i, j, larr(j))
          j += 1
        }
        i += 1
    }
    incChunk(patternschunk)
  }

  private def pinvSVD(H0: DenseMatrix) = {
    val rows = H0.numRows()
    val svd = new SVD(rows, hiddenLayerSize, true)
    val s = svd.factor(H0) //changes H! (H: NxL)
    val U = s.getU //LxL
    val S = s.getS //min(N,L)
    val Vt = s.getVt //LxL
    var i = 0
    val max = S.head
    smallestSV = S.last
    maxMinRatioSV = max / smallestSV
    val Sinv = new DenseMatrix(hiddenLayerSize, rows) //LxN
    while (i < hiddenLayerSize) {
      Sinv.set(i, i, 1d / S(i))
      i += 1
    }

    //calculate pinv via SVD
    val tmp = new DenseMatrix(hiddenLayerSize, rows) //LxN
    Vt.transpose()
    U.transpose()
    Vt.mult(Sinv, tmp) //Vt is V now
    tmp.mult(U, Sinv) //LxN reuses Sinv; Ut is U now
    Sinv
  }

  def buildQR() {
    transP.transpose(P)
    T.transpose(Tt)
    hiddenLayer.mult(P, tempH)
    tempH.add(BiasMatrix)

    //Calculate H and Ht.
    i = 0
    while (i < hiddenLayerSize) {
      j = 0
      while (j < ninsts) {
        Ht.set(i, j, sigm2(tempH.get(i, j)))
        j += 1
      }
      i += 1
    }
    Ht.transpose(H)
    //    Ht.mult(H, H0tH0_LxL)
    //    System.arraycopy(tmp2_LxL.getData, 0, tmp_LxL.getData, 0, PkDataLength)
    pinvH = pinvQR(H.copy())
    pinvH.mult(T, outputLayer)
    outputLayer.transpose(outputLayert)

    //    val P0 = inv(H0tH0_LxL)
    //    Pk = P0

    //create BiasMatrixinc for tests and for increments
    i = 0
    while (i < hiddenLayerSize) {
      BiasMatrixinc.set(i, BiasMatrix.get(i, 0))
      i += 1
    }
  }

  def buildLU() {
    transP.transpose(P)
    T.transpose(Tt)
    hiddenLayer.mult(P, tempH)
    tempH.add(BiasMatrix)

    //Calculate H and Ht.
    i = 0
    while (i < hiddenLayerSize) {
      j = 0
      while (j < ninsts) {
        Ht.set(i, j, sigm2(tempH.get(i, j)))
        j += 1
      }
      i += 1
    }
    Ht.transpose(H)
    //    Ht.mult(H, H0tH0_LxL)
    //    System.arraycopy(tmp2_LxL.getData, 0, tmp_LxL.getData, 0, PkDataLength)
    pinvH = pinvLU(H.copy())
    pinvH.mult(T, outputLayer)
    outputLayer.transpose(outputLayert)

    //    val P0 = inv(H0tH0_LxL)
    //    Pk = P0

    //create BiasMatrixinc for tests and for increments
    i = 0
    while (i < hiddenLayerSize) {
      BiasMatrixinc.set(i, BiasMatrix.get(i, 0))
      i += 1
    }
  }

  private def pinvQR(H0: DenseMatrix) = {
    val rows = H0.numRows()
    val qr = new QR(rows, hiddenLayerSize)
    qr.factor(H0) //changes H! (H: NxL)
    val Q = qr.getQ
    val R = qr.getR //LxL
    val Qt = new DenseMatrix(hiddenLayerSize, rows) //LxN
    val pinv = new DenseMatrix(hiddenLayerSize, rows) //LxN
    Q.transpose(Qt)
    val Rinv = invLxL(R)
    Rinv.mult(Qt, pinv)
    pinv
  }

  def rcondOne = {
    val rows = H.numRows()
    val lu = new DenseLU(rows, hiddenLayerSize)
    lu.factor(H.copy)
    lu.rcond(H, Norm.One)
  }

  def rcondInf = {
    val rows = H.numRows()
    val lu = new DenseLU(rows, hiddenLayerSize)
    lu.factor(H.copy)
    lu.rcond(H, Norm.Infinity)
  }

  private def pinvLU(H0: DenseMatrix) = {
    val rows = H0.numRows()
    val lu = new DenseLU(rows, hiddenLayerSize)
    lu.factor(H0) //changes H! (H: NxL)
    ???
    //    val Q = qr.getQ
    //    val R = qr.getR //LxL
    //    val Qt = new DenseMatrix(hiddenLayerSize, rows) //LxN
    //    val pinv = new DenseMatrix(hiddenLayerSize, rows) //LxN
    //    Q.transpose(Qt)
    //    val Rinv = inv(R)
    //    Rinv.mult(Qt, pinv)
    //    pinv
  }
}

 */
