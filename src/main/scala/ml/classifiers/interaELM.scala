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
import ml.models.{ELMGenericModel, Model}
import ml.neural.elm.ELMUtils
import no.uib.cipr.matrix.DenseMatrix

/**
 * Grows network from 1 to Lmax according to arriving instances.
 * @param Lmax
 * @param seed
 */
case class interaELM(Lmax: Int, override val seed: Int = 1) extends ConvergentIncremental with ConvergentGrowing {
  override val toString = "interaELM"
  val Lbuild = 1

  override def build(trSet: Seq[Pattern]) = {
    val model = cast(super.build(trSet))
    modelSelection(model)
  }

  override def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val m = cast(model)
    if (math.sqrt(m.H.numRows() + 1).toInt > math.sqrt(m.H.numRows()).toInt) modelSelection(m) else m
  }

  protected def modelSelection(model: ELMGenericModel) = {
    var m = model
    val (_, best) = 1 to Lmax map { L =>
      if (L > 1) m = growByOne(m)
      val H = m.H
      val Beta = m.Beta
      val Y = m.Y
      val HHinv = m.HHinv
      val E = errorMatrix(H, Beta, Y)
      val press = PRESS(E)(HHinv)
      //      println("PRESS: " + press + " L: " + L)
      (press, m)
    } minBy (_._1)

    //    l foreach (x => println(x._2.rnd.getSeed))
    //    val (_, best) = l minBy (_._1)
    //     println(" L: " + best.H.numColumns())
    best
  }

  protected def errorMatrix(H: DenseMatrix, Beta: DenseMatrix, Y: DenseMatrix) = {
    val Prediction = new DenseMatrix(H.numRows(), Beta.numColumns())
    val E = Y.copy()
    ELMUtils.feedOutput(H, Beta, Prediction)
    E.add(-1, Prediction)
    E
  }

}


/*
========================================================

package ml.neural.old

import ml.Pattern
import no.uib.cipr.matrix.Matrix.Norm
import no.uib.cipr.matrix._

/**
 * Essas demoram mais pra dar matriz singular (ao incrementar L):
 * /usr/lib/lapack/liblapack.so.3
 * /usr/lib/libblas/libblas.so.3
 * @param hiddenLayerSize
 * @param patterns
 * @param seed
 */
case class OSELMold(hiddenLayerSize: Int, patterns: Seq[Pattern], seed: Int, testRank: Boolean = false, QR: Boolean = false)
  extends ELMtrait {
  var denPRESS = 0d
  patterns2matrices(patterns)
  initializeWeights()
  if (QR) buildQR() else build()

  /**
   * Moore-Penrose generalized inverse matrix.
   * Evita ill-conditioned matrices, i.e. singular ones (det = 0).
   * Based on the Java code, which was based on Huang Matlab code.
   * Theory:Ridge regression
   * MP(A) = inv((H'*H+lumda*I))*H'
   * @return (M0, pseudo-inverse) M0 is used for incremental learning (OS-ELM)
   */
  val lumda = 0.0

  /**
   * nothing in this class is thread safe
   * not even the resulting array
   * use .toList to avoid problems
   * @param pattern
   * @return
   */
  def distribution(pattern: Pattern) = {
    val data = squashedOutput(pattern)
    val Tinc = new DenseVector(data, false)
    val divBySum = 1 / data.sum
    Tinc.scale(divBySum)
    //    println(data.toList)
    data
  }

  /**
   * Good to put outputs inside [0,1] interval for probability estimation.
   * @param pattern
   * @return
   */
  def squashedOutput(pattern: Pattern) = {
    calculateHinc(pattern.arraymtj, testHincm, testHinc)
    outputLayert.mult(testHinc, Tinc)
    val Ox1 = new DenseVector(nclasses)
    outputLayert.mult(testHinc, Ox1)

    //Aplica sigmoide na camada de saida.
    i = 0
    var v = 0d
    while (i < Ox1.size) {
      v = sigm(Ox1.get(i) * 2 - 1)
      Ox1.set(i, v)
      i += 1
    }
    Ox1.getData
  }

  def checkAvgStability(p: Pattern) = {
    val a = predict(p)
    save()
    increment(p)
    decrement(p)
    val b = predict(p)
    restore()
    a == b
  }

  def checkStability(p: Pattern) = {
    val a = p.label_array.zip(output(p)).map {
      case (a, b) => a - b
    }.toList
    save()
    increment(p)
    val b = PRESSarray(p).toList
    restore()
    Datasets.dist(a, b)
  }

  def denPRESSaccuracy(testSet: Seq[Pattern]) = {
    denPRESS = 0
    val n = testSet.length.toDouble
    (testSet count PRESShit) / n
    denPRESS / n
  }

  def PRESSsqrt(p: Pattern) = {
    calculateHinc(p.arraymtj, Hincm, Hinc)
    outputLayert.mult(Hinc, Tinc)
    val output = Tinc.getData

    val h = Hinc
    val Pkt = Pk.transpose()
    val hP = new DenseVector(hiddenLayerSize)
    Pkt.mult(h, hP)
    val hPh = hP.dot(h) //.round
    val den = 1 - hPh
    var totalpress = 0d
    var o = 0
    //    var idx = -1
    while (o < p.nclasses) {
      //      val tmp = (output(o)) / den
      val tmp = (p.label_array(o) - output(o)) / den
      val press = tmp.abs //* tmp
      //      if (press > maxpress) {
      totalpress += press
      //        idx = o
      //      }
      o += 1
    }
    //    if (maxpress < 0.5) 1d else 0d
    totalpress / p.nclasses
  }

  def denPRESS0(p: Pattern) = {
    calculateHinc(p.arraymtj, Hincm, Hinc)
    outputLayert.mult(Hinc, Tinc)
    val h = Hinc
    val Pkt = Pk.transpose()
    val hP = new DenseVector(hiddenLayerSize)
    Pkt.mult(h, hP)
    val hPh = hP.dot(h) //.round
    val den = 1 - hPh
    den.abs
  }

  def PRESSaccuracy(testSet: Seq[Pattern]) = {
    denPRESS = 0
    (testSet count PRESShit) / testSet.length.toDouble
  }

  def PRESScru(testSet: Seq[Pattern]) = {
    (testSet map PRESShitcru).toList
  }


  def PRESShit(p: Pattern) = {
    calculateHinc(p.arraymtj, Hincm, Hinc)
    outputLayert.mult(Hinc, Tinc)
    val output = Tinc.getData
    val h = Hinc
    val Pkt = Pk.copy()
    Pk.transpose(Pkt)
    val hP = new DenseVector(hiddenLayerSize)
    Pkt.mult(h, hP)
    val hPh = hP.dot(h)
    val den = 1 - hPh
    denPRESS += den.abs
    var o = 0
    val a = new Array[Double](nclasses)
    while (o < p.nclasses) {
      a(o) = (p.label_array(o) - output(o)) / den
      o += 1
    }
    p.label_array.zip(a).map {
      case (u, v) => u - v
    }.toList.zipWithIndex.max._2 == p.label
  }

  def PRESSarray(p: Pattern) = {
    calculateHinc(p.arraymtj, Hincm, Hinc)
    outputLayert.mult(Hinc, Tinc)
    val output = Tinc.getData
    val h = Hinc
    val Pkt = Pk.transpose()
    val hP = new DenseVector(hiddenLayerSize)
    Pkt.mult(h, hP)
    val hPh = hP.dot(h)
    val den = 1 - hPh
    var o = 0
    val a = new Array[Double](nclasses)
    while (o < p.nclasses) {
      a(o) = (p.label_array(o) - output(o)) / den
      o += 1
    }
    a
  }

  def PRESShitcru(p: Pattern) = {
    calculateHinc(p.arraymtj, Hincm, Hinc)
    outputLayert.mult(Hinc, Tinc)
    val output = Tinc.getData
    val h = Hinc
    val Pkt = Pk.copy()
    Pk.transpose(Pkt)
    val hP = new DenseVector(hiddenLayerSize)
    Pkt.mult(h, hP)
    val hPh = hP.dot(h)
    val den = 1 - hPh
    denPRESS += den.abs
    var o = 0
    val a = new Array[Double](nclasses)
    while (o < p.nclasses) {
      a(o) = (p.label_array(o) - output(o)) / den
      o += 1
    }
    a.toList.zip(p.label_array).map{case(a,b)=>b-a}
    //    (p.label, a.toList.map(_ / a.sum))
  }

  def rank = {
    ??? //how to define zero?
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

  private def pinvpre(H0: DenseMatrix, H0T: DenseMatrix, H0TH0_1: DenseMatrix) = {
    //H0TH0.add(lumda, I) //this modifies H0TH0!
    val tmp_LxL = new DenseMatrix(hiddenLayerSize, hiddenLayerSize)
    try {
      //      H0TH0.solve(ILxL, tmp_LxL)
      val pseudo_inverse = new DenseMatrix(hiddenLayerSize, H0.numRows)
      //      tmp_LxL.mult(H0T, pseudo_inverse)
      H0TH0_1.mult(H0T, pseudo_inverse)
      pseudo_inverse
    } catch {
      case e: MatrixSingularException => println("L=" + hiddenLayerSize + "N=" + ninsts + ". Singular matrix:\n" + H0TH0_1)
        sys.exit(0)
    }
  }

  private def build() {
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
    Ht.mult(H, H0tH0_LxL)
    val P0 = invLxL(H0tH0_LxL)
    pinvH = if (testRank) pinvSVD(H.copy()) else pinvpre(H, Ht, P0.copy()) //precisa ser pinv pois a matriz raramente vai ser quadrada
    pinvH.mult(T, outputLayer)
    outputLayer.transpose(outputLayert)
    Pk = P0

    //create BiasMatrixinc for tests and for increments
    i = 0
    while (i < hiddenLayerSize) {
      BiasMatrixinc.set(i, BiasMatrix.get(i, 0))
      i += 1
    }
  }

  def invLxL(A: Matrix) = {
    val tmp_LxL = new DenseMatrix(hiddenLayerSize, hiddenLayerSize)
    try {
      A.solve(ILxL, tmp_LxL)
      tmp_LxL
    } catch {
      case e: MatrixSingularException => println("L=" + hiddenLayerSize + "N=" + ninsts + ". Singular matrix:\n" + A)
        sys.exit(0)
    }
  }

  def invNxN(A: Matrix) = {
    A.solve(INxN, tmp_NxN)
    tmp_NxN
  }

  def inv(A: Matrix) = {
    val I = Matrices.identity(A.numRows())
    val tmp = A.copy()
    A.solve(I, tmp)
    tmp
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

  protected def calculateHinc(p: DenseVector, Hmat: DenseMatrix, Hvec: DenseVector) {
    hiddenLayer.mult(p, tempHinc) //LxE Ex1 = Lx1
    tempHinc.add(BiasMatrixinc) //Lx1
    //Aplica sigmoide na camada oculta.
    i = 0
    var v = 0d
    while (i < hiddenLayerSize) {
      v = sigm2(tempHinc.get(i))
      Hmat.set(0, i, v)
      Hvec.set(i, v)
      i += 1
    }
  }

  /**
   * OS-ELM one-by-one incremental steps.
   * inputs: hiddenLayer BiasMatrixinc Pk outputLayert outputLayer
   * changes: Hincm Hinc tempHinc Ainc tmp_LxLinc newPinc Pk tmp_1xOinc tmp_1xOincm newOutputLayerinc outputLayer outputLayert
   */
  def inc(p: DenseVector, tt: DenseVector) {
    calculateHinc(p, Hincm, Hinc) //cpu 7%
    System.arraycopy(tt.getData, 0, tincm.getData, 0, nclasses)

    //P update
    Pk.mult(Hinc, Ainc) //cpu 39%  LxL Lx1 = Lx1

    val tmp = Hinc.dot(Ainc)
    val factor = -1 / (1 + tmp)

    Aincm.mult(Hincm, tmp_LxL2)
    tmp_LxL2.mult(Pk, newPinc)
    newPinc.scale(factor)
    newPinc.add(Pk) //cpu 11%
    Pk.set(newPinc)

    //cpu 40%
    Hincm.mult(outputLayer, tmp_1xOincm)
    tmp_1xOincm.scale(-1)
    tmp_1xOincm.add(tincm)
    newPinc.mult(Hinc, Ainc) //reuses A's memory space
    Aincm.mult(tmp_1xOincm, newOutputLayerinc) //Am is updated when A changes
    newOutputLayerinc.add(outputLayer)
    outputLayer.set(newOutputLayerinc)
    outputLayer.transpose(outputLayert)
  }

  def decrement(pattern: Pattern) {
    val transP = pattern.arraymtj
    val t = pattern.reversed_weighted_label_array_mtj
    inc(transP, t)
  }

  def increment(pattern: Pattern) {
    //cpu 97%
    val p = pattern.arraymtj
    val tt = pattern.weighted_label_array_mtj
    inc(p, tt)
  }

  //    //Converts OutputWeight to Layer format.
  //    for ((neuron, ne) <- layers(1).neurons.zipWithIndex) {
  //      val tmp = neuron.weights.length - 1
  //      0 until tmp foreach {
  //        we => neuron.weights(we) = Beta0.get(we, ne)
  //      }
  //    }

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
