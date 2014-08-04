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
import no.uib.cipr.matrix._
import util.{Datasets, Tempo, XSRandom}

/**
 * Stateful EMELM
 * @param patterns
 * @param seed
 * @param SVD
 */
case class EMELM(patterns: Seq[Pattern], seed: Int, SVD: Boolean = false) {
  val nattributes = patterns.head.nattributes
  val nclasses = patterns.head.nclasses
  val ninsts = patterns.length
  val rnd = new XSRandom(seed)
  /**
   * Depends on: mH, mHinv, hiddenLayer, biases
   * Changes: hiddenLayer, biases, mH, mHinv, outputLayert
   */
  val HHinv = new DenseMatrix(ninsts, ninsts)
  val I = Matrices.identity(ninsts)
  val lastAddedHColumnt = new DenseMatrix(1, ninsts)
  val tmp2 = new DenseVector(1)
  patterns2matrices(patterns)
  protected val T = new DenseMatrix(ninsts, nclasses)
  protected val Tt = new DenseMatrix(nclasses, ninsts)
  protected val inputLayer = new DenseMatrix(ninsts, nattributes)
  protected val inputLayert = new DenseMatrix(nattributes, ninsts)
  var (hiddenLayer, biases, mH, mHinv, outputLayert, p) = {
    val (hiddenLayer, neuron, biases) = addNeuron(new DenseMatrix(0, nattributes), new DenseVector(0))
    val (mH, h) = resizeH(new DenseMatrix(ninsts, 0), neuron, biases)

    val (hinv, p) = if (SVD) {
      val hinv = pinvSVD(mH)

      //Calculate P for PRESS and also OS-ELM.
      val mHinvt = new DenseMatrix(mH.numColumns(), hinv.numRows())
      mH.transpose(mHinvt)
      val P = new DenseMatrix(hinv.numRows(), mHinvt.numColumns())
      hinv.mult(mHinvt, P)

      (hinv, P)
    } else {
      val HtH_LxL = new DenseMatrix(hiddenLayer.numRows(), hiddenLayer.numRows())
      val Ht = new DenseMatrix(mH.numColumns, mH.numRows)
      mH.transpose(Ht)
      Ht.mult(mH, HtH_LxL)
      val hinv = pinvpre(mH, Ht, HtH_LxL)

      //Calculate P for PRESS and also OS-ELM.
      //      val P = inv(HtH_LxL)

      (hinv, mH) //TODO: trocar mH por P
    }

    val outputLayert = updateOutputLayer(hiddenLayer, hinv)

    (hiddenLayer, biases, mH, hinv, outputLayert, p)
  }

  def inv(A: Matrix) = {
    val tmp = new DenseMatrix(A.numRows(), A.numColumns())
    try {
      A.solve(Matrices.identity(A.numRows()), tmp)
      tmp
    } catch {
      case e: MatrixSingularException => println("L=" + A.numRows() + "N=" + ninsts + ". Singular matrix:\n" + A)
        sys.exit(1)
    }
  }

  def grow() {
    mH.mult(mHinv, HHinv) //64
    val (newHiddenLayer, newNeuron, newBiases) = addNeuron(hiddenLayer, biases) //66
    val (newH, newh) = resizeH(mH, newNeuron, newBiases) //h_j in the paper is the newh 70

    val I_HHinv = HHinv.add(-1, I) //285
    val tmp = new DenseMatrix(newh, false) //285

    tmp.transpose(lastAddedHColumnt) //286
    val num = new DenseMatrix(1, ninsts) //286
    lastAddedHColumnt.mult(I_HHinv, num) //419
    num.mult(newh, tmp2) //420
    val factor = 1 / tmp2.get(0)
    num.scale(factor)
    val D = num //1xN 420

    val Hinvh = new DenseVector(hiddenLayer.numRows()) //Lx1
    mHinv.mult(newh, Hinvh)
    val Hinvhm = new DenseMatrix(Hinvh, false)
    val HinvhD = new DenseMatrix(hiddenLayer.numRows(), ninsts) //LxN
    Hinvhm.mult(D, HinvhD)
    val tmp3 = mHinv.copy()
    tmp3.add(-1, HinvhD)
    val U = tmp3 //LxN

    val newHinv = stackUD(U, D)
    val newOutputLayert = updateOutputLayer(newHiddenLayer, newHinv)

    //Update state of the ELM.
    hiddenLayer = newHiddenLayer
    biases = newBiases
    mH = newH

    mHinv = newHinv
    outputLayert = newOutputLayert
    //    Tempo.print_stop

    //Calculate P for PRESS and also OS-ELM.
    //    Tempo.start
    //    val mHinvt = new DenseMatrix(mHinv.numColumns(), mHinv.numRows())
    //    mHinv.transpose(mHinvt)
    //    p = new DenseMatrix(mHinv.numRows(), mHinvt.numColumns())
    //    mHinv.mult(mHinvt, p)
    //    Tempo.print_stop
  }

  protected def updateOutputLayer(hiddenLayer: DenseMatrix, Hinv: DenseMatrix) = {
    val outputLayer = new DenseMatrix(hiddenLayer.numRows(), nclasses) //LxO
    val outputLayert = new DenseMatrix(nclasses, hiddenLayer.numRows()) //OxL
    Hinv.mult(T, outputLayer)
    outputLayer.transpose(outputLayert)
    outputLayert
  }

  private def stackUD(U: DenseMatrix, D: DenseMatrix) = {
    val UD = new DenseMatrix(U.numRows + 1, ninsts)
    var i = 0
    var j = 0
    while (i < U.numRows) {
      j = 0
      while (j < ninsts) {
        UD.set(i, j, U.get(i, j))
        j += 1
      }
      i += 1
    }
    j = 0
    while (j < ninsts) {
      UD.set(i, j, D.get(0, j))
      j += 1
    }
    UD
  }

  protected def resizeH(H: DenseMatrix, lastNeuron: DenseVector, biases: DenseVector) = {
    val newH = new DenseMatrix(ninsts, H.numColumns + 1)
    val h = new DenseVector(ninsts)
    var i = 0
    var j = 0
    while (i < ninsts) {
      j = 0
      while (j < H.numColumns()) {
        val v = H.get(i, j)
        newH.set(i, j, v)
        j += 1
      }
      i += 1
    }
    inputLayer.mult(lastNeuron, h)
    i = 0
    while (i < ninsts) {
      val v = sigm2(h.get(i) + biases.get(j))
      newH.set(i, j, v)
      h.set(i, v)
      i += 1
    }
    (newH, h)
  }

  protected def sigm2(x: Double) = (1.0 / (1 + math.exp(-x)) - 0.5) * 2d

  protected def addNeuron(hiddenLayer: DenseMatrix, biases: DenseVector) = {
    val newHiddenLayer = new DenseMatrix(hiddenLayer.numRows + 1, nattributes)
    val newBiases = new DenseVector(biases.size + 1)
    val newNeuron = new DenseVector(nattributes)
    var i = 0
    var j = 0
    while (i < hiddenLayer.numRows) {
      j = 0
      while (j < nattributes) {
        newHiddenLayer.set(i, j, hiddenLayer.get(i, j))
        j += 1
      }
      newBiases.set(i, biases.get(i))
      i += 1
    }
    j = 0
    while (j < nattributes) {
      val v = rnd.nextDouble() * 2 - 1
      newHiddenLayer.set(i, j, v)
      newNeuron.set(j, v)
      j += 1
    }
    newBiases.set(i, rnd.nextDouble() * 2 - 1)
    (newHiddenLayer, newNeuron, newBiases)
  }

  def accuracy(testSet: Seq[Pattern]) = hits(testSet) / testSet.length.toDouble

  def hits(testSet: Seq[Pattern]) = testSet count hit

  def hit(pattern: Pattern) = predict(pattern) == pattern.label

  def predict(pattern: Pattern) = {
    val data = output(pattern)
    var max = Double.MinValue
    var pred = 0
    var j = 0
    while (j < nclasses) {
      val v = data(j)
      if (v > max) {
        pred = j
        max = v
      }
      j += 1
    }
    pred
  }

  def distribution(pattern: Pattern) = {
    val data = squashedOutput(pattern)
    val Tinc = new DenseVector(data, false)
    val sum = data.sum
    Tinc.scale(1 / sum)
    data.toList
  }

  def squashedOutput(pattern: Pattern) = {
    val Ox1 = output(pattern)

    //Aplica sigmoide na camada de saida.
    var i = 0
    var v = 0d
    while (i < Ox1.size) {
      v = sigm(Ox1(i) * 2 - 1)
      Ox1(i) = v
      i += 1
    }
    Ox1
  }

  protected def sigm(x: Double) = 1.0 / (1 + math.exp(-x))

  def output(pattern: Pattern) = {
    val Lx1 = new DenseVector(mH.numColumns())
    hiddenLayer.mult(pattern.arraymtj, Lx1) //LxE Ex1 = Lx1
    Lx1.add(biases) //Lx1

    //Aplica sigmoide na camada oculta.
    var i = 0
    var v = 0d
    while (i < mH.numColumns()) {
      v = sigm2(Lx1.get(i))
      Lx1.set(i, v)
      i += 1
    }
    val Ox1 = new DenseVector(nclasses)
    outputLayert.mult(Lx1, Ox1)
    Ox1.getData
  }

  protected def patterns2matrices(insts: Seq[Pattern]) {
    var i = 0
    var j = 0
    insts foreach {
      inst =>
        val arr = inst.array
        val larr = inst.weighted_label_array
        j = 0
        while (j < nattributes) {
          inputLayer.set(i, j, arr(j))
          j += 1
        }
        j = 0
        while (j < nclasses) {
          T.set(i, j, larr(j))
          j += 1
        }
        i += 1
    }
    inputLayer.transpose(inputLayert)
    T.transpose(Tt)
  }

  private def pinvpre(H0: DenseMatrix, H0T: DenseMatrix, H0TH0: DenseMatrix) = {
    val tmp_LxL = new DenseMatrix(H0.numColumns, H0.numColumns)
    val I = Matrices.identity(H0.numColumns)
    H0TH0.solve(I, tmp_LxL)
    val pseudo_inverse = new DenseMatrix(H0.numColumns, H0.numRows)
    tmp_LxL.mult(H0T, pseudo_inverse)
    pseudo_inverse
  }

  private def pinvSVD(H0: DenseMatrix) = {
    val rows = H0.numRows()
    val svd = new SVD(rows, H0.numColumns, true)
    val s = svd.factor(H0) //changes H! (H: NxL)
    val U = s.getU //LxL
    val S = s.getS //min(N,L)
    val Vt = s.getVt //LxL
    var i = 0
    val max = S.head
    //    smallestSV = S.last
    //    maxMinRatioSV = max / smallestSV
    val Sinv = new DenseMatrix(H0.numColumns, rows) //LxN
    while (i < H0.numColumns) {
      Sinv.set(i, i, 1d / S(i))
      i += 1
    }

    //calculate pinv via SVD
    val tmp = new DenseMatrix(H0.numColumns, rows) //LxN
    Vt.transpose()
    U.transpose()
    Vt.mult(Sinv, tmp) //Vt is V now
    tmp.mult(U, Sinv) //LxN reuses Sinv; Ut is U now
    Sinv
  }
}

object EMTest extends App {
  val dataset = "iris.arff"
  val data = Datasets.arff(bina = true)(dataset) match {
    case Right(x) => x.take(10)
    case Left(str) => println("Could not load " + dataset + " dataset from the program path: " + str); sys.exit(1)
  }
  1 to 10 foreach { _ =>
    val e1 = ml.classifiers.EMELM(999, 1)
    var m = e1.build(data)
    m = e1.growByOne(m, true)
    m = e1.growByOne(m, true)
    m = e1.growByOne(m, true)
    //    println(m.accuracy(data))
  }

  val n = 5000 / data.length

  Tempo.start
  val e1 = ml.classifiers.EMELM(999, 1)
  var m = e1.build(data)
  1 to n foreach { _ =>
    val e1 = ml.classifiers.EMELM(999, 1)
    m = e1.build(data)
    1 to 100 foreach { _ =>
      m = e1.growByOne(m)
    }
  }
  println(m.accuracy(data))
  Tempo.print_stop
  println("")

  Tempo.start
  var e0 = EMELM(data, 1)
  1 to n foreach { _ =>
    e0 = EMELM(data, 1)
    1 to 100 foreach { _ =>
      e0.grow
    }
  }
  Tempo.print_stop
  println(e0.accuracy(data))

  println("")
}