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
import ml.models.{ELMSimpleModel, Model}
import ml.mtj.ResizableDenseMatrix
import ml.neural.elm.ConvexIELMTrait
import ml.neural.elm.Data._
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.{Tempo, XSRandom}

case class ECIELM(Lbuild: Int, seed: Int = 42, callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ()) extends ConvexIELMTrait {
  override val toString = "ECIELM"
  val CANDIDATES = 10

  override def build(trSet: Seq[Pattern]) = {
    Tempo.start
    val rnd = new XSRandom(seed)
    val ninsts = checkEmptyness(trSet)
    val nclasses = trSet.head.nclasses
    val natts = trSet.head.nattributes
    val X = patterns2matrix(trSet, ninsts)
    val biases = Array.fill(Lbuild)(0d)
    val Alfat = new ResizableDenseMatrix(Lbuild, natts)
    val Beta = new ResizableDenseMatrix(Lbuild, nclasses)
    val H = new ResizableDenseMatrix(ninsts, Lbuild)
    H.resizeCols(0)
    val (t, e) = patterns2te(trSet, ninsts)
    var l = 0
    val tmp = new DenseVector(H.numRows())
    val tmp2 = new DenseVector(H.numRows())
    while (l < Lbuild) {
      Alfat.resizeRows(l + 1) //needed to call f()
      Beta.resizeRows(l + 1)
      val (weights, bias, h, beta) = createNodeForConvexUpdateAmongCandidates(rnd, X, t, e, tmp, tmp2)
      H.addCol(h)
      biases(l) = bias
      updateNetwork(l, weights, beta, Beta, Alfat)
      //      println((1 + l) + ": res. error = " + error + " delta error: " + (deltaerror).formatted("%3.4f"))
      //            println(PRESSaccuracy(Yt, Beta, H, Ht, P, labels))
      l += 1
      val te = Tempo.stop
      Tempo.start
      f(ELMSimpleModel(rnd.clone(), Alfat, biases, Beta, ninsts), te)
    }
    //    Alfat.resizeRows(l)
    //    Beta.resizeRows(l)
    //    ELMModel(Alfat, biases.take(Beta.numRows()), new DenseMatrix(1, 1), Beta)
    val model = ELMSimpleModel(rnd, Alfat, biases, Beta, ninsts)
    model
  }

  /**
   * Mutate e and rnd
   * @return
   */
  def createNodeForConvexUpdateAmongCandidates(rnd: XSRandom, X: DenseMatrix, t: Array[DenseVector], e: Array[DenseVector], tmp: DenseVector, tmp2: DenseVector) = {
    val nclasses = t.size
    val natts = X.numColumns()
    val ninsts = X.numRows()
    val candidates = 0 until CANDIDATES map { idx =>
      val newe = Array.fill(nclasses)(new DenseVector(ninsts))
      val (weights, bias, newRnd) = newNode(natts, rnd)
      rnd.setSeed(newRnd.getSeed)
      val alfa = new DenseVector(weights, false)
      val beta = new Array[Double](nclasses)
      val h = feedHidden(X, alfa, bias)
      val hneg = h.copy()
      hneg.scale(-1)
      var o = 0
      while (o < nclasses) {
        val neweo = newe(o)
        neweo.set(e(o))

        //Calculate new weight.
        tmp.set(t(o))
        tmp.add(hneg)
        tmp2.set(tmp)
        tmp2.scale(-1)
        tmp2.add(neweo)
        val nume = neweo.dot(tmp2)
        val deno = tmp2.dot(tmp2)
        val b = nume / deno
        beta(o) = b

        //Recalculate residual error.
        neweo.scale(1 - b)
        tmp.scale(b)
        neweo.add(tmp)

        o += 1
      }
      val err = newe.map(x => math.sqrt(x.dot(x))).sum
      (err, weights, bias, h, beta, newe)
    }
    val (_, weights, bias, h, beta, newe) = candidates.minBy(_._1)
    //    println("-------xxxxx--------" + candidates.map(_._1))
    e.zip(newe).foreach { case (a, b) => a.set(b)}
    (weights, bias, h, beta)
  }
}

//rnd ok

