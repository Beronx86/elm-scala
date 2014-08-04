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
import ml.neural.elm.Data._
import ml.neural.elm.{ConvexIELMTrait, Data}
import no.uib.cipr.matrix.{ResizableDenseMatrix, DenseMatrix, DenseVector}
import util.{Datasets, Tempo, XSRandom}

import scala.util.Random

/**
 * CI-ELM
 * Created by davi on 19/05/14.
 */
case class CIELM(seed: Int = 42, notes: String = "", callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ()) extends ConvexIELMTrait {
  override val toString = "CIELM_" + notes
  val Lbuild = -1

  def build(trSet: Seq[Pattern]): Model = {
    val nclasses = trSet.head.nclasses
    if (trSet.size < nclasses) {
      println("At least |Y| instances required.")
      sys.exit(1)
    }
    val initialTrSet = trSet.take(nclasses)
    val natts = initialTrSet.head.nattributes
    val X = patterns2matrix(initialTrSet, nclasses)
    val (t, e) = patterns2te(initialTrSet, nclasses)
    val firstModel = bareBuild(nclasses, natts, nclasses, X, e, t)
    trSet.drop(nclasses).foldLeft(firstModel)((m, p) => cast(update(m, fast_mutable = true)(p)))
  }

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val m = cast(model)
    val newE = m.e.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
    val newT = m.t.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
    val newX = Data.appendRowToMatrix(m.X, pattern.array)

    val (weights, bias, newRnd) = newNode(m.Alfat.numColumns(), m.rnd)
    val newAlfat = Data.appendRowToMatrix(m.Alfat, weights)
    val newBiases = Data.appendToArray(m.biases, bias)
    val (h, beta) = addNodeForConvexUpdate(weights, bias, newX, newT, newE)
    val newBeta = Data.appendRowToMatrix(m.Beta, beta)
    ELMSimpleModel(newRnd, newAlfat, newBiases, newBeta, newX, newE, newT)
  }

  /**
   * Mutate e
   * @return
   */
  def addNodeForConvexUpdate(weights: Array[Double], bias: Double, X: DenseMatrix, t: Vector[DenseVector], e: Vector[DenseVector]) = {
    val nclasses = t.size
    //Generate node and calculate h.
    val alfa = new DenseVector(weights, false)
    val beta = new Array[Double](nclasses)
    val h = feedHidden(X, alfa, bias)
    val hneg = h.copy()
    hneg.scale(-1)
    var o = 0
    while (o < nclasses) {
      //Calculate new weight.
      val t_h = t(o).copy()
      t_h.add(hneg)
      val a = t_h.copy()
      a.scale(-1)
      a.add(e(o))
      val nume = e(o).dot(a)
      val deno = a.dot(a)
      val b = nume / deno
      beta(o) = b

      //Recalculate residual error.
      e(o).scale(1 - b)
      t_h.scale(b)
      e(o).add(t_h)

      o += 1
    }
    (h, beta)
  }

  def bareBuild(ninsts: Int, natts: Int, nclasses: Int, X: DenseMatrix, e: Vector[DenseVector], t: Vector[DenseVector]) = {
    val L = nclasses
    val rnd = new XSRandom(seed)
    val biases = Array.fill(L)(0d)
    val Alfat = new ResizableDenseMatrix(L, natts)
    val Beta = new ResizableDenseMatrix(L, nclasses)
    var l = 0
    if (callf) while (l < L) {
      Alfat.resizeRows(l + 1) //needed to call f()
      Beta.resizeRows(l + 1)
      val (weights, b, newRnd) = newNode(natts, rnd)
      rnd.setSeed(newRnd.getSeed)
      val (h, beta) = addNodeForConvexUpdate(weights, b, X, t, e)
      biases(l) = b
      updateNetwork(l, weights, beta, Beta, Alfat)
      l += 1
      val te = Tempo.stop
      Tempo.start
      f(ELMSimpleModel(newRnd, Alfat, biases, Beta, X, e, t), te)
    }
    else while (l < L) {
      val (weights, b, newRnd) = newNode(natts, rnd)
      rnd.setSeed(newRnd.getSeed)
      val (h, beta) = addNodeForConvexUpdate(weights, b, X, t, e)
      biases(l) = b
      updateNetwork(l, weights, beta, Beta, Alfat)
      l += 1
    }
    ELMSimpleModel(rnd, Alfat, biases, Beta, X, e, t)
  }
}

object CIELMincTest extends App {
  //  val patts0 = new Random(0).shuffle(Datasets.patternsFromSQLite("/home/davi/wcs/ucipp/uci")("gas-drift").right.get.take(1000000))
  //  val patts0 = new Random(0).shuffle(Datasets.arff(true)("/home/davi/wcs/ucipp/uci/banana.arff").right.get.take(200000))
  //    val patts0 = new Random(0).shuffle(Datasets.arff(true)("/home/davi/wcs/ucipp/uci/iris.arff").right.get.take(200000))
  val patts0 = new Random(0).shuffle(Datasets.arff(true)("/home/davi/wcs/ucipp/uci/abalone-11class.arff").right.get.take(200000))
  val filter = Datasets.zscoreFilter(patts0)
  val patts = Datasets.applyFilterChangingOrder(patts0, filter)

  val n = patts.length / 2
  val tr = patts.take(n)
  val ts = patts.drop(n)

  val l = NB()
  //KNN(5,"eucl",patts)
  val tt = patts.head.nclasses
  Tempo.start
  var m = CIELM(n).build(tr.take(tt))
  tr.drop(tt).foreach { x =>
    m = CIELM(n).update(m)(x)
    println(s"${m.accuracy(ts)}")
  }
  Tempo.print_stop

  //  Tempo.start
  //  var m2 = l.build(tr.take(tt))
  //  tr.drop(tt).foreach(x => m2 = l.update(m2)(x))
  //  Tempo.print_stop
  //  println(s"${m2.accuracy(ts)}")
  //
  //  Tempo.start
  //  m = CIELM(n).build(tr)
  //  Tempo.print_stop
  //  println(s"${m.accuracy(ts)}")
  //
  //  println("")
  //
  //  Tempo.start
  //  m = CIELM(n).build(tr.take(tt))
  //  tr.drop(tt).foreach(x => m = CIELM(n).update(m)(x))
  //  Tempo.print_stop
  //  println(s"${m.accuracy(ts)}")
  //
  //  Tempo.start
  //  m2 = l.build(tr.take(tt))
  //  tr.drop(tt).foreach(x => m2 = l.update(m2)(x))
  //  Tempo.print_stop
  //  println(s"${m2.accuracy(ts)}")
  //
  //  Tempo.start
  //  m = CIELM(n).build(tr)
  //  Tempo.print_stop
  //  println(s"${m.accuracy(ts)}")

}