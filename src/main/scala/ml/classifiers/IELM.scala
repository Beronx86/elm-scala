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
import ml.neural.elm.{Data, IELMTrait}
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.{Tempo, Datasets, XSRandom}

import scala.util.Random

case class IELM(seed: Int = 42, notes: String = "", callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ())
  extends IELMTrait {
  override val toString = "IELM_" + notes
  val Lbuild = -1

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val m = cast(model)
    val newE = m.e.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
    val newX = Data.appendRowToMatrix(m.X, pattern.array)
    val newTmp = new DenseVector(newX.numRows())

    val (weights, bias, newRnd) = newNode(m.Alfat.numColumns(), m.rnd)
    val newAlfat = Data.appendRowToMatrix(m.Alfat, weights)
    val newBiases = Data.appendToArray(m.biases, bias)
    val (h, beta) = addNode(weights, bias, newX, newE, newTmp)
    val newBeta = Data.appendRowToMatrix(m.Beta, beta)

    ELMSimpleModel(newRnd, newAlfat, newBiases, newBeta, newX, newE)
  }

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector) = {
    val (weights, bias, newRnd) = newNode(X.numColumns(), rnd)
    rnd.setSeed(newRnd.getSeed)
    val (h, beta) = addNode(weights, bias, X, e, tmp)
    (weights, bias, h, beta)
  }

  def addNode(weights: Array[Double], bias: Double, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector) = {
    val nclasses = e.size
    //Generate node and calculate h.
    val alfa = new DenseVector(weights, false)
    val beta = new Array[Double](nclasses)
    val h = feedHidden(X, alfa, bias)
    var o = 0
    while (o < nclasses) {
      tmp.set(h)
      //Calculate new weight.
      val nume = e(o).dot(h)
      val deno = h.dot(h)
      val b = nume / deno
      beta(o) = b

      //Recalculate residual error.
      tmp.scale(-b)
      e(o).add(tmp)

      o += 1
    }
    (h, beta)
  }
}


object IELMincTest extends App {
  //  val patts0 = new Random(0).shuffle(Datasets.patternsFromSQLite("/home/davi/wcs/ucipp/uci")("gas-drift").right.get.take(1000000))
  val patts0 = new Random(0).shuffle(Datasets.arff(true)("/home/davi/wcs/ucipp/uci/banana.arff").right.get.take(200000))
  //  val patts0 = new Random(0).shuffle(Datasets.arff(true)("/home/davi/wcs/ucipp/uci/iris.arff").right.get.take(200000))
  val filter = Datasets.zscoreFilter(patts0)
  val patts = Datasets.applyFilterChangingOrder(patts0, filter)

  val n = patts.length / 2
  val tr = patts.take(n)
  val ts = patts.drop(n)

  val l = NB() //KNN(5,"eucl",patts)

  Tempo.start
  var m = IELM(n).build(tr.take(2))
  tr.drop(2).foreach(x => m = IELM(n).update(m)(x))
  Tempo.print_stop
  println(s"${m.accuracy(ts)}")

  Tempo.start
  var m2 = l.build(tr.take(2))
  tr.drop(2).foreach(x => m2 = l.update(m2)(x))
  Tempo.print_stop
  println(s"${m2.accuracy(ts)}")

  Tempo.start
  m = IELM(n).build(tr)
  Tempo.print_stop
  println(s"${m.accuracy(ts)}")

  println("")

  Tempo.start
  m = IELM(n).build(tr.take(2))
  tr.drop(2).foreach(x => m = IELM(n).update(m)(x))
  Tempo.print_stop
  println(s"${m.accuracy(ts)}")

  Tempo.start
  m2 = l.build(tr.take(2))
  tr.drop(2).foreach(x => m2 = l.update(m2)(x))
  Tempo.print_stop
  println(s"${m2.accuracy(ts)}")

  Tempo.start
  m = IELM(n).build(tr)
  Tempo.print_stop
  println(s"${m.accuracy(ts)}")

}