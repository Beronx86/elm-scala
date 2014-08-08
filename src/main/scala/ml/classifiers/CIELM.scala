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

  /**
   * Não altera rnd.
   * @param rnd
   * @param X
   * @param e
   * @param t
   * @return
   */
  def grow(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], t: Vector[DenseVector]) = {
    val natts = X.numColumns()

    val (weights, bias, newRnd) = newNode(natts, rnd)
    val (h, beta) = addNodeForConvexUpdate(weights, bias, X, t, e)

    (weights, bias, h, beta, newRnd)
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