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
    //    if (math.sqrt(m.N + 1).toInt > math.sqrt(m.N).toInt) build(?) //<- not possible to rebuild, since we don't have Y nor all patterns anymore. But do we have t? No.
    val newE = m.e.zip(pattern.weighted_label_array) map { case (dv, v) => Data.appendToVector(dv, v)}
    val newX = Data.appendRowToMatrix(m.X, pattern.array)
    val newTmp = new DenseVector(newX.numRows())

    val (weights, bias, newRnd) = newNode(m.Alfat.numColumns(), m.rnd)
    val newAlfat = Data.appendRowToMatrix(m.Alfat, weights)
    val newBiases = Data.appendToArray(m.biases, bias)
    val (h, beta) = addNode(weights, bias, newX, newE, newTmp)
    val newBeta = Data.appendRowToMatrix(m.Beta, beta)

    ELMSimpleModel(newRnd, newAlfat, newBiases, newBeta, newX, newE, null)
  }

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector) = {
    val (weights, bias, newRnd) = newNode(X.numColumns(), rnd)
    rnd.setSeed(newRnd.getSeed)
    val (h, beta) = addNode(weights, bias, X, e, tmp)
    (weights, bias, h, beta)
  }
}


object IELMincTest extends App {
  //  val patts0 = new Random(0).shuffle(Datasets.patternsFromSQLite("/home/davi/wcs/ucipp/uci")("gas-drift").right.get.take(1000000))
  //  val patts0 = new Random(0).shuffle(Datasets.arff(true)("/home/davi/wcs/ucipp/uci/iris.arff").right.get.take(200000))
  val patts0 = new Random(650).shuffle(Datasets.arff(true)("/home/davi/wcs/ucipp/uci/abalone-3class.arff").right.get.take(2000))
  val filter = Datasets.zscoreFilter(patts0)
  val patts = Datasets.applyFilterChangingOrder(patts0, filter)

  val n = patts.length / 2
  val initialN = patts.head.nclasses
  val tr = patts.take(n)
  val ts = patts.drop(n)

  val li = IELM()
  val lis = IELMScratch()
  val lei = EIELM()
  val lie = IELMEnsemble(10)
  val lci = CIELM()
  var mi = li.build(tr.take(initialN))
  var mis = lis.build(tr.take(initialN))
  var mei = lei.build(tr.take(initialN))
  var mie = lie.build(tr.take(initialN))
  var mci = lci.build(tr.take(initialN))
  tr.drop(initialN).foreach { x =>
    mi = li.update(mi)(x)
    mis = lis.update(mis)(x)
    mei = lei.update(mei)(x)
    mie = lie.update(mie)(x)
    mci = lci.update(mci)(x)
    println(s"${mi.accuracy(ts)} ${mis.accuracy(ts)} ${mei.accuracy(ts)} ${mie.accuracy(ts)} ${mci.accuracy(ts)}")
  }


  //total times and accs
  //  Tempo.t {
  //    val l = IELM(1)
  //    var m = l.build(tr.take(initialN))
  //    tr.drop(initialN).foreach { x => m = l.update(m)(x)}
  //    println(s"${m.accuracy(ts)}")
  //  }
  //
  //  Tempo.t {
  //    val l = IELMScratch(1)
  //    var m = l.build(tr.take(initialN))
  //    tr.drop(initialN).foreach { x => m = l.update(m)(x)}
  //    println(s"${m.accuracy(ts)}")
  //  }
}

