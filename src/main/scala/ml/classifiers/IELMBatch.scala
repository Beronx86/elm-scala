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
import ml.neural.elm.{Data, IELMTrait}
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.{Tempo, Datasets, XSRandom}

import scala.util.Random

/**
 * Non-incremental.
 * 230 vezes mais lento que IELM (se crescer 1 a cada exemplo)
 * @param seed
 * @param notes
 * @param callf
 * @param f
 */
case class IELMBatch(seed: Int = 42, callf: Boolean = false, f: (Model, Double) => Unit = (_, _) => ())
  extends IELMTrait {
   override val toString = "IELMBatch"
   val id = 99801
  val Lbuild = -1
  val abr = toString

   override def build(trSet: Seq[Pattern]) = {
      val nclasses = trSet.head.nclasses
      val n = trSet.size
      if (trSet.size < nclasses) {
         println("At least |Y| instances required.")
         sys.exit(1)
      }
      val natts = trSet.head.nattributes
      val X = patterns2matrix(trSet, n)
      val e = patterns2t(trSet, n)
      bareBuild(n, natts, nclasses, X, e, trSet)
  }

  def update(model: Model, fast_mutable: Boolean)(pattern: Pattern) = {
    val trSet = pattern +: cast(model).patterns
    val nclasses = pattern.nclasses
    val ninsts = checkEmptyness(trSet: Seq[Pattern])
    val natts = trSet.head.nattributes
    val X = patterns2matrix(trSet, ninsts)
    val e = patterns2t(trSet, ninsts)
    bareBuild(ninsts, natts, nclasses, X, e, trSet)
  }

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector) = {
    val (weights, bias, newRnd) = newNode(X.numColumns(), rnd)
    rnd.setSeed(newRnd.getSeed)
    val (h, beta) = addNode(weights, bias, X, e, tmp)
    (weights, bias, h, beta)
  }
}


