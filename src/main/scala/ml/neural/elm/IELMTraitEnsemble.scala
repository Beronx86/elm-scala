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
import ml.models.{ELMSimpleEnsembleModel, ELMSimpleModel, Model}
import ml.mtj.ResizableDenseMatrix
import ml.neural.elm.Data._
import no.uib.cipr.matrix.{DenseMatrix, DenseVector}
import util.{Tempo, XSRandom}

/**
 * Created by davi on 21/05/14.
 */
trait IELMTraitEnsemble extends IteratedBuildELM {
  val M: Int

  def bareBuild(ninsts: Int, natts: Int, nclasses: Int, X: DenseMatrix, eS: Seq[Vector[DenseVector]]) = {
    val L = nclasses
    val biasesS = Seq.fill(M)(Array.fill(L)(0d))
    val AlfatS = Seq.fill(M)(new ResizableDenseMatrix(L, natts))
    val BetaS = Seq.fill(M)(new ResizableDenseMatrix(L, nclasses))
    val rnd = new XSRandom(seed)

    var m = 0
    while (m < M) {
      var l = 0
      val tmp = new DenseVector(ninsts)
      while (l < L) {
        val (weights, bias, h, beta) = buildCore(rnd, X, eS(m), tmp)
        biasesS(m)(l) = bias
        AlfatS(m).setRow(l, weights)
        BetaS(m).setRow(l, beta)
        l += 1
      }
      m += 1
    }
    ELMSimpleEnsembleModel(rnd, AlfatS, biasesS, BetaS, X, eS, null)
  }

  def build(trSet: Seq[Pattern]) = {
    val nclasses = trSet.head.nclasses
    if (trSet.size < nclasses) {
      println("At least |Y| instances required.")
      sys.exit(0)
    }
    val initialTrSet = trSet.take(nclasses)
    val natts = initialTrSet.head.nattributes
    val X = patterns2matrix(initialTrSet, nclasses)
    val eS = Seq.fill(M)(patterns2t(initialTrSet, nclasses))
    val firstModel = bareBuild(nclasses, natts, nclasses, X, eS)
    trSet.drop(nclasses).foldLeft(firstModel)((m, p) => ensCast(update(m, fast_mutable = true)(p)))
  }

  protected def buildCore(rnd: XSRandom, X: DenseMatrix, e: Vector[DenseVector], tmp: DenseVector): (Array[Double], Double, DenseVector, Array[Double])

  protected def ensCast(model: Model) = model match {
    case m: ELMSimpleEnsembleModel => m
    case _ => println("IELMTraitEnsemble ELMs require ELMSimpleEnsembleModels.")
      sys.exit(0)
  }
}