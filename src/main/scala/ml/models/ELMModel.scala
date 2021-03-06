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

package ml.models

import ml.Pattern
import ml.neural.elm.{ELMUtils, Math}
import no.uib.cipr.matrix.{DenseVector, DenseMatrix}
import util.XSRandom

trait ELMModel extends Model {
   def JS(pattern: Pattern) = ???
   lazy val HHinv = {
      val r = new DenseMatrix(H.numRows(), H.numRows())
      H.mult(Hinv, r)
      r
   }
   lazy val L = Alfat.numRows()
   lazy val I = Math.identity(N)
   val rnd: XSRandom
   //needed by test()
   val biases: Array[Double]
   val Alfat: DenseMatrix
   val Beta: DenseMatrix
   //needed by P and Hinv
   val H: DenseMatrix
   val Ht: DenseMatrix
   //needed by grow()
   val Xt: DenseMatrix
   val N: Int
   val Hinv: DenseMatrix
   //needed by modelSelection() and grow()
   val Y: DenseMatrix
   //needed by update()
   val P: DenseMatrix

   def output(pattern: Pattern) = ELMUtils.output(pattern, Alfat, biases, Beta).getData

   def predict(pattern: Pattern) = {
      val o = ELMUtils.test(pattern, Alfat, biases, Beta).getData
      val n = o.size
      var i = 0
      var c = 0
      var v = Double.MinValue
      while (i < n) {
         if (o(i) > v) {
            v = o(i)
            c = i
         }
         i += 1
      }
      c
   }

   def distribution(pattern: Pattern) = ELMUtils.distribution(ELMUtils.test(pattern, Alfat, biases, Beta))
}

//todo: ELMSimpleModel does not need rnd nor N; I and CI-ELM need updateable versions and the creation of proper specific ELMXXXXModels
case class ELMSimpleModel(rnd: XSRandom, Alfat: DenseMatrix, biases: Array[Double], Beta: DenseMatrix, X: DenseMatrix, e: Vector[DenseVector], t: Vector[DenseVector], patterns: Seq[Pattern] = Seq()) extends ELMModel {
   val H = null
   val Ht = null
   val Hinv = null
   val P = null
   val Y = null
   val Xt = null
   val N = X.numRows()
}

case class ELMSimpleEnsembleModel(rnd: XSRandom, AlfatS: Seq[DenseMatrix], biasesS: Seq[Array[Double]], BetaS: Seq[DenseMatrix], X: DenseMatrix, eS: Seq[Vector[DenseVector]], t: Vector[DenseVector]) extends Model {
   def JS(pattern: Pattern) = ???
   lazy val L = BetaS.head.numRows()
   lazy val M = AlfatS.size
   lazy val O = BetaS.head.numColumns()
   val N = X.numRows()

   def predict(pattern: Pattern) = output(pattern).zipWithIndex.max._2

   def distribution(pattern: Pattern) = {
      var o = 0
      var m = 0
      val res = Array.fill(O)(0d)
      while (m < M) {
         val dis = ELMUtils.distribution(ELMUtils.test(pattern, AlfatS(m), biasesS(m), BetaS(m)))
         o = 0
         while (o < O) {
            res(o) += dis(o)
            o += 1
         }
         m += 1
      }
      o = 0
      while (o < O) {
         res(o) /= M
         o += 1
      }
      res
   }

   def output(pattern: Pattern) = {
      var o = 0
      var m = 0
      val res = Array.fill(O)(0d)
      while (m < M) {
         val dis = ELMUtils.test(pattern, AlfatS(m), biasesS(m), BetaS(m)).getData
         o = 0
         while (o < O) {
            res(o) += dis(o)
            o += 1
         }
         m += 1
      }
      res
   }
}

//todo: Xt,Y -> queue[Pattern ou (array,array)] (pra evitar copias de Xt e Y inteiros na memoria; Xt e Y só vão ser needed ao final dos incrementos, podem ser criados inteiros de uma vez from queue)
case class ELMIncModel(rnd: XSRandom, Alfat: DenseMatrix, biases: Array[Double], Beta: DenseMatrix,
                       P: DenseMatrix, N: Int, Xt: DenseMatrix, Y: DenseMatrix) extends ELMModel {
   lazy val H = tupleHHt._1
   lazy val Ht = tupleHHt._2
   lazy val Hinv = {
      val m = new DenseMatrix(L, N)
      P.mult(Ht, m)
      m
   }
   private lazy val tupleHHt = ELMUtils.feedHiddent(Xt, Alfat, biases)
}

case class ELMGroModel(rnd: XSRandom, Alfat: DenseMatrix, biases: Array[Double], Beta: DenseMatrix,
                       Xt: DenseMatrix, Y: DenseMatrix, H: DenseMatrix, Hinv: DenseMatrix) extends ELMModel {
   lazy val P = ELMUtils.calculateP(H, Ht)
   lazy val Ht = {
      val m = new DenseMatrix(H.numColumns(), H.numRows())
      H.transpose(m)
      m
   }
   val N = H.numRows()
}

case class ELMConvergentModel(rnd: XSRandom, Alfat: DenseMatrix, biases: Array[Double], Beta: DenseMatrix,
                              H: DenseMatrix, Ht: DenseMatrix, Hinv: DenseMatrix, P: DenseMatrix, N: Int, Xt: DenseMatrix, Y: DenseMatrix) extends ELMModel
