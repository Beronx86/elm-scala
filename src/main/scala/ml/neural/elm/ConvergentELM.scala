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
import ml.models.{ELMConvergentModel, ELMModel, Model}
import ml.neural.elm.Data._
import no.uib.cipr.matrix.DenseMatrix
import util.XSRandom

/**
 * Created by davi on 21/05/14.
 */
trait ConvergentELM extends ELM {
  val Lbuild: Int

  protected def cast(model: Model) = model match {
    case m: ELMModel => m
    case _ => println("Convergent ELMs require ELMModels.")
      sys.exit(0)
  }

  def build(trSet: Seq[Pattern]): ELMModel = {
    val rnd = new XSRandom(seed)
    val ninsts = checkEmptyness(trSet)
    checkFullRankness(ninsts)
    val t = patterns2matrices(trSet, ninsts)
    val Xt = t._1
    val Y = t._2
    buildCore(Lbuild, Xt, Y, rnd)
  }

  protected def buildCore(L: Int, Xt: DenseMatrix, Y: DenseMatrix, rnd: XSRandom) = {
    val ninsts = Xt.numColumns()
    val natts = Xt.numRows()
    val nclasses = Y.numColumns()
    val biasesArray = new Array[Double](L)
    //    val Alfa = new DenseMatrix(L, natts)
    val Alfat = new DenseMatrix(L, natts)
    initializeWeights(Alfat, biasesArray, rnd)

    val tupleHHt = ELMUtils.feedHiddent(Xt, Alfat, biasesArray)
    val H = tupleHHt._1
    val Ht = tupleHHt._2
    val P = ELMUtils.calculateP(H, Ht)

    val Hinv = new DenseMatrix(L, ninsts)
    P.mult(Ht, Hinv)
    val Beta = new DenseMatrix(L, nclasses)
    Hinv.mult(Y, Beta)

    ELMConvergentModel(rnd, Alfat, biasesArray, Beta, H, Ht, Hinv, P, ninsts, Xt, Y)
  }

  /**
   * This is not a problem for I-ELM and variants, but is needed for OS-ELM (and EM-ELM?).
   * At the moment the test is fast, i.e. it doesn't really check the rank.
   * @param ninsts
   * @return
   */
  protected def checkFullRankness(ninsts: Int) {
    if (ninsts < Lbuild) {
      println("ERROR: Training set size (" + ninsts + ") is lesser than L (" + Lbuild + ")!")
      sys.exit(0)
    }
  }

  def LOOError(model: Model): Double = {
    val m = cast(model)
    LOOError(m.Y)(errorMatrix(m.H, m.Beta, m.Y))(m.HHinv)
  }

  /**
   * Calculates fast LOO accuracy over all instances using PRESS.
   * (for classifiers)
   * (assumes the correct output is the only one 1-valued)
   * @param Y matrix of expected values NxO
   * @param E matrix of errors (difference between expected and predicted)
   * @param HHinv
   * @return
   */
  protected def LOOError(Y: DenseMatrix)(E: DenseMatrix)(HHinv: DenseMatrix) = {
    val n = HHinv.numRows()
    val nclasses = E.numColumns()
    val PredictionMatrix = PREdictions(Y)(E)(HHinv)

    var c = 0
    var i = 0
    var max = 0d
    var cmax = 0
    var hits = 0
    while (i < n) {
      cmax = -1
      max = Double.MinValue
      c = 0
      while (c < nclasses) {
        val v = PredictionMatrix.get(i, c)
        if (v > max) {
          cmax = c
          max = v
        }
        c += 1
      }
      if (Y.get(i, cmax) == 1) hits += 1
      i += 1
    }
    1 - hits / n.toDouble
  }

  /**
   * Calculates individual predictions (from PRESS statistic) for each instance for each output/class.
   * @param Y matrix of expected values NxO
   * @param E matrix of errors (difference between expected and predicted)
   * @param HHinv
   * @return
   */
  protected def PREdictions(Y: DenseMatrix)(E: DenseMatrix)(HHinv: DenseMatrix) = {
    val PredictionMatrix = Y.copy()
    PredictionMatrix.add(-1, PRESSMatrix(E)(HHinv))
    PredictionMatrix
  }

  def PRESS(model: Model): Double = {
    val m = cast(model)
    PRESS(errorMatrix(m.H, m.Beta, m.Y))(m.HHinv)
  }

  /**
   * Calculates the PRESS statistic (Prediction REsidual Sum of Squares) for all outputs/classes.
   * (usually for regressors)
   * @param E matrix of errors (difference between expected and predicted)
   * @param HHinv
   * @return
   */
  protected def PRESS(E: DenseMatrix)(HHinv: DenseMatrix) = {
    //todo: this can be more efficient, because the loop is also inside PRESSMatrix()
    val n = HHinv.numRows()
    val nclasses = E.numColumns()
    val M = PRESSMatrix(E)(HHinv)
    var sum = 0d
    var i = 0
    while (i < n * nclasses) {
      sum += M.getData()(i)
      i += 1
    }
    sum
  }

  /**
   * Calculates individual PRESS for each instance for each output/class.
   * @param E matrix of errors (difference between expected and predicted)
   * @param HHinv HAT matrix
   * @return N x O matrix of individual PRESS values for each output
   */
  protected def PRESSMatrix(E: DenseMatrix)(HHinv: DenseMatrix) = {
    val nclasses = E.numColumns()
    val M = new DenseMatrix(E.numRows(), E.numColumns())
    val n = HHinv.numRows()
    var c = 0
    var i = 0
    while (c < nclasses) {
      i = 0
      while (i < n) {
        M.set(i, c, fPRESS(HHinv.get(i, i))(E.get(i, c)))
        i += 1
      }
      c += 1
    }
    M
  }

  /**
   * Calculates an instance contribution to the PRESS statistic.
   * @param HATvalue value in the respective position at the diagonal of HHinv
   * @param error difference between expected and predicted for the respective output
   * @return
   */
  protected def fPRESS(HATvalue: Double)(error: Double) = error / (1 - HATvalue)

  //  protected def cast(model: Model) = model match {
  //    case m: ELMGenericModel => m
  //    case _ => println("Convergent ELMs require ELMGenericModel.")
  //      sys.exit(0)
  //  }

  protected def errorMatrix(H: DenseMatrix, Beta: DenseMatrix, Y: DenseMatrix) = {
    val Prediction = new DenseMatrix(H.numRows(), Beta.numColumns())
    val E = Y.copy()
    ELMUtils.feedOutput(H, Beta, Prediction)
    E.add(-1, Prediction)
    E
  }

}

//rnd ok