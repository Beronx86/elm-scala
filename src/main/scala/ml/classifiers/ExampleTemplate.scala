package ml.classifiers

import java.io.File

import ml.Pattern
import util.Datasets

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
trait ExampleTemplate {
  val dataset: String
  val k: Int
  val parallel: Boolean
  lazy val data = (Datasets.arff(bina = true)(dataset + ".arff") match {
    case Right(x) => x
    case Left(str) => println("Could not load banana dataset from the program path: " + str); sys.exit(0)
  }).take(1000)

  def kfoldIteration[T](tr: => Seq[Pattern], ts: => Seq[Pattern], fold: Int, bla: Int)

  def run {
    println(
      """
 elm-scala Copyright (C) 2014 Davi Pereira dos Santos

 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to redistribute it
 under certain conditions.
 Refer to LICENSE file for details.
      """)
    val appPath = new File(".").getCanonicalPath + "/"
    println(appPath)

    println("Warming up JVM-BLAS interface...")
    val warmingdata = Datasets.arff(bina = true)("banana.arff") match {
      case Right(x) => x
      case Left(str) => println("Could not load iris dataset from the program path: " + str); sys.exit(0)
    }
    val currentSeed = (System.currentTimeMillis() % 1000000).toInt
    IELM(seed = currentSeed).build(warmingdata)

    util.Datasets.kfoldCV(data, k, parallel)(kfoldIteration)
  }
}
