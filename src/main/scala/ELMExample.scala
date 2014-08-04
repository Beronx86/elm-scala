import java.io.File

import ml.classifiers._
import util.{Datasets, Tempo}

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
object ELMExample extends App {
  println( """
 elm-scala Copyright (C) 2014 Davi Pereira dos Santos

 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to redistribute it
 under certain conditions.
 Refer to LICENSE file for details.
           """)

  val warmingdata = Datasets.arff(bina = true)("banana.arff") match {
    case Right(x) => x
    case Left(str) => println("Could not load banana dataset from the program path: " + str); sys.exit(1)
  }
  val currentSeed = (System.currentTimeMillis() % 1000000).toInt

  val elms = Seq(
    IELM(seed = currentSeed),
    EIELM(seed = currentSeed),
    CIELM(seed = currentSeed),
    ECIELM(seed = currentSeed),
    OSELM(L = 16, seed = currentSeed)
  )
  val appPath = new File(".").getCanonicalPath + "/"
  println(appPath)

  println("Warming up JVM-BLAS interface...")
  IELM(seed = currentSeed).build(warmingdata)

  Seq("banana.arff", "iris.arff") foreach { dataset =>
    println("Comparing all ELMs in " + dataset + " dataset...")
    val data = Datasets.arff(bina = true)(dataset) match {
      case Right(x) => x
      case Left(str) => println("Could not load iris dataset from the program path: " + str); sys.exit(1)
    }
    elms foreach { elm =>
      println(elm)

      util.Datasets.kfoldCV(data, k = 3, parallel = true) { (trainingSet, testingSet, fold, _) =>
        val (model, t) = Tempo.timev(elm.build(trainingSet))
        val acc = model.accuracy(testingSet).formatted("%2.2f")
        println("Fold " + fold + ": " + acc + " in " + t + "ms.")
      }

      println("")
    }
  }
}

//rnd ok
