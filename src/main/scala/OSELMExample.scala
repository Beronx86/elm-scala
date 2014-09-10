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
object OSELMExample extends App {
  println( """
 elm-scala Copyright (C) 2014 Davi Pereira dos Santos

 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to redistribute it
 under certain conditions.
 Refer to LICENSE file for details.
           """)
  val appPath = new File(".").getCanonicalPath + "/"
  println(appPath)

  println("Warming up JVM-BLAS interface...")
  val warmingdata = Datasets.arff("banana.arff") match {
    case Right(x) => x
    case Left(str) => println("Could not load iris dataset from the program path: " + str); sys.exit(1)
  }
  val currentSeed = (System.currentTimeMillis() % 1000000).toInt
  IELM(seed = currentSeed).build(warmingdata)


  val data = Datasets.arff("banana.arff") match {
    case Right(x) => x
    case Left(str) => println("Could not load banana dataset from the program path: " + str); sys.exit(1)
  }

  util.Datasets.kfoldCV(data, k = 10, parallel = true) { (trainingSet, testingSet, fold, _) =>
    val elm = OSELM(L = 16, seed = currentSeed)
    val oselm = OSELM(L = 16, seed = currentSeed)

    val (model, t) = Tempo.timev(elm.build(trainingSet))
    val acc = model.accuracy(testingSet).formatted("%2.2f")

    val (osmodel, ost) = Tempo.timev {
      val firstModel = oselm.build(trainingSet.take(20))
      trainingSet.drop(20).foldLeft(firstModel)((m, p) => ???) //oselm.update(m)(p))
    }
    val osacc = osmodel.accuracy(testingSet).formatted("%2.2f")
    println("Fold " + fold + ".  OSELM: " + osacc + " in " + ost + "ms.    " + "ELM: " + acc + " in " + t + "ms.")
  }

  println("Note that OS-ELM can be faster than ELM due to cache scarcity in the processor." +
    "When there are no numerical instability, they should behave exactly the same in terms of accuracy.")

  println("Testing equality of PRESS and LOO...")
  val d = data.take(500)
  val LOOerror = Datasets.LOO(d) { (tr, p) =>
    if (OSELM(10) build tr hit p) 0 else 1
  }.sum / d.length.toDouble
  println("LOO error: " + LOOerror + " \nPRESSLOO: ")
  interaELM().build(d)
  val em = EMELM(10)
  val me = em.growTo(10, em.build(d))
  val os = OSELM(10)
  val mo = os.build(d)
  println("EMPRESSLOO: " + em.LOOError(me))
  println("OSPRESSLOO: " + os.LOOError(mo))

}