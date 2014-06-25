import ml.classifiers._
import ml.classifiers.interaELMExample._
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
object EMELMExample extends App {
  println( """
 elm-scala Copyright (C) 2014 Davi Pereira dos Santos

 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to redistribute it
 under certain conditions.
 Refer to LICENSE file for details.
           """)

  println("Warming up JVM-BLAS interface...")
  val warmingdata = Datasets.arff(bina = true)("banana.arff") match {
    case Right(x) => x
    case Left(str) => println("Could not load iris dataset from the program path: " + str); sys.exit(0)
  }
  val currentSeed = (System.currentTimeMillis() % 1000000).toInt
  IELM(initialL = 15, seed = currentSeed).build(warmingdata)

  println("seed " + currentSeed)
  val dataset = "banana.arff"
  val data = (Datasets.arff(bina = true)(dataset) match {
    case Right(x) => x
    case Left(str) => println("Could not load " + dataset + " dataset from the program path: " + str); sys.exit(0)
  }).take(1000)

  util.Datasets.kfoldCV(data, k = 10, parallel = true) { (trainingSet, testingSet, fold, _) =>
    val elm = OSELM(L = 16, seed = currentSeed)
    val emelm = EMELM(Lmax = 1, seed = currentSeed)

    val (model, t) = Tempo.timev(elm.build(trainingSet))
    val acc = model.accuracy(testingSet).formatted("%2.2f")

    val (emmodel, ost) = Tempo.timev {
      val m = emelm.build(trainingSet)
      emelm.growTo(16, m)
    }
    val emacc = emmodel.accuracy(testingSet).formatted("%2.2f")
    println("Fold " + fold + ".  EMELM: " + emacc + " in " + ost + "ms.    " + "ELM: " + acc + " in " + t + "ms.")
  }


//  val LOOos = Datasets.LOO(data) { (tr, p) =>
//    if (OSELM(20) build tr hit p) 1 else 0
//  }.sum / data.length.toDouble
//
//  val LOOem = Datasets.LOO(data) { (tr, p) =>
//    if (EMELM(20) build tr hit p) 1 else 0
//  }.sum / data.length.toDouble
//
//  println("os "+ LOOos)
//  println("em "+ LOOem)

  println("Note that OS-ELM can be faster than ELM due to cache scarcity in the processor." +
    "When there are no numerical instability, they should behave exactly the same in terms of accuracy.")
}

//rnd ok