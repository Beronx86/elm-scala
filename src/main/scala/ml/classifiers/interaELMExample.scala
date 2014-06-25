package ml.classifiers

import java.io.File

import ml.Pattern
import ml.classifiers._
import util.{Tempo, Datasets}

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
object interaELMExample extends App with ExampleTemplate {
  val dataset =
    "banana"
  "iris"

  def kfoldIteration[T](tr: Seq[Pattern], ts: Seq[Pattern], fold: Int, bla: Int) {
    val e = interaELM(data.length / 2)
    val m = e.build(tr)
    println(m.accuracy(ts))
  }

  //  run

  println("")
  val m = interaELM(20).build(data)

}
