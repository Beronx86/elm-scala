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
  val dataset =    "banana"
  // "iris"
  val k = 2
  def kfoldIteration[T](tr: Seq[Pattern], ts: Seq[Pattern], fold: Int, bla: Int) {
    val i = interaELM(16)
    val mi = i.build(tr)

    val c = C45()
    val mc = c.build(tr)

    val o = OSELM(16)
    val mo = o.build(tr)

    val e = EMELM(166)
    var me = e.build(tr)
    me = e.growTo(16,me)

    println("i: " + mi.accuracy(ts) + "\tc: " + mc.accuracy(ts) + "\to: " + mo.accuracy(ts) + "\te: " + me.accuracy(ts))
  }

  run
}
