package ml.classifiers

import ml.Pattern
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
object interaELMExample extends App with ExampleTemplate {
  val parallel = false
  val dataset = "banana"
  // "iris"
  val k = 2
  val l = 20

  def kfoldIteration[T](tr0: Seq[Pattern], ts: Seq[Pattern], fold: Int, bla: Int) {
    val tr = tr0.take(333)

    Tempo.start
    val i = interaELM(l)
    val mi = i.build(tr)
    Tempo.print_stop

    Tempo.start
    val i2 = interaELMNoEM(l)
    val mi2 = i2.build(tr)
    Tempo.print_stop

    val c = C45()
    val mc = c.build(tr)

    val o = OSELM(l)
    val mo = o.build(tr)
    println(mo.Beta)

    val e = EMELM(1)
    var me = e.build(tr)
    me = e.growTo(l, me)
    println(me.Beta)

    lazy val LOOi = Datasets.LOO(tr) { (trloo, p) =>
      if (interaELM(l) build trloo hit p) 0 else 1
    }.sum / tr.length.toDouble

    lazy val LOOos = Datasets.LOO(tr) { (trloo, p) =>
      if (OSELM(l) build trloo hit p) 0 else 1
    }.sum / tr.length.toDouble
    //        println("LOOPRESSos: " + o.LOOError(mo) + "  LOOPRESSi: " + i.LOOError(mi) + "  LOOos: " + LOOos + "  LOOi: " + LOOi)

    println(s"i(${mi.L}): ${mi.accuracy(ts)} \ti2(${mi2.L}): ${mi2.accuracy(ts)} \tc(-}): ${mc.accuracy(ts)}\to(${o.L}): ${mo.accuracy(ts)}\te(${me.L}): " + me.accuracy(ts))
    println(s"PRESSi: ${i.PRESS(mi)} PRESSi2: ${i2.PRESS(mi2)} PRESSos: ${o.PRESS(mo)} PRESSem: ${e.PRESS(me)}")
  }

  run
  println("o menor LOO (p. ex. no model selection do interaELM) não leva necessariamente a menor acuracia no 5-fold CV")
}
