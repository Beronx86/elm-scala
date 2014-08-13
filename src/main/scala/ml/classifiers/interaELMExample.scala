package ml.classifiers

import ml.Pattern
import ml.models.ELMModel
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

  def kfoldIteration[T](tr0: => Seq[Pattern], ts: => Seq[Pattern], fold: Int, bla: Int) {
    val tr = tr0.take(31533)

    //warming up
    val wii = interaELM(l)
    tr.drop(2).foldLeft(wii.build(tr.take(2)))((m, p) => wii.update(m)(p))

    Tempo.start
    //todo: pq mii ta ficando com L=37 e 73? O press está gigante 10^60
    val ii = interaELM(l)
    val mii = tr.drop(50).foldLeft(ii.build(tr.take(50)))((m, p) => ii.update(m)(p))
    Tempo.print_stop

    Tempo.start
    val i = interaELM(l)
    val mi = i.build(tr)
    Tempo.print_stop


    val c = C45()
    val mc = c.build(tr)

    val o = OSELM(l)
    val mo = o.build(tr)

    val e = EMELM(1)
    var me = e.build(tr)
    me = e.growTo(l, me)

    lazy val LOOi = Datasets.LOO(tr) { (trloo, p) =>
      if (interaELM(l) build trloo hit p) 0 else 1
    }.sum / tr.length.toDouble

    lazy val LOOos = Datasets.LOO(tr) { (trloo, p) =>
      if (OSELM(l) build trloo hit p) 0 else 1
    }.sum / tr.length.toDouble
    //        println("LOOPRESSos: " + o.LOOError(mo) + "  LOOPRESSi: " + i.LOOError(mi) + "  LOOos: " + LOOos + "  LOOi: " + LOOi)

    println(s"ii(${mii.asInstanceOf[ELMModel].L}): ${mii.accuracy(ts)} \ti(${mi.asInstanceOf[ELMModel].L}): ${mi.accuracy(ts)} \tc(-}): ${mc.accuracy(ts)}\to(${mo.asInstanceOf[ELMModel].L}): ${mo.accuracy(ts)}\te(${me.asInstanceOf[ELMModel].L}): " + me.accuracy(ts))
    println(s"PRESSii: ${ii.PRESS(mii)} PRESSi: ${i.PRESS(mi)} PRESSos: ${o.PRESS(mo)} PRESSem: ${e.PRESS(me)}")
  }

  run
  println("o menor LOO (p. ex. no model selection do interaELM) não leva necessariamente a menor acuracia no 5-fold CV (o conjunto usado no PRESS pode ser muito inferior ao cjt disponivel)")
}
