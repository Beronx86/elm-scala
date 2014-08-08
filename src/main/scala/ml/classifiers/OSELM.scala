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

package ml.classifiers

/**
 * Updates weights at each instance arrival.
 * The topology is fixed.
 * build() é não continuável, isto é, ele não simula internamente um modelo incremental.
 * @param L
 * @param seed
 */
case class OSELM(L: Int, seed: Int = 42, notes: String = "") extends ConvergentIncremental {
  override val toString = "OSELM_" + notes
  val Lbuild = L
}