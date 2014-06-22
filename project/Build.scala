import sbt._
import Keys._
import sbtassembly.Plugin._
import AssemblyKeys._

object Builds extends Build {
/*
  lazy val buildSettings = Defaults.defaultSettings ++ Seq(
    version := "0.1-SNAPSHOT",
    organization := "com.example",
    scalaVersion := "2.10.1"
  )

  lazy val app = Project("app", file("app"),
    settings = buildSettings ++ assemblySettings) settings(

    )
*/

  lazy val elmScala = Project("elm", file(".")) dependsOn(mlsProj)
  lazy val mlsProj = RootProject(uri("https://github.com/machine-learning-scala/mls.git"))
}
