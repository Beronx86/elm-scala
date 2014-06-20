import sbt._

object MyBuild extends Build {

  lazy val root = Project("root", file(".")) dependsOn(mlsProj)
  lazy val mlsProj = RootProject(uri("git://github.com/machine-learning-scala/mls.git"))

}

/*import sbt._

object MyBuild extends Build {

  lazy val root = Project("root", file("."))
                    .dependsOn(p1)
//                    .dependsOn(p2)

  lazy val p1 = RootProject(uri("https://github.com/machine-learning-scala/mls.git"))
//  lazy val p2   = RootProject(uri("git://github.com/alvinj/AppleScriptUtils.git"))

}*/
