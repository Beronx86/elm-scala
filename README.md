elm-scala
=========

ELM implementation in Scala using MTJ/netlib-java/BLAS/LAPACK
Based on [machine-learning-scala](https://github.com/machine-learning-scala/mls "mls") framework.
This is result of research.
Therefore, if you use this software in your own research,
please cite properly using the bibtex at the end or the alternatives provided by this DOI link:
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13733.svg)](http://dx.doi.org/10.5281/zenodo.13733)

Installation
------------

* Install [sbt](http://www.scala-sbt.org/release/tutorial/Installing-sbt-on-Linux.html "installing sbt")

* Clone repo:
```
    git clone https://github.com/extreme-learning-machine/elm-scala.git
```

* Run included example:
```
    cd elm-scala
    sbt run
```


Use as a library
----------------

* Add a file 'Build.scala' to the 'project' folder of your own project with the contents:
```
    import sbt._

    object MyBuild extends Build {

      lazy val myProj = Project("elm", file(".")) dependsOn(elmsProj)
      lazy val elmsProj = RootProject(uri("https://github.com/extreme-learning-machine/elm-scala.git"))

    }
```

Reference
---------
```
@article{doi/elm,
      author        = "Davi P. Santos and Andr{\'e} Carlos Ponce Leon Ferreira Carvalho",
      journal = "GitHub Software Repository",
      title         = "",
      month         = "Jan",
      year          = "2015",
      doi           = "{10.5281/zenodo.13734}",
      url           = "{http://dx.doi.org/10.5281/zenodo.13734}",
}
```

* Be happy.
