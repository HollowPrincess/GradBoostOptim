name := "SparkBayes"
version := "1.0"
scalaVersion := "2.11.12"
libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.4.0",
  "org.apache.spark" % "spark-sql_2.11" % "2.4.0",
  "org.apache.spark" % "spark-streaming_2.11" % "2.4.0",
  "org.apache.spark" % "spark-mllib_2.11" % "2.4.0",
  "org.jmockit" % "jmockit" % "1.34" % "test",
  "ml.dmlc" % "xgboost4j-spark" % "0.90",
  "ml.dmlc" % "xgboost4j" % "0.90"
)

libraryDependencies +=   "ru.odnoklassniki" %% "pravda-ml" % "0.6.1" withSources()