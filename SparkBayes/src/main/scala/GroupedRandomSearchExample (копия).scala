package sample

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{BooleanType, DoubleType, IntegerType}
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.odkl.hyperopt._
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.param.ParamMap
import java.io._
import sample.GroupedSearchXGBoost


import sample.BayesOpt.castAllTypedColumnsTo

object GroupedRandomSearch {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().
      setMaster("local").
      setAppName("LearnScalaSpark")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val spark = SparkSession.builder
      .master("local")
      .appName("SparkBayes")
      .getOrCreate()

    // Data preparation
    var df = spark.read.format("csv")
      .option("header", "true")
      .load("../data/input/prepared_facebook_data.csv")

    // get features vector:
    // convert columns for suitable datatype:
    df = df.withColumnRenamed("target","label")
    val colsNames = df.columns
    val colsToInt = colsNames.filter(_.contains("comments_")) ++ Array("page_talking_about", "base_time", "share_num")
    val colsToBool = colsNames.filter(_.contains("h_local_"))
    var colsToDouble = colsNames
    colsToDouble = List(colsToDouble, colsToBool).reduce((a, b) => a diff b)
    colsToDouble = List(colsToDouble, colsToInt).reduce((a, b) => a diff b)

    var preparedDF = castAllTypedColumnsTo(df, colsToInt, IntegerType)
    preparedDF = castAllTypedColumnsTo(preparedDF, colsToBool, BooleanType)
    preparedDF = castAllTypedColumnsTo(preparedDF, colsToDouble, DoubleType)

    val featuresArray = colsNames.filter(! _.contains("label"))
    val features_assembler = new VectorAssembler()
      .setInputCols(featuresArray)
      .setOutputCol("features")
    preparedDF = features_assembler.transform(preparedDF)
    preparedDF = preparedDF.select("features", "label")



    val optimizer = new GroupedSearchXGBoost()
    optimizer.setGroupIterNums(2)
//    optimizer.setGroupIterNums(Array(2, 1, 2))
    optimizer.runTest()(preparedDF)
//    optimizer.setSearchModes(Array("RANDOM", "RANDOM", "GAUSSIAN_PROCESS"))
//    optimizer.runOptimization()(preparedDF)

  }


}

