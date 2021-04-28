package sample

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{BooleanType, DoubleType, IntegerType}
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.odkl.hyperopt.{GroupedSearch, _}
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.param.ParamMap
import java.io._

import org.apache.hadoop.io.MD5Hash
import org.apache.spark.annotation.Since
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.repro.ReproContext
import org.apache.spark.sql._

import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param._


class GroupedSearchXGBoost [ModelIn <: ModelWithSummary[ModelIn]]{
  var groupIterNums: Array[Int] = Array(1, 1, 1)
  def setGroupIterNums(values: Array[Int]): Unit = {
    groupIterNums = values
  }


  var searchModes = Array(BayesianParamOptimizer.RANDOM, BayesianParamOptimizer.RANDOM, BayesianParamOptimizer.RANDOM)
//  def setSearchModes(values: Array[String]): Unit = {
//    searchModes = values
//  }


  def runOptimization()(preparedDF: DataFrame): Unit = {

    // Hyperparameters optimization
    var model = new XGBoostRegressor()
      .setFeatureCol("features")
      .setEta(0.1)
      .setMaxDepth(3)
      .setNumRounds(1)
      .setNumWorkers(1)

    val evaluator = Evaluator.crossValidate(
      model,
      new TrainTestEvaluator(new RegressionEvaluator()),
      numThreads = 1,
      numFolds = 3
    )

    val optimizer_first_group = new StochasticHyperopt(evaluator)
      .setSearchMode(searchModes(0))
      .setParamDomains(
        ParamDomainPair(model.maxDepth, IntRangeDomain(1, 20)),
        ParamDomainPair(model.minChildWeight, DoubleRangeDomain(1.0, 20.0))
      )
      .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'r2' AND isTest")
      .setNumThreads(1)
      .setMaxIter(groupIterNums(0))//728
      .setNanReplacement(-999)
      .setEpsilonGreedy(0.1)
      .setParamNames(
        model.maxDepth -> "maxDepth",
        model.minChildWeight -> "minChildWeight"
      )
      .setMaxNoImproveIters(10)

    val optimizer_second_group = new StochasticHyperopt(evaluator)
      .setSearchMode(searchModes(1))
      .setParamDomains(
        ParamDomainPair(model.alpha, DoubleRangeDomain(0.0, 1.0)),
        ParamDomainPair(model.lambda, DoubleRangeDomain(0.0, 1.0))
      )
      .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'r2' AND isTest")
      .setNumThreads(1)
      .setMaxIter(groupIterNums(1))//728
      .setNanReplacement(-999)
      .setEpsilonGreedy(0.1)
      .setParamNames(
        model.alpha -> "alpha",
        model.lambda -> "lambda"
      )
      .setMaxNoImproveIters(10)

    val optimizer_third_group = new StochasticHyperopt(evaluator)
      .setSearchMode(searchModes(2))
      .setParamDomains(
        ParamDomainPair(model.subsample, DoubleRangeDomain(0.5, 0.9)),
        ParamDomainPair(model.colsampleBytree, DoubleRangeDomain(0.5, 0.9))
      )
      .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'r2' AND isTest")
      .setNumThreads(1)
      .setMaxIter(groupIterNums(2))//728
      .setNanReplacement(-999)
      .setEpsilonGreedy(0.1)
      .setParamNames(
        model.subsample -> "subsample",
        model.colsampleBytree -> "colsampleBytree"
      )
      .setMaxNoImproveIters(10)

    val estimator = new GroupedSearch(Seq(
      "firstGroup" -> optimizer_first_group,
      "secondGroup" -> optimizer_second_group,
      "thirdGroup" -> optimizer_third_group
    ))

    val startTime = System.nanoTime
    val opt_result = estimator.fit(preparedDF)
    val duration = (System.nanoTime - startTime) / 1e9d

    println("configs")
    val configs = opt_result.summary(Block("configurations"))
    configs.show(9, true)



//    configs.write.format("csv")
//      .option("header", "true")
//      .save("data/output/GRS_configs_1.csv")


//    println("metrics") // значения разных метрик по фолдам
//    val metrics = opt_result.summary(Block("metrics"))
//    metrics.show(9, true)
//    metrics.write.format("csv")
//      .option("header", "true")
//      .save("data/output/GRS_metrics_1.csv")
//
//    println("Total time: " + duration.toString())
//    val file = new File("data/output/GRS_info_1.csv")
//    val bw = new BufferedWriter(new FileWriter(file))
//    bw.write("Total time: " + duration.toString() + "\n") // + estimator.extractConfig(opt_result).toString())
//    bw.close()
  }
}
