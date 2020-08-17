package sample

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{BooleanType, DataType, DoubleType, IntegerType}
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.odkl.hyperopt._
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import java.io._

object BayesOpt {
  def castAllTypedColumnsTo(df: DataFrame,
                            columnsToBeCasted: Array[String],
                            targetType: DataType) : DataFrame = {
    /*
    * Change columns datatype by name
    */
    columnsToBeCasted.foldLeft(df) { (foldedDf, col) =>
      foldedDf.withColumn(col, df(col).cast(targetType))
    }
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().
      setMaster("local").
      setAppName("LearnScalaSpark")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    //println(sc.defaultParallelism)

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
    //preparedDF.printSchema()

    val featuresArray = colsNames.filter(! _.contains("label"))
    val features_assembler = new VectorAssembler()
      .setInputCols(featuresArray)
      .setOutputCol("features")
    preparedDF = features_assembler.transform(preparedDF)
    preparedDF = preparedDF.select("features", "label")

    // Hyperparameters optimization
    val model = new XGBoostRegressor()
      .setFeatureCol("features")
      .setEta(0.1)
      .setMaxDepth(3)
    //.setNumRounds(3)
    //.setNumWorkers(1)

    //println(model.extractParamMap())
    //println(model.explainParams())

    val evaluator = Evaluator.crossValidate(
      model,
      new TrainTestEvaluator(new RegressionEvaluator()),
      //numThreads = 5,
      numFolds = 5
    )


    /*param_dist = {"max_depth": sp_randint(1, 20),
      "min_child_weight": sp_randint(1, 20),
      "alpha": uniform(loc=0, scale=1),
      "lambda": uniform(loc=0, scale=1),
      "subsample":uniform(loc=0.5, scale=0.4),
      "colsample_bytree":uniform(loc=0.5, scale=0.4)}
     */

    val estimator = new StochasticHyperopt(evaluator)
      .setSearchMode(BayesianParamOptimizer.GAUSSIAN_PROCESS)
      .setParamDomains(
        ParamDomainPair(model.maxDepth, IntRangeDomain(1,10)),
        ParamDomainPair(model.minChildWeight, DoubleRangeDomain(1.0,10.0)),
        ParamDomainPair(model.alpha, DoubleRangeDomain(0.0,1.0)),
        ParamDomainPair(model.lambda, DoubleRangeDomain(0.0,1.0)),
        ParamDomainPair(model.subsample, DoubleRangeDomain(0.5,0.9)),
        ParamDomainPair(model.colsampleBytree, DoubleRangeDomain(0.5,0.9))
      )
      .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'r2' AND isTest")
      //.setNumThreads(15)
      .setMaxIter(728)//728
      .setNanReplacement(-999)
      .setEpsilonGreedy(0.1)

    val startTime = System.nanoTime
    val opt_result = estimator.fit(preparedDF)
    val duration = (System.nanoTime - startTime) / 1e9d

    println(opt_result.extractParamMap())
    println(opt_result.toString())

    println("configs")
    val configs = opt_result.summary(Block("configurations"))
    configs.show(5, true)
    configs.write.format("csv")
      .option("header", "true")
      .save("data/output/BO_configs_1.csv")


    println("metrics") // значения разных метрик по фолдам
    val metrics = opt_result.summary(Block("metrics"))
    metrics.show(5, true)
    metrics.write.format("csv")
      .option("header", "true")
      .save("data/output/BO_metrics_1.csv")

    println("Totaltime: "+duration.toString())
    println(estimator.extractConfig(opt_result).toString())
    val file = new File("data/output/BO_info_1.csv")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write("Totaltime: "+duration.toString()+"\n"+estimator.extractConfig(opt_result).toString())
    bw.close()
  }
}
