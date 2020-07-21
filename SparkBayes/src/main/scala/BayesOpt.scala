package sample
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{BooleanType, DataType, DoubleType, IntegerType}
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.odkl.hyperopt._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator

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

    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("SparkBayes")
      .getOrCreate()

    val startTime = System.nanoTime
    var df = spark.read.format("csv")
      .option("header", "true")
      .load("../data/input/prepared_facebook_data.csv")
    val duration = (System.nanoTime - startTime) / 1e9d

    //df.show(5, false)
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

    // TODO: check default values of hyperparameters
    // XGBoost from pravda-ml:
    val model = new XGBoostRegressor()
      .setFeatureCol("features")
      .setNumEarlyStoppingRounds(2)

    val evaluator = Evaluator.crossValidate(
      model,
      new TrainTestEvaluator(new RegressionEvaluator()),
      //new RegressionEvaluator(),
      numThreads = 5,
      numFolds = 5
    )

    val optimizer = new StochasticHyperopt(evaluator)
      .setSearchMode(BayesianParamOptimizer.GAUSSIAN_PROCESS)
      .setParamDomains(
        ParamDomainPair(model.maxDepth, new IntRangeDomain(1,10)),
        ParamDomainPair(model.lambda, new DoubleRangeDomain(0.1,1.0))
      )
      .setNumThreads(10)
      .setMaxIter(2)
      //.setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'rmse' AND isTest")
    println("optimizer works")
    preparedDF.show(5,true)
    val res = optimizer.fit(preparedDF)
    println("optimizer fitted")
    println(res.getEvalMetric)



    /*
    val pipeline = new Pipeline()
      .setStages(Array(
        features_assembler,
        new AutoAssembler().setColumnsToExclude("label")
      ))
     */



  }

}
