package sample
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{BooleanType, DataType, DoubleType, IntegerType}
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor

object BayesOpt {
  def castAllTypedColumnsTo(df: DataFrame,
                            columnsToBeCasted: Array[String],
                            targetType: DataType) : DataFrame = {
    /*
    * Change columns datatype
    */
    columnsToBeCasted.foldLeft(df) { (foldedDf, col) =>
      foldedDf.withColumn(col, df(col).cast(targetType))
    }
  }

  def main(args: Array[String]): Unit = {

    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("SparkBayes")
      .getOrCreate;

    val startTime = System.nanoTime
    val df = spark.read.format("csv")
      .option("header", "true")
      .load("../data/input/prepared_facebook_data.csv")
    val duration = (System.nanoTime - startTime) / 1e9d

    //df.show(5, false)
    // get features vector:
    // convert columns for suitable datatype:
    val colsNames = df.columns
    val colsToInt = colsNames.filter(_.contains("comments_")) ++ Array("page_talking_about", "base_time", "share_num", "target")
    val colsToBool = colsNames.filter(_.contains("h_local_"))
    var colsToDouble = colsNames
    colsToDouble = List(colsToDouble, colsToBool).reduce((a, b) => a diff b)
    colsToDouble = List(colsToDouble, colsToInt).reduce((a, b) => a diff b)

    var preparedDF = castAllTypedColumnsTo(df, colsToInt, IntegerType)
    preparedDF = castAllTypedColumnsTo(preparedDF, colsToBool, BooleanType)
    preparedDF = castAllTypedColumnsTo(preparedDF, colsToDouble, DoubleType)
    //preparedDF.printSchema()

    val featuresArray = colsNames.filter(! _.contains("target"))
    val assembler = new VectorAssembler()
      .setInputCols(featuresArray)
      .setOutputCol("features")
    preparedDF = assembler.transform(preparedDF)
    preparedDF.select("features", "target").show(true)

    // run XGBoost Classifier:
    // TODO: check default values of hyperparameters
    val xgbParam = Map("eta" -> 0.3,
      "max_depth" -> 6,
      //"objective" -> "reg:squarederror",
      "num_round" -> 1,
      "num_workers" -> 1
    )

    // train the model
    val model = new XGBoostRegressor(xgbParam)
      .setFeaturesCol("features")
      .setLabelCol("target")
    val xgbClassificationModel = model.fit(preparedDF)
    val prediction = xgbClassificationModel.transform(preparedDF)

    println((preparedDF.count(), ":", prediction.count()))
  }

}
