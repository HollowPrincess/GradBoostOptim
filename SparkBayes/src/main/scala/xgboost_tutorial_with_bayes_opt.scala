import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

// Tutorial from https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html

object LogisticTry {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").getOrCreate()

    // Read Dataset with Spark’s Built-In Reader
    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField("class", StringType, true)))
    val rawInput = spark.read.schema(schema).csv("data/iris.data")
    //rawInput.show(5)

    // Transform Raw Iris Dataset
    // Этот блок кодирует классы как double числа
    val stringIndexer = new StringIndexer().
      setInputCol("class").
      setOutputCol("classIndex").
      fit(rawInput)
    val labelTransformed = stringIndexer.transform(rawInput).drop("class")
    //labelTransformed.show(5)

    // Здесь все фичи преобразуются в вектора знаков
    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
      setOutputCol("features")
    val xgbInput = vectorAssembler.transform(labelTransformed).select("features", "classIndex")
    //xgbInput.show(5)

    // Dealing with missing values
    val Array(trainset, testset) = xgbInput.randomSplit(Array[Double](0.7, 0.3), 18)

    val xgbParam = Map("eta" -> 0.1f,
      "missing" -> -999,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 10,
      "num_workers" -> 1)
    val xgbClassifier = new XGBoostClassifier(xgbParam).
      setFeaturesCol("features").
      setLabelCol("classIndex")

    // Training
    xgbClassifier.setMaxDepth(2)
    val xgbClassificationModel = xgbClassifier.fit(trainset)
    val results = xgbClassificationModel.transform(testset)
  }

}
