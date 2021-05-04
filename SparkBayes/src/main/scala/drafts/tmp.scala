import org.apache.spark.{SparkConf, SparkContext}

object tmp {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().
      setMaster("local").
      setAppName("LearnScalaSpark")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    println(sc.defaultParallelism)}
}
