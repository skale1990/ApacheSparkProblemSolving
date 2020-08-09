package com.som.spark.learning

import org.apache.log4j.LogManager
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.UserDefinedFunction

// SO=63064483
class RotatingKeys(spark: SparkSession, nRotations: Integer) {
  import spark.implicits._

  val logger = LogManager.getLogger(getClass)

  logger.info("Initializing KeyRotatorJob")

  def rotateKeyUdf: UserDefinedFunction = {
    udf{ (key: String, nRotations: Integer) => key.substring(nRotations) + key.substring(0, nRotations) }
  }

  def run(): Unit =
    spark
      .sql("SELECT '0123456' as key")
      .withColumn("rotated_key", rotateKeyUdf($"key", lit(nRotations)))
      .show()
}

object Main {
  val spark = SparkSession.builder()
    .appName("Run Trials")
    .config("spark.master", "local")
    .getOrCreate()

  def main(args: Array[String]): Unit = {
    val rkRun = new RotatingKeys(spark,4)
    rkRun.run()
  }
}