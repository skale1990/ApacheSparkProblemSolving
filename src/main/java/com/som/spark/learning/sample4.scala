package com.som.spark.learning
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

// SO = 63093134
object sample4 {

  val spark = SparkSession
    .builder()
    .appName("Sample app")
    .master("local")
    .getOrCreate()

  val sc = spark.sparkContext

  final case class Owner(a: Long,
                         b: String,
                         c: Long,
                         d: Short,
                         e: String,
                         f: String,
                         o_qtty: Double)

  final case class Result(a: Long,
                          b: String,
                          c: Long,
                          d: Short,
                          e: String,
                          f: String,
                          o_qtty: Double,
                          isDiff: Boolean)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)

    import spark.implicits._

    val data1 = Seq(
      Owner(11, "A", 666, 2017, "x", "y", 50),
      Owner(11, "A", 222, 2018, "x", "y", 20),
      Owner(33, "C", 444, 2018, "x", "y", 20),
      Owner(33, "C", 555, 2018, "x", "y", 120),
      Owner(22, "B", 555, 2018, "x", "y", 20),
      Owner(99, "D", 888, 2018, "x", "y", 100),
      Owner(11, "A", 888, 2018, "x", "y", 100),
      Owner(11, "A", 666, 2018, "x", "y", 80),
      Owner(33, "C", 666, 2018, "x", "y", 80),
      Owner(11, "A", 444, 2018, "x", "y", 50)
    )

    val data2 = Seq(
      Owner(11, "A", 666, 2017, "x", "y", 50),
      Owner(11, "A", 222, 2018, "x", "y", 20),
      Owner(33, "C", 444, 2018, "x", "y", 20),
      Owner(33, "C", 555, 2018, "x", "y", 55),
      Owner(22, "B", 555, 2018, "x", "y", 20),
      Owner(99, "D", 888, 2018, "x", "y", 100),
      Owner(11, "A", 888, 2018, "x", "y", 100),
      Owner(11, "A", 666, 2018, "x", "y", 80),
      Owner(33, "C", 666, 2018, "x", "y", 80),
      Owner(11, "A", 444, 2018, "x", "y", 50)
    )

    val expected = Seq(
      Result(11, "A", 666, 2017, "x", "y", 50, isDiff = false),
      Result(11, "A", 222, 2018, "x", "y", 20, isDiff = false),
      Result(33, "C", 444, 2018, "x", "y", 20, isDiff = false),
      Result(33, "C", 555, 2018, "x", "y", 55, isDiff = true),
      Result(22, "B", 555, 2018, "x", "y", 20, isDiff = false),
      Result(99, "D", 888, 2018, "x", "y", 100, isDiff = false),
      Result(11, "A", 888, 2018, "x", "y", 100, isDiff = false),
      Result(11, "A", 666, 2018, "x", "y", 80, isDiff = false),
      Result(33, "C", 666, 2018, "x", "y", 80, isDiff = false),
      Result(11, "A", 444, 2018, "x", "y", 50, isDiff = false)
    )

    val df1 = spark
      .createDataset(data1)
      .as[Owner]
      .cache()

    val df2 = spark
      .createDataset(data2)
      .as[Owner]
      .cache()

    val df1_hash = df1.withColumn("x", lit(0))
    df2.join(df1_hash, df2.columns, "left")
      .select(when(col("x").isNull, false).otherwise(true).as("isDiff") +: df2.columns.map(df2(_)): _*)
      .show(false)

    /**
      * +------+---+---+---+----+---+---+------+
      * |isDiff|a  |b  |c  |d   |e  |f  |o_qtty|
      * +------+---+---+---+----+---+---+------+
      * |true  |11 |A  |666|2017|x  |y  |50.0  |
      * |true  |11 |A  |222|2018|x  |y  |20.0  |
      * |true  |33 |C  |444|2018|x  |y  |20.0  |
      * |false |33 |C  |555|2018|x  |y  |55.0  |
      * |true  |22 |B  |555|2018|x  |y  |20.0  |
      * |true  |99 |D  |888|2018|x  |y  |100.0 |
      * |true  |11 |A  |888|2018|x  |y  |100.0 |
      * |true  |11 |A  |666|2018|x  |y  |80.0  |
      * |true  |33 |C  |666|2018|x  |y  |80.0  |
      * |true  |11 |A  |444|2018|x  |y  |50.0  |
      * +------+---+---+---+----+---+---+------+
      */

  }

}