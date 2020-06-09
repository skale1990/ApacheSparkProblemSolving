package com.som.spark.learning

import java.lang.reflect.Method

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkFiles
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, Row, SaveMode, SparkSession}
import org.apache.spark.sql.types.StructType
import org.testng.annotations.{BeforeClass, BeforeMethod, Test}
import org.apache.spark.ml.linalg.{Matrices, Matrix, SparseVector, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.catalog.{Database, Table}
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.catalyst.expressions.aggregate.ApproximatePercentile
import org.apache.spark.sql.catalyst.util.{ArrayBasedMapData, GenericArrayData, MapData}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{struct, _}
import org.apache.spark.sql.types._

import scala.collection.mutable


class ProblemSolverTestJun2020 extends Serializable {

  private val spark: SparkSession = SparkSession.builder().master("local[2]")
    .appName("TestSuite")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  import spark.implicits._

  @BeforeClass
  def setupBeforeAllTests(): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
  }

  @BeforeMethod
  def nameBefore(method: Method): Unit = {
    println("\n==========================================================================")
    println("Test name: " + method.getName)
    println(s"Stack Overflow Link: https://stackoverflow.com/questions/${
      method.getName
        .replaceFirst("test", "")
    }")
    println("===========================================================================\n")
  }

  // ############################################################################################################
  @Test
  def test62274300(): Unit = {
    val data =
      """
        |Key
        |bankInfo.SBI.C_1.Kothrud.Pune.displayInfo
        |bankInfo.ICICI.C_2.TilakRoad.Pune.displayInfo
        |bankInfo.Axis.C_3.Santacruz.Mumbai.displayInfo
        |bankInfo.HDFC.C_4.Deccan.Pune.displayInfo
      """.stripMargin
    val stringDS1 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()

    /**
      * +----------------------------------------------+
      * |Key                                           |
      * +----------------------------------------------+
      * |bankInfo.SBI.C_1.Kothrud.Pune.displayInfo     |
      * |bankInfo.ICICI.C_2.TilakRoad.Pune.displayInfo |
      * |bankInfo.Axis.C_3.Santacruz.Mumbai.displayInfo|
      * |bankInfo.HDFC.C_4.Deccan.Pune.displayInfo     |
      * +----------------------------------------------+
      *
      * root
      * |-- Key: string (nullable = true)
      */

    df1.select($"key", split($"key", "\\.").as("x"))
      .withColumn("bankInfo",
        expr(
          """
            |named_struct('name', element_at(x, 2), 'cust_id', element_at(x, 3),
            | 'branch', element_at(x, 4), 'dist', element_at(x, 5))
          """.stripMargin))
      .select($"key",
        concat_ws(".", $"bankInfo.name", $"bankInfo.cust_id", $"bankInfo.branch", $"bankInfo.dist")
          .as("Local_Address"),
        concat_ws(".", $"bankInfo.name", $"bankInfo.cust_id", $"bankInfo.dist")
          .as("Address"))
      .show(false)

    /**
      * +----------------------------------------------+-------------------------+---------------+
      * |key                                           |Local_Address            |Address        |
      * +----------------------------------------------+-------------------------+---------------+
      * |bankInfo.SBI.C_1.Kothrud.Pune.displayInfo     |SBI.C_1.Kothrud.Pune     |SBI.C_1.Pune   |
      * |bankInfo.ICICI.C_2.TilakRoad.Pune.displayInfo |ICICI.C_2.TilakRoad.Pune |ICICI.C_2.Pune |
      * |bankInfo.Axis.C_3.Santacruz.Mumbai.displayInfo|Axis.C_3.Santacruz.Mumbai|Axis.C_3.Mumbai|
      * |bankInfo.HDFC.C_4.Deccan.Pune.displayInfo     |HDFC.C_4.Deccan.Pune     |HDFC.C_4.Pune  |
      * +----------------------------------------------+-------------------------+---------------+
      */
  }

  // ############################################################################################################
  @Test
  def test62277164(): Unit = {
    val data =
      """
        |Date     | Name  | Tasks
        |01-03-20 | Bob   | 3
        |01-04-20 | Bob   | 2
        |01-06-20 | Bob   | 9
        |01-07-20 | Bob   | 9
        |01-02-20 | Alice | 7
        |01-03-20 | Alice | 5
        |01-04-20 | Alice | 4
      """.stripMargin
    val stringDS1 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()
    /**
      * +--------+-----+-----+
      * |Date    |Name |Tasks|
      * +--------+-----+-----+
      * |01-03-20|Bob  |3    |
      * |01-04-20|Bob  |2    |
      * |01-06-20|Bob  |9    |
      * |01-02-20|Alice|7    |
      * |01-03-20|Alice|5    |
      * |01-04-20|Alice|4    |
      * +--------+-----+-----+
      *
      * root
      * |-- Date: string (nullable = true)
      * |-- Name: string (nullable = true)
      * |-- Tasks: integer (nullable = true)
      */

    // Given timespan
    val startDate = "01-01-20" // MM-dd-yy
    val endDate = "01-06-20" // MM-dd-yy

    val df2 =
      df1.selectExpr("Name").distinct().selectExpr("Name as distinct_name",
        s"""
           |explode(sequence(
           |   to_date('$startDate', 'MM-dd-yy'),
           |   to_date('$endDate', 'MM-dd-yy'),
           |   interval 1 day
           |   )) as Date
        """.stripMargin)

    val processedDF =  df2.join(df1,
      df2("Date") === to_date(df1("Date"), "MM-dd-yy") && df2("distinct_name") === df1("Name"),
      "full")
      .select(coalesce(df2("distinct_name"), df1("Name")).as("Name"),
        coalesce(df2("Date"), to_date(df1("Date"), "MM-dd-yy")).as("Date"),
        coalesce($"Tasks", lit(0)).as("Tasks"))

    processedDF.orderBy($"Name".desc, $"Date").show(false)
    processedDF.printSchema()

    /**
      * +-----+----------+-----+
      * |Name |Date      |Tasks|
      * +-----+----------+-----+
      * |Bob  |2020-01-01|0    |
      * |Bob  |2020-01-02|0    |
      * |Bob  |2020-01-03|3    |
      * |Bob  |2020-01-04|2    |
      * |Bob  |2020-01-05|0    |
      * |Bob  |2020-01-06|9    |
      * |Bob  |2020-01-07|9    |
      * |Alice|2020-01-01|0    |
      * |Alice|2020-01-02|7    |
      * |Alice|2020-01-03|5    |
      * |Alice|2020-01-04|4    |
      * |Alice|2020-01-05|0    |
      * |Alice|2020-01-06|0    |
      * +-----+----------+-----+
      *
      * root
      * |-- Name: string (nullable = true)
      * |-- Date: date (nullable = true)
      * |-- Tasks: integer (nullable = false)
      */
  }

}
