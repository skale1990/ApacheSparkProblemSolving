package com.som.spark.learning

import java.lang.reflect.Method
import java.text.SimpleDateFormat
import java.time.LocalDateTime
import java.time.format.{DateTimeFormatter, DateTimeFormatterBuilder}
import java.time.temporal.ChronoField
import java.util.concurrent.TimeUnit
import java.util.{Collections, Locale}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkFiles
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType
import org.testng.annotations.{BeforeClass, BeforeMethod, Test}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalog.{Database, Table}
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.{GenericRowWithSchema, MonthsBetween, Rand}
import org.apache.spark.sql.catalyst.expressions.aggregate.{ApproximatePercentile, Percentile}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{first, _}
import org.apache.spark.sql.types._
import org.joda.time.{DateTimeConstants, Months}
import org.json4s.JsonAST

import scala.collection.{JavaConverters, mutable}
import scala.util.Random

class ProblemSolverJul2020Test extends Serializable {

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
  def test62635600(): Unit = {
    val data =
      """
        |cats | letter| pet
        |cat1 | A     | cat
        |cat1 | A     | dog
        |cat2 | B     | cat
        |cat2 | B     | cat
        |cat2 | A     | cat
        |cat1 | B     | cat
        |cat1 | B     | dog
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

    df1.groupBy("cats").agg(countDistinct("letter", "pet"))
      .show(false)

    /**
      * +----+---------------------------+
      * |cats|count(DISTINCT letter, pet)|
      * +----+---------------------------+
      * |cat1|4                          |
      * |cat2|2                          |
      * +----+---------------------------+
      */
  }

  // ############################################################################################################
  @Test
  def test62635808(): Unit = {
    val sDF = Seq("""001\r\nLuc  Krier\r\n2363  Ryan Road, Long Lake South Dakota""",
      """002\r\nJeanny  Thorn\r\n2263 Patton Lane Raleigh North Carolina""",
      """003\r\nTeddy E Beecher\r\n2839 Hartland Avenue Fond Du Lac Wisconsin""",
      """004\r\nPhilippe  Schauss\r\n1 Im Oberdorf Allemagne""",
      """005\r\nMeindert I Tholen\r\nHagedoornweg 138 Amsterdam""").toDF("""s""")

    val processedDF = sDF.withColumn("col1", slice(split(col("s"), """\\r\\n"""), -2, 2))
    processedDF.show(false)
    processedDF.printSchema()

    /**
      * +--------------------------------------------------------------------+-------------------------------------------------------------+
      * |s                                                                   |col1                                                         |
      * +--------------------------------------------------------------------+-------------------------------------------------------------+
      * |001\r\nLuc  Krier\r\n2363  Ryan Road, Long Lake South Dakota        |[Luc  Krier, 2363  Ryan Road, Long Lake South Dakota]        |
      * |002\r\nJeanny  Thorn\r\n2263 Patton Lane Raleigh North Carolina     |[Jeanny  Thorn, 2263 Patton Lane Raleigh North Carolina]     |
      * |003\r\nTeddy E Beecher\r\n2839 Hartland Avenue Fond Du Lac Wisconsin|[Teddy E Beecher, 2839 Hartland Avenue Fond Du Lac Wisconsin]|
      * |004\r\nPhilippe  Schauss\r\n1 Im Oberdorf Allemagne                 |[Philippe  Schauss, 1 Im Oberdorf Allemagne]                 |
      * |005\r\nMeindert I Tholen\r\nHagedoornweg 138 Amsterdam              |[Meindert I Tholen, Hagedoornweg 138 Amsterdam]              |
      * +--------------------------------------------------------------------+-------------------------------------------------------------+
      *
      * root
      * |-- s: string (nullable = true)
      * |-- col1: array (nullable = true)
      * |    |-- element: string (containsNull = true)
      */
  }

  // ############################################################################################################
  @Test
  def test62635570(): Unit = {
    val data =
      """
        |Artist | Skill
        |Bono | Vocals
        |Bono | Vocals
        |Bono | Vocals
        |Bono | Guitar
        |Edge | Vocals
        |Edge | Guitar
        |Edge | Guitar
        |Edge | Guitar
        |Edge |     Bass
        | Larry   | Drum
        | Larry   | Drum
        | Larry   | Guitar
        | Clayton | Bass
        | Clayton | Bass
        | Clayton | Guitar
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

    df1.groupBy("Artist", "skill").count()
      .select($"Artist", struct("count", "skill").as("count_skill"))
      .groupBy("Artist").max("count_skill")
      .show(false)
  }

  // ############################################################################################################
  @Test
  def test62639267(): Unit = {
    val df = spark.sql(
      """
        |select amount, transaction_code
        |from values
        | (array(10, 20, 30, 40), array('buy', 'buy', 'sell'))
        | T(amount, transaction_code)
      """.stripMargin)

    val df1 = df.withColumn("zip", arrays_zip($"amount", $"transaction_code"))
    df1.show(false)
    df1.printSchema()
    /**
      * +----------------+----------------+-----------------------------------------+
      * |amount          |transaction_code|zip                                      |
      * +----------------+----------------+-----------------------------------------+
      * |[10, 20, 30, 40]|[buy, buy, sell]|[[10, buy], [20, buy], [30, sell], [40,]]|
      * +----------------+----------------+-----------------------------------------+
      *
      * root
      * |-- amount: array (nullable = false)
      * |    |-- element: integer (containsNull = false)
      * |-- transaction_code: array (nullable = false)
      * |    |-- element: string (containsNull = false)
      * |-- zip: array (nullable = false)
      * |    |-- element: struct (containsNull = false)
      * |    |    |-- amount: integer (nullable = true)
      * |    |    |-- transaction_code: string (nullable = true)
      */
    val df2 = df.withColumn("zip", struct($"amount", $"transaction_code"))
    df2.show(false)
    df2.printSchema()

    /**
      * +----------------+----------------+------------------------------------+
      * |amount          |transaction_code|zip                                 |
      * +----------------+----------------+------------------------------------+
      * |[10, 20, 30, 40]|[buy, buy, sell]|[[10, 20, 30, 40], [buy, buy, sell]]|
      * +----------------+----------------+------------------------------------+
      *
      * root
      * |-- amount: array (nullable = false)
      * |    |-- element: integer (containsNull = false)
      * |-- transaction_code: array (nullable = false)
      * |    |-- element: string (containsNull = false)
      * |-- zip: struct (nullable = false)
      * |    |-- amount: array (nullable = false)
      * |    |    |-- element: integer (containsNull = false)
      * |    |-- transaction_code: array (nullable = false)
      * |    |    |-- element: string (containsNull = false)
      */
  }

  // ############################################################################################################
  @Test
  def test62639511(): Unit = {
   val df = Seq("abc", "abc", "abc", "abc", "de", "de", "opqrs", "opqrs", "opqrs", "opqrs", "opqrs").toDF("id")
      .withColumn("value", row_number().over(Window.orderBy("id")))
    df.show(false)

    // If you have id as categorical column
    // expecting less distinct values
    val tuples = df.groupBy("id").count().as[(String, Long)].collect()
    val totalCount = tuples.map(_._2).sum
    val fractions: Map[String, Double] = tuples.map(t => t._1 -> 0.3).toMap
    df.stat.sampleBy("id", fractions , System.currentTimeMillis())
      .show(false)

    spark.createDataFrame(df.rdd.map(r => r.getAs[String]("id") -> r)
      .sampleByKeyExact(false, df.select("id").distinct().as[String].collect().map(t => t -> 0.3).toMap)
      .map(_._2), df.schema)
      .show(false)

    df.select("id").distinct().sample(true, 0.3)
      .join(df, Seq("id"), "left")
      .show(false)

  }
  // ############################################################################################################
  @Test
  def test62648853(): Unit = {
    //    val percentile = udf((n: Seq[Int], arr: Seq[Double]) =>
    //      new Percentile(lit(arr.toArray).expr, lit(n.map(_ * 0.01).toArray).expr))
    //
//        def percentile1(double: Column): Column = {
//          new Column(new Percentile(double.cast("bigint").expr, lit(Range(1, 100).map(_ * 0.01).toArray).expr))
//        }
    //
    //    def percentile2(arr: Column, quantile: Column): Column = {
    //      new Column(new Percentile(arr.cast("bigint").expr, quantile.expr))
    //    }
    val df = Seq(("F1", "I1", 2.4),("F2", "I1", 3.17),("F3", "I1", 4.25))
      .toDF("FacilityKey", "ItemKey", "ItemValue")
    df.show(false)
    df.printSchema()

    /**
      * +-----------+-------+---------+
      * |FacilityKey|ItemKey|ItemValue|
      * +-----------+-------+---------+
      * |F1         |I1     |2.4      |
      * |F2         |I1     |3.17     |
      * |F3         |I1     |4.25     |
      * +-----------+-------+---------+
      *
      * root
      * |-- FacilityKey: string (nullable = true)
      * |-- ItemKey: string (nullable = true)
      * |-- ItemValue: double (nullable = false)
      */

    df
      .groupBy("ItemKey")
      .agg(
        expr(s"percentile(ItemValue, array(${Range(1, 100).map(_ * 0.01).mkString(", ")}))")
          .as("percentile"))
      .withColumn("percentile", explode($"percentile"))
      .show(false)

    /**
      * +-------+------------------+
      * |ItemKey|percentile        |
      * +-------+------------------+
      * |I1     |2.4154            |
      * |I1     |2.4307999999999996|
      * |I1     |2.4461999999999997|
      * |I1     |2.4616000000000002|
      * |I1     |2.4770000000000003|
      * |I1     |2.4924            |
      * |I1     |2.5078            |
      * |I1     |2.5232            |
      * |I1     |2.5385999999999997|
      * |I1     |2.554             |
      * |I1     |2.5694            |
      * |I1     |2.5847999999999995|
      * |I1     |2.6002            |
      * |I1     |2.6156            |
      * |I1     |2.631             |
      * |I1     |2.6464            |
      * |I1     |2.6618            |
      * |I1     |2.6772            |
      * |I1     |2.6925999999999997|
      * |I1     |2.708             |
      * +-------+------------------+
      * only showing top 20 rows
      */

    df
      .groupBy("ItemKey")
      .agg(
        expr(s"percentile(ItemValue, array(${Range(1, 100).map(_ * 0.01).mkString(", ")}))")
          .as("percentile"))
      .withColumn("percentile", explode($"percentile"))
      .explain()

    /**
      * == Physical Plan ==
      * Generate explode(percentile#58), [ItemKey#8], false, [percentile#67]
      * +- ObjectHashAggregate(keys=[ItemKey#8], functions=[percentile(ItemValue#9, [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35000000000000003,0.36,0.37,0.38,0.39,0.4,0.41000000000000003,0.42,0.43,0.44,0.45,0.46,0.47000000000000003,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.5700000000000001,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.6900000000000001,0.7000000000000001,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.8200000000000001,0.8300000000000001,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.9400000000000001,0.9500000000000001,0.96,0.97,0.98,0.99], 1, 0, 0)])
      * +- Exchange hashpartitioning(ItemKey#8, 2)
      * +- ObjectHashAggregate(keys=[ItemKey#8], functions=[partial_percentile(ItemValue#9, [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35000000000000003,0.36,0.37,0.38,0.39,0.4,0.41000000000000003,0.42,0.43,0.44,0.45,0.46,0.47000000000000003,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.5700000000000001,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.6900000000000001,0.7000000000000001,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.8200000000000001,0.8300000000000001,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.9400000000000001,0.9500000000000001,0.96,0.97,0.98,0.99], 1, 0, 0)])
      * +- LocalTableScan [ItemKey#8, ItemValue#9]
      */

    df
      .groupBy("ItemKey")
      .agg(
        expr(s"approx_percentile(ItemValue, array(${Range(1, 100).map(_ * 0.01).mkString(", ")}))")
          .as("percentile"))
      .withColumn("percentile", explode($"percentile"))
      .show(false)

    /**
      * +-------+----------+
      * |ItemKey|percentile|
      * +-------+----------+
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * |I1     |2.4       |
      * +-------+----------+
      * only showing top 20 rows
      */

    df
      .groupBy("ItemKey")
      .agg(
        expr(s"approx_percentile(ItemValue, " +
          s"")
          .as("percentile"))
      .withColumn("percentile", explode($"percentile"))
      .show(false)
  }
  // ############################################################################################################
  @Test
  def test62634170(): Unit = {
    val data =
      """
        |account_id|credit_card_Number|credit_card_limit|first_name|last_name|phone_number|amount|      date|    shop|transaction_code
        |     12345|      123456789123|           10000|       abc|      xyz|  1234567890|  1000|01/06/2020|  amazon|             buy
        |     12345|      123456789123|           10000|       abc|      xyz|  1234567890|  1100|02/06/2020|    ebay|             buy
        |     12345|      123456789123|           10000|       abc|      xyz|  1234567890|   500|02/06/2020|  amazon|            sell
        |     12345|      123456789123|           10000|       abc|      xyz|  1234567890|   200|03/06/2020|flipkart|             buy
        |     12345|      123456789123|           10000|       abc|      xyz|  1234567890|  4000|04/06/2020|    ebay|             buy
        |     12345|      123456789123|           10000|       abc|      xyz|  1234567890|   900|05/06/2020|  amazon|             buy
      """.stripMargin

    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    /**
      * root
      * |-- account_id: integer (nullable = true)
      * |-- credit_card_Number: long (nullable = true)
      * |-- credit_card_limit: integer (nullable = true)
      * |-- first_name: string (nullable = true)
      * |-- last_name: string (nullable = true)
      * |-- phone_number: integer (nullable = true)
      * |-- amount: integer (nullable = true)
      * |-- date: string (nullable = true)
      * |-- shop: string (nullable = true)
      * |-- transaction_code: string (nullable = true)
      */

    // hive syntax
    // The following query selects all columns except ds and hr.
    // SELECT `(ds|hr)?+.+` FROM sales
    // Java regex syntax

    df.selectExpr(df.columns.filter(_.matches("(account_id|credit_card_limit)?+.+")): _*)
      .printSchema()

    /**
      * root
      * |-- credit_card_Number: long (nullable = true)
      * |-- first_name: string (nullable = true)
      * |-- last_name: string (nullable = true)
      * |-- phone_number: integer (nullable = true)
      * |-- amount: integer (nullable = true)
      * |-- date: string (nullable = true)
      * |-- shop: string (nullable = true)
      * |-- transaction_code: string (nullable = true)
      */
    df.select(df.colRegex("`(account_id|credit_card_limit)?+.+`"))
      .printSchema()

    /**
      * root
      * |-- credit_card_Number: long (nullable = true)
      * |-- first_name: string (nullable = true)
      * |-- last_name: string (nullable = true)
      * |-- phone_number: integer (nullable = true)
      * |-- amount: integer (nullable = true)
      * |-- date: string (nullable = true)
      * |-- shop: string (nullable = true)
      * |-- transaction_code: string (nullable = true)
      */

  }
  // ############################################################################################################
  @Test
  def test62651743(): Unit = {
    val df = Seq("azerty").toDF("runner")
      .withColumn("positions", expr("array(10, 8, 11, null, 1, 5, 4, 1, 9, 7, 1)"))

    df.withColumn("x",
      expr("TRANSFORM(split(replace(array_join(positions, '#', ''), '#1#' , '#1$'), '[$]')," +
        " x -> split(x, '[#]'))"))
      .show(false)

    /**
      * +------+---------------------------------+----------------------------------------+
      * |runner|positions                        |x                                       |
      * +------+---------------------------------+----------------------------------------+
      * |azerty|[10, 8, 11,, 1, 5, 4, 1, 9, 7, 1]|[[10, 8, 11, , 1], [5, 4, 1], [9, 7, 1]]|
      * +------+---------------------------------+----------------------------------------+
      *
      */


  }
  // ############################################################################################################
  @Test
  def test62648809(): Unit = {
    val df = spark.sql(
      """
        |select Class_Name, Customer, Date_Time, Median_Percentage
        |from values
        |   ('ClassA', 'A', '6/13/20', 64550),
        |   ('ClassA', 'B', '6/6/20', 40200),
        |   ('ClassB', 'F', '6/20/20', 26800),
        |   ('ClassB', 'G', '6/20/20', 18100)
        |  T(Class_Name, Customer, Date_Time, Median_Percentage)
      """.stripMargin)
    df.show(false)
    df.printSchema()

    /**
      * +----------+--------+---------+-----------------+
      * |Class_Name|Customer|Date_Time|Median_Percentage|
      * +----------+--------+---------+-----------------+
      * |ClassA    |A       |6/13/20  |64550            |
      * |ClassA    |B       |6/6/20   |40200            |
      * |ClassB    |F       |6/20/20  |26800            |
      * |ClassB    |G       |6/20/20  |18100            |
      * +----------+--------+---------+-----------------+
      *
      * root
      * |-- Class_Name: string (nullable = false)
      * |-- Customer: string (nullable = false)
      * |-- Date_Time: string (nullable = false)
      * |-- Median_Percentage: integer (nullable = false)
      */
    df.groupBy("Class_Name")
      .agg(max(struct($"Median_Percentage", $"Date_Time", $"Customer")).as("struct"))
      .selectExpr("Class_Name", "struct.Customer", "struct.Date_Time", "struct.Median_Percentage")
      .show(false)

    /**
      * +----------+--------+---------+-----------------+
      * |Class_Name|Customer|Date_Time|Median_Percentage|
      * +----------+--------+---------+-----------------+
      * |ClassA    |A       |6/13/20  |64550            |
      * |ClassB    |F       |6/20/20  |26800            |
      * +----------+--------+---------+-----------------+
      */
  }
  // ############################################################################################################
  @Test
  def test62655107(): Unit = {
    val data =
      """
        |{
        |    "array0": [
        |        {
        |            "a": "1",
        |            "b": "2"
        |        },
        |        {
        |            "a": "3",
        |            "b": "4"
        |        }
        |    ]
        |}
      """.stripMargin
    val df = spark.read
      .option("multiLine", true)
      .json(Seq(data).toDS())
    df.show(false)
    df.printSchema()
    /**
      * +----------------+
      * |array0          |
      * +----------------+
      * |[[1, 2], [3, 4]]|
      * +----------------+
      *
      * root
      * |-- array0: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- a: string (nullable = true)
      * |    |    |-- b: string (nullable = true)
      */

      //alternative-1 - Schema DSL
    val newDF = df.select(col("array0").cast("array<struct<A:string,B:string>>"))
    newDF.printSchema()
    /**
      * root
      * |-- array0: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- A: string (nullable = true)
      * |    |    |-- B: string (nullable = true)
      */
      // alternative-2 Transform
    val processedDF = df.withColumn("array0", expr("TRANSFORM(array0, " +
      "x -> named_struct('A', x.a, 'B', x.b))"))
    processedDF.show(false)
    processedDF.printSchema()

    /**
      * +----------------+
      * |array0          |
      * +----------------+
      * |[[1, 2], [3, 4]]|
      * +----------------+
      *
      * root
      * |-- array0: array (nullable = true)
      * |    |-- element: struct (containsNull = false)
      * |    |    |-- A: string (nullable = true)
      * |    |    |-- B: string (nullable = true)
      */

    // change type
    val newDF1 = df.select(col("array0").cast("array<struct<A:int,B:int>>"))
    newDF1.printSchema()

    /**
      * root
      * |-- array0: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- A: integer (nullable = true)
      * |    |    |-- B: integer (nullable = true)
      */
  }
  // ############################################################################################################
  @Test
  def test62646802(): Unit = {
    val documentDF = spark.createDataFrame(Seq(
      "U U U I".split(" "),
      "U I V OTHER".split(" "),
      "V OTHER I".split(" ")
    ).map(Tuple1.apply)).toDF("sentence")

    documentDF.show(false)
    documentDF.printSchema()
    /**
      * +----------------+
      * |sentence        |
      * +----------------+
      * |[U, U, U, I]    |
      * |[U, I, V, OTHER]|
      * |[V, OTHER, I]   |
      * +----------------+
      *
      * root
      * |-- sentence: array (nullable = true)
      * |    |-- element: string (containsNull = true)
      */
    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("sentence")
      .setOutputCol("model")
      .setVectorSize(2)
      .setSeed(42)
      .setWindowSize(5)
      .setMinCount(1)
      .setMaxSentenceLength(1000000)
      .setNumPartitions(40)
    val model = word2Vec.fit(documentDF)
    model.getVectors.show(false)

    /**
      * +-----+--------------------------------------------+
      * |word |vector                                      |
      * +-----+--------------------------------------------+
      * |U    |[0.08220049738883972,-0.14330919086933136]  |
      * |I    |[-0.03325378894805908,-0.061152905225753784]|
      * |OTHER|[0.07802027463912964,0.14332491159439087]   |
      * |V    |[-0.016965162009000778,-0.12242981046438217]|
      * +-----+--------------------------------------------+
      */

  }
  // ############################################################################################################
  @Test
  def test62659241(): Unit = {
    val data =
      """
        | first_name|        card
        |  abc|999999999999
        |  lmn|222222222222
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    df.withColumn("masked_cc", expr("concat(translate(left(card, length(card)-4), '0123456789', '**********')," +
      "right(card, 4))"))
      .show(false)

    /**
      * +----------+------------+------------+
      * |first_name|card        |masked_cc   |
      * +----------+------------+------------+
      * |abc       |999999999999|********9999|
      * |lmn       |222222222222|********2222|
      * +----------+------------+------------+
      */
  }

  // ############################################################################################################
  @Test
  def test62659947(): Unit = {
    val timeDF = spark.sql(
      """
        |select current_timestamp() as time1,
        | translate(date_format(current_timestamp(), 'yyyy-MM-dd HH:mm:ssZ') ,' ', 'T') as time2,
        | translate(date_format(current_timestamp(), 'yyyy-MM-dd#HH:mm:ss$') ,'#$', 'Tz') as time3
      """.stripMargin)
    timeDF.show(false)
    timeDF.printSchema()

    /**
      * +-----------------------+------------------------+--------------------+
      * |time1                  |time2                   |time3               |
      * +-----------------------+------------------------+--------------------+
      * |2020-06-30 21:22:04.541|2020-06-30T21:22:04+0530|2020-06-30T21:22:04z|
      * +-----------------------+------------------------+--------------------+
      *
      * root
      * |-- time1: timestamp (nullable = false)
      * |-- time2: string (nullable = false)
      * |-- time3: string (nullable = false)
      */

  }
  // ############################################################################################################

  @Test
  def test62675999(): Unit = {

    val Length = 2
    val Maxsplit = 3
    val df = Seq("Rahul", "Ravi", "Raghu", "Romeo").toDF("Names")
    df.show(false)
    /**
      * +-----+
      * |Names|
      * +-----+
      * |Rahul|
      * |Ravi |
      * |Raghu|
      * |Romeo|
      * +-----+
      */

    val schema = StructType(Range(1, Maxsplit + 1).map(f => StructField(s"Col_$f", StringType)))
    val split = udf((str:String, length: Int, maxSplit: Int) =>{
      val splits = str.toCharArray.grouped(length).map(_.mkString).toArray
      RowFactory.create(splits ++ Array.fill(maxSplit-splits.length)(null): _*)
    }, schema)

    val p = df
     .withColumn("x", split($"Names", lit(Length), lit(Maxsplit)))
     .selectExpr("x.*")

    p.show(false)
    p.printSchema()

    /**
      * +-----+-----+-----+
      * |Col_1|Col_2|Col_3|
      * +-----+-----+-----+
      * |Ra   |hu   |l    |
      * |Ra   |vi   |null |
      * |Ra   |gh   |u    |
      * |Ro   |me   |o    |
      * +-----+-----+-----+
      *
      * root
      * |-- Col_1: string (nullable = true)
      * |-- Col_2: string (nullable = true)
      * |-- Col_3: string (nullable = true)
      */


    val x = df.map(r => {
      val splits = r.getString(0).toCharArray.grouped(Length).map(_.mkString).toArray
      splits ++ Array.fill(Maxsplit-splits.length)(null)
    })
    x.show(false)
    x.printSchema()

    /**
      * +-----------+
      * |value      |
      * +-----------+
      * |[Ra, hu, l]|
      * |[Ra, vi,]  |
      * |[Ra, gh, u]|
      * |[Ro, me, o]|
      * +-----------+
      *
      * root
      * |-- value: array (nullable = true)
      * |    |-- element: string (containsNull = true)
      */
  }
  // ############################################################################################################

  @Test
  def test62680587(): Unit = {
    val df = Seq("01/06/w2020",
    "02/06/2!020",
    "02/06/2020",
    "03/06/2020",
    "04/06/2020",
    "05/06/2020",
    "02/06/2020",
    "//01/0/4/202/0").toDF("date")
    df.withColumn("cleaned_map", regexp_replace($"date", "[^0-9T]", ""))
      .withColumn("date_type", to_date($"cleaned_map", "ddMMyyyy"))
      .show(false)

    /**
      * +--------------+-----------+----------+
      * |date          |cleaned_map|date_type |
      * +--------------+-----------+----------+
      * |01/06/w2020   |01062020   |2020-06-01|
      * |02/06/2!020   |02062020   |2020-06-02|
      * |02/06/2020    |02062020   |2020-06-02|
      * |03/06/2020    |03062020   |2020-06-03|
      * |04/06/2020    |04062020   |2020-06-04|
      * |05/06/2020    |05062020   |2020-06-05|
      * |02/06/2020    |02062020   |2020-06-02|
      * |//01/0/4/202/0|01042020   |2020-04-01|
      * +--------------+-----------+----------+
      */
  }
  // ############################################################################################################
  @Test
  def test62688702(): Unit = {
    val data =
      """
        |install_id|influencer_date_time|id1|id2
        |  68483732| 2020-05-28 22:56:43|21 | 543
        |  68483732| 2020-05-28 23:21:53|35 | 231
        |  68483732| 2020-05-29 00:03:21|23 | 23
        |  68483732| 2020-05-29 00:05:21|54 | 654
        |  68483732| 2020-05-29 00:06:21|12 | 12
        |  68483732| 2020-05-29 00:07:21|54 | 654
        |  68486103| 2020-06-01 00:37:38|23 | 234
        |  68486103| 2020-06-01 00:59:30|12 | 14
        |  68486103| 2020-06-01 01:59:30|54 | 54
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df.show(false)
    df.printSchema()
    /**
      * +----------+--------------------+---+---+
      * |install_id|influencer_date_time|id1|id2|
      * +----------+--------------------+---+---+
      * |68483732  |2020-05-28 22:56:43 |21 |543|
      * |68483732  |2020-05-28 23:21:53 |35 |231|
      * |68483732  |2020-05-29 00:03:21 |23 |23 |
      * |68483732  |2020-05-29 00:05:21 |54 |654|
      * |68483732  |2020-05-29 00:06:21 |12 |12 |
      * |68483732  |2020-05-29 00:07:21 |54 |654|
      * |68486103  |2020-06-01 00:37:38 |23 |234|
      * |68486103  |2020-06-01 00:59:30 |12 |14 |
      * |68486103  |2020-06-01 01:59:30 |54 |54 |
      * +----------+--------------------+---+---+
      *
      * root
      * |-- install_id: integer (nullable = true)
      * |-- influencer_date_time: timestamp (nullable = true)
      * |-- id1: integer (nullable = true)
      * |-- id2: integer (nullable = true)
      */

      // drop rows after first matching id1 and id2 in a group
    val w = Window.partitionBy("install_id").orderBy("influencer_date_time")
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    df.withColumn("new_col", min(when($"id1" === $"id2", $"influencer_date_time")).over(w))
      .filter($"influencer_date_time".cast("long") - $"new_col".cast("long")<=0)
      .show(false)

    /**
      * +----------+--------------------+---+---+-------------------+
      * |install_id|influencer_date_time|id1|id2|new_col            |
      * +----------+--------------------+---+---+-------------------+
      * |68483732  |2020-05-28 22:56:43 |21 |543|2020-05-29 00:03:21|
      * |68483732  |2020-05-28 23:21:53 |35 |231|2020-05-29 00:03:21|
      * |68483732  |2020-05-29 00:03:21 |23 |23 |2020-05-29 00:03:21|
      * |68486103  |2020-06-01 00:37:38 |23 |234|2020-06-01 01:59:30|
      * |68486103  |2020-06-01 00:59:30 |12 |14 |2020-06-01 01:59:30|
      * |68486103  |2020-06-01 01:59:30 |54 |54 |2020-06-01 01:59:30|
      * +----------+--------------------+---+---+-------------------+
      */

    // drop rows after last matching id1 and id2 in a group
    df.withColumn("new_col", max(when($"id1" === $"id2", $"influencer_date_time")).over(w))
      .filter($"influencer_date_time".cast("long") - $"new_col".cast("long")<=0)
      .show(false)

    /**
      * +----------+--------------------+---+---+-------------------+
      * |install_id|influencer_date_time|id1|id2|new_col            |
      * +----------+--------------------+---+---+-------------------+
      * |68483732  |2020-05-28 22:56:43 |21 |543|2020-05-29 00:06:21|
      * |68483732  |2020-05-28 23:21:53 |35 |231|2020-05-29 00:06:21|
      * |68483732  |2020-05-29 00:03:21 |23 |23 |2020-05-29 00:06:21|
      * |68483732  |2020-05-29 00:05:21 |54 |654|2020-05-29 00:06:21|
      * |68483732  |2020-05-29 00:06:21 |12 |12 |2020-05-29 00:06:21|
      * |68486103  |2020-06-01 00:37:38 |23 |234|2020-06-01 01:59:30|
      * |68486103  |2020-06-01 00:59:30 |12 |14 |2020-06-01 01:59:30|
      * |68486103  |2020-06-01 01:59:30 |54 |54 |2020-06-01 01:59:30|
      * +----------+--------------------+---+---+-------------------+
      */
  }

  // ############################################################################################################
  @Test
  def test62688296(): Unit = {
    val data1 =
      """
        |key  |dc_count|dc_day_count
        | 123 |13      |66
        | 123 |13      |12
      """.stripMargin
    val stringDS1 = data1.split(System.lineSeparator())
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
      * +---+--------+------------+
      * |key|dc_count|dc_day_count|
      * +---+--------+------------+
      * |123|13      |66          |
      * |123|13      |12          |
      * +---+--------+------------+
      *
      * root
      * |-- key: integer (nullable = true)
      * |-- dc_count: integer (nullable = true)
      * |-- dc_day_count: integer (nullable = true)
      */
    val data2 =
      """
        |key  |rule_dc_count|rule_day_count   |rule_out
        | 123 |2            |30               |139
        | 123 |null         |null             |64
        | 124 |2            |30               |139
        | 124 |null         |null             |64
      """.stripMargin
    val stringDS2 = data2.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    df2.show(false)
    df2.printSchema()

    /**
      * +---+-------------+--------------+--------+
      * |key|rule_dc_count|rule_day_count|rule_out|
      * +---+-------------+--------------+--------+
      * |123|2            |30            |139     |
      * |123|null         |null          |64      |
      * |124|2            |30            |139     |
      * |124|null         |null          |64      |
      * +---+-------------+--------------+--------+
      *
      * root
      * |-- key: integer (nullable = true)
      * |-- rule_dc_count: integer (nullable = true)
      * |-- rule_day_count: integer (nullable = true)
      * |-- rule_out: integer (nullable = true)
      */

    df1.createOrReplaceTempView("table1")
    df2.createOrReplaceTempView("table2")

    spark.sql(
      """
        |SELECT
        | t1.key, t2.rule_out
        |FROM table1 t1 join table2 t2 on t1.key=t2.key and
        |t1.dc_count > t2.rule_dc_count and t1.dc_day_count > t2.rule_day_count
      """.stripMargin)
      .show(false)

    /**
      * +---+--------+
      * |key|rule_out|
      * +---+--------+
      * |123|139     |
      * +---+--------+
      */
  }
  // ############################################################################################################
  @Test
  def test62688559(): Unit = {
    def modifyword = (file_path:String) => {file_path+"_"}
    val spark = SparkSession.builder().appName("spp").master("local").getOrCreate()
    spark.udf.register("customudf",modifyword)
    val someData = Seq(
      Row(8, "bat"),
      Row(9, "bat"),
      Row(64, "mouse"),
      Row(9, "mouse"),
      Row(-27, "horse"),
      Row(9, "horse")
    )
    val someSchema = List(
      StructField("number", IntegerType, true),
      StructField("word", StringType, true)
    )

    val someDF = spark.createDataFrame(
      spark.sparkContext.parallelize(someData),
      StructType(someSchema)
    )
    val new_df = someDF.withColumn("new_column",callUDF("customudf",cols = col("word")))
    new_df.show()
    val grouped_df = new_df.groupBy("word").agg(collect_list(struct(col("new_column"),col("number")))).toDF("word","combined")
    grouped_df.show()

    //
    //    +------+-----+----------+
    //    |number| word|new_column|
    //    +------+-----+----------+
    //    |     8|  bat|      bat_|
    //    |     9|  bat|      bat_|
    //    |    64|mouse|    mouse_|
    //    |     9|mouse|    mouse_|
    //    |   -27|horse|    horse_|
    //    |     9|horse|    horse_|
    //    +------+-----+----------+
    //
    //    +-----+--------------------+
    //    | word|            combined|
    //    +-----+--------------------+
    //    |  bat|[[bat_, 8], [bat_...|
    //    |horse|[[horse_, -27], [...|
    //    |mouse|[[mouse_, 64], [m...|
    //    +-----+--------------------+
    //
  }
  // ############################################################################################################
  @Test
  def test62686930(): Unit = {
    val df = spark.sql("select array('E14000530', 'E1400') as wc_code, array(28, 6, 17, 15) as sign_count")
    df.selectExpr("inline_outer(arrays_zip(wc_code, sign_count)) as (wc_code, sign_count)").show()

    /**
      * +---------+----------+
      * |  wc_code|sign_count|
      * +---------+----------+
      * |E14000530|        28|
      * |    E1400|         6|
      * |     null|        17|
      * |     null|        15|
      * +---------+----------+
      */
  }

  // ############################################################################################################
  @Test
  def test62689666(): Unit = {
    val df1 = Seq((1, "2020-05-12 10:23:45", 5000), (2, "2020-11-11 12:12:12", 2000)).toDF("id", "DateTime", "miliseconds")
    df1.withColumn("week", date_trunc("week", $"DateTime"))
      .show(false)

    /**
      * +---+-------------------+-----------+-------------------+
      * |id |DateTime           |miliseconds|week               |
      * +---+-------------------+-----------+-------------------+
      * |1  |2020-05-12 10:23:45|5000       |2020-05-11 00:00:00|
      * |2  |2020-11-11 12:12:12|2000       |2020-11-09 00:00:00|
      * +---+-------------------+-----------+-------------------+
      */

      // convert dateTime -> date truncated to the first day of week
    val findFirstDayOfWeek = udf((x:String) => {

      val dateFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
      val time = LocalDateTime.parse(x,dateFormat)
      val dayOfWeek = time.getDayOfWeek

      if (dayOfWeek.getValue != DateTimeConstants.MONDAY ) {
        val newDateTime = time.plusDays(DateTimeConstants.MONDAY - dayOfWeek.getValue())
        java.sql.Date.valueOf(newDateTime.toLocalDate)
      } else {
        java.sql.Date.valueOf(time.toLocalDate)
      }
    })

    val udf_new_df1 = df1.withColumn("week",findFirstDayOfWeek(col("DateTime")))
    udf_new_df1.show(false)
    udf_new_df1.printSchema()

    /**
      * +---+-------------------+-----------+----------+
      * |id |DateTime           |miliseconds|week      |
      * +---+-------------------+-----------+----------+
      * |1  |2020-05-12 10:23:45|5000       |2020-05-11|
      * |2  |2020-11-11 12:12:12|2000       |2020-11-09|
      * +---+-------------------+-----------+----------+
      *
      * root
      * |-- id: integer (nullable = false)
      * |-- DateTime: string (nullable = true)
      * |-- miliseconds: integer (nullable = false)
      * |-- week: date (nullable = true)
      */
  }
  // ############################################################################################################
  @Test
  def test62695886(): Unit = {
    val data =
      """
        |{
        |    "participants": [{
        |            "flaggedR": "null",
        |            "participantId": "80d-1961-4e85",
        |            "participantName": "XYZ",
        |            "purpose": "external"
        |        },
        |        {
        |            "flaggedR": "null",
        |            "participantId": "909ba80d-1961",
        |            "participantName": "ABC",
        |            "purpose": "external"
        |        }
        |    ]
        |}
      """.stripMargin
    val df = spark.read
      .option("multiLine", true)
      .json(Seq(data).toDS())
    df.show(false)
    df.printSchema()
//
//    +----------------------------------------------------------------------------+
//    |participants                                                                |
//    +----------------------------------------------------------------------------+
//    |[[null, 80d-1961-4e85, XYZ, external], [null, 909ba80d-1961, ABC, external]]|
//    +----------------------------------------------------------------------------+
//
//    root
//    |-- participants: array (nullable = true)
//    |    |-- element: struct (containsNull = true)
//    |    |    |-- flaggedR: string (nullable = true)
//    |    |    |-- participantId: string (nullable = true)
//    |    |    |-- participantName: string (nullable = true)
//    |    |    |-- purpose: string (nullable = true)
//

    val p = df.withColumn("participants", expr("TRANSFORM(participants, " +
      "x ->named_struct('flaggedR', x.flaggedR, 'participantId', x.participantId," +
      " 'participantName', x.participantName, 'purpose',  x.purpose, 'IsWrap-Up', null))"))
    p.show(false)
    p.printSchema()

//
//   +------------------------------------------------------------------------------+
//   |participants                                                                  |
//   +------------------------------------------------------------------------------+
//   |[[null, 80d-1961-4e85, XYZ, external,], [null, 909ba80d-1961, ABC, external,]]|
//   +------------------------------------------------------------------------------+
//
//   root
//   |-- participants: array (nullable = true)
//   |    |-- element: struct (containsNull = false)
//   |    |    |-- flaggedR: string (nullable = true)
//   |    |    |-- participantId: string (nullable = true)
//   |    |    |-- participantName: string (nullable = true)
//   |    |    |-- purpose: string (nullable = true)
//   |    |    |-- IsWrap-Up: null (nullable = true)
//
  }

  // ############################################################################################################
  @Test
  def test62696083(): Unit = {
    val df = spark.sql("select values from values array('U5', '-1.11115'), array('U6', '-1.11115') T(values)")
    df.show(false)
    df.printSchema()

    /**
      * +--------------+
      * |values        |
      * +--------------+
      * |[U5, -1.11115]|
      * |[U6, -1.11115]|
      * +--------------+
      *
      * root
      * |-- values: array (nullable = false)
      * |    |-- element: string (containsNull = false)
      */

    df.agg(collect_list("values").as("values"))
      .write
      .mode(SaveMode.Overwrite)
      .json("/Users/sokale/models/test62696083")

    /**
      * file written-
      * {"values":[["U5","-1.11115"],["U6","-1.11115"]]}
      */

    spark.sql(
      """
        |select
        |    percentile(if(v % 2 != 0, v, null),array(0.25,0.5,0.75,0.9)) as p_odd,
        |    percentile(if(v % 2 = 0, v, null),array(0.25,0.5,0.75,0.9)) as p_even
        |from (values (1),(2),(3),(4)) as a(v)
      """.stripMargin)
      .show(false)

    /**
      * +-----------------------------------+--------------------+
      * |p_odd                              |p_even              |
      * +-----------------------------------+--------------------+
      * |[1.5, 2.0, 2.5, 2.8000000000000003]|[2.5, 3.0, 3.5, 3.8]|
      * +-----------------------------------+--------------------+
      */
  }
  // ############################################################################################################
  @Test
  def test62697652(): Unit = {
    val fileName = "spark-test-data.json"
    val path = getClass.getResource("/" + fileName).getPath
    spark.catalog.createTable("df", path, "json")
      .show(false)

    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |1  |abc1|
      * |2  |abc2|
      * |3  |abc3|
      * +---+----+
      */

    val table = spark.catalog.getTable("df")
    val df = spark.sql(s"select * from ${table.name}")
    df.show(false)
    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |1  |abc1|
      * |2  |abc2|
      * |3  |abc3|
      * +---+----+
      */
    val aggs = df.columns.map(f => avg(length(trim(col(f)))).as(f))
    val values = df.agg(aggs.head, aggs.tail: _*).head.getValuesMap[Double](df.columns).values.toSeq
    df.schema.map(sf => (sf.name, sf.dataType)).zip(values).map{ case ((name, dt), value) => (name, dt.simpleString, value)}
      .toDF("column_name", "data_type", "avg_length")
      .show(false)

    /**
      * +-----------+---------+----------+
      * |column_name|data_type|avg_length|
      * +-----------+---------+----------+
      * |id         |bigint   |1.0       |
      * |name       |string   |4.0       |
      * +-----------+---------+----------+
      */
  }

  // ############################################################################################################
  @Test
  def test62706052(): Unit = {
    val A = """[[15,["Printing Calculators"]],[13811,["Office Products"]]]"""
    val B = """[[30888,["Paper & Printable Media"]],[223845,["Office Products"]]]"""
    val C = """[[64,["Office Calculator Accessories"]]]"""

    val df = List(A,B,C).toDF("bestseller_ranks")
    df.show(false)
    df.printSchema()
    /**
      * +------------------------------------------------------------------+
      * |bestseller_ranks                                                  |
      * +------------------------------------------------------------------+
      * |[[15,["Printing Calculators"]],[13811,["Office Products"]]]       |
      * |[[30888,["Paper & Printable Media"]],[223845,["Office Products"]]]|
      * |[[64,["Office Calculator Accessories"]]]                          |
      * +------------------------------------------------------------------+
      *
      * root
      * |-- bestseller_ranks: string (nullable = true)
      */
    val  p = df.withColumn("arr", split(
      translate(
        regexp_replace($"bestseller_ranks", """\]\s*,\s*\[""", "##"), "][", ""
      ), "##"
    ))

    val processed = p.withColumn("bestseller_ranks_transformed", expr("TRANSFORM(arr, x -> " +
      "named_struct('Ranking', cast(split(x, ',')[0] as int), 'Category', split(x, ',')[1]))"))
        .select("bestseller_ranks", "bestseller_ranks_transformed")
    processed.show(false)
    processed.printSchema()

    /**
      * +------------------------------------------------------------------+-----------------------------------------------------------------+
      * |bestseller_ranks                                                  |bestseller_ranks_transformed                                     |
      * +------------------------------------------------------------------+-----------------------------------------------------------------+
      * |[[15,["Printing Calculators"]],[13811,["Office Products"]]]       |[[15, "Printing Calculators"], [13811, "Office Products"]]       |
      * |[[30888,["Paper & Printable Media"]],[223845,["Office Products"]]]|[[30888, "Paper & Printable Media"], [223845, "Office Products"]]|
      * |[[64,["Office Calculator Accessories"]]]                          |[[64, "Office Calculator Accessories"]]                          |
      * +------------------------------------------------------------------+-----------------------------------------------------------------+
      *
      * root
      * |-- bestseller_ranks: string (nullable = true)
      * |-- bestseller_ranks_transformed: array (nullable = true)
      * |    |-- element: struct (containsNull = false)
      * |    |    |-- Ranking: integer (nullable = true)
      * |    |    |-- Category: string (nullable = true)
      */
  }
  // ############################################################################################################
  @Test
  def test62709439(): Unit = {
    val df1 = spark.range(2).withColumn("name", lit("foo"))
    df1.show(false)

    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |0  |foo |
      * |1  |foo |
      * +---+----+
      */

    df1.write
      .mode(SaveMode.Overwrite)
      .csv("/Users/sokale/models/test62709439")

    /**
      * part-0...csv
      * -------------
      * 0,foo
      *
      * part-1...csv
      * -------------
      * 1,foo
      *
      */

    spark.sql(s"select ${df1.columns.map(s => s"'$s'").mkString(", ")}")
      .write
      .mode(SaveMode.Append)
      .csv("/Users/sokale/models/test62709439")

    /**
      * part...csv
      * -----------
      * id,name
      */
  }
  // ############################################################################################################
  @Test
  def test62714273(): Unit = {
    val df1 = spark.range(4).withColumn("x", row_number().over(Window.orderBy("id")) * lit(1f))
    df1.show(false)
    /**
      * +---+---+
      * |id |x  |
      * +---+---+
      * |0  |1.0|
      * |1  |2.0|
      * |2  |3.0|
      * |3  |4.0|
      * +---+---+
      */
    val df2 = spark.range(2).withColumn("x", row_number().over(Window.orderBy("id")) * lit(1f))
    df2.show(false)
    /**
      * +---+---+
      * |id |x  |
      * +---+---+
      * |0  |1.0|
      * |1  |2.0|
      * +---+---+
      */

    val inner = df1.join(df2, Seq("x"))
      .select(
        $"x", concat(df1("id"), df2("id")).as("id")
      )
    val commonPlusUncommon =
      df1.join(df2, Seq("x"), "leftanti")
        .unionByName(
          df2.join(df1, Seq("x"), "leftanti")
        ).unionByName(inner)
    commonPlusUncommon.show(false)

    /**
      * +---+---+
      * |x  |id |
      * +---+---+
      * |3.0|2  |
      * |4.0|3  |
      * |1.0|00 |
      * |2.0|11 |
      * +---+---+
      */

    df1.join(df2, Seq("x"), "full")
      .select(
        $"x",
        concat(coalesce(df1("id"), lit("")), coalesce(df2("id"), lit(""))).as("id")
      )
      .show(false)

    /**
      * +---+---+
      * |x  |id |
      * +---+---+
      * |1.0|00 |
      * |2.0|11 |
      * |3.0|2  |
      * |4.0|3  |
      * +---+---+
      */
  }
  // ############################################################################################################
  @Test
  def test62717323(): Unit = {
    spark.sparkContext.parallelize(Range(1, 102))
      .coalesce(1)
      .mapPartitions(iter => iter.grouped(2))
      .map(seq => seq.sum)
      .foreach(println)

    /**
      * 3
      * 7
      * 11
      * 15
      * 19
      * 23
      * 27
      * 31
      * 35
      * 39
      * 43
      * 47
      * 51
      * 55
      * 59
      * ...
      * 191
      * 195
      * 199
      */
    println("###")
    spark.sparkContext.parallelize(Range(1, 102, 2))
      .map(x => x + (x+1))
      .foreach(println)

    // 62717857
    val rdd = spark.sparkContext.parallelize(Seq((0, Seq("transworld", "systems", "inc", "trying", "collect", "debt",
      "mine",
    "owed", "inaccurate"))))
    rdd.flatMap{case (i, seq) => Seq.fill(seq.length)((i, seq)).zip(seq).map(x => (x._1._1, x._1._2, x._2))}
      .foreach(println)

    /**
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),transworld)
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),systems)
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),inc)
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),trying)
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),collect)
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),debt)
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),mine)
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),owed)
      * (0,List(transworld, systems, inc, trying, collect, debt, mine, owed, inaccurate),inaccurate)
      */

  }



}

case class BestSellerRank(
                           Ranking: Integer,
                           Category: String
                         )