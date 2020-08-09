package com.som.spark.learning

import java.io.File
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
import org.apache.spark.sql.catalyst.{InternalRow, ScalaReflection}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Cast, GenericInternalRow, GenericRow, GenericRowWithSchema, MonthsBetween, Rand, Shuffle}
import org.apache.spark.sql.catalyst.expressions.aggregate.{ApproximatePercentile, Percentile}
import org.apache.spark.sql.catalyst.plans.logical.LocalRelation
import org.apache.spark.sql.catalyst.util.DateTimeUtils
import org.apache.spark.sql.expressions.{Aggregator, Window}
import org.apache.spark.sql.functions.{first, _}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String
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

    spark.sql("SET spark.sql.parser.quotedRegexColumnNames=true")
    df.createOrReplaceTempView("table")
    spark.sql("select `(account_id|credit_card_limit)?+.+` from table")
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
    // SO=63052279
    spark.sql("with some_data (values ('A',1),('B',2) T(label, value)) select * from some_data").show()

    /**
      * +-----+-----+
      * |label|value|
      * +-----+-----+
      * |    A|    1|
      * |    B|    2|
      * +-----+-----+
      */

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
  // SO- 62734945
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

  // ############################################################################################################
  @Test
  def test62725934(): Unit = {

    val df = Seq(
      (2020, 1, 1, 1.0),
      (2020, 1, 2, 2.0),
      (2020, 1, 3, 3.0)
    ).toDF("year", "id", "t", "value")

    val df1 = df.groupBy("year", "id").agg(collect_list("value"))
    val df2 = df1.where(col("year") === 2020)
    df2.explain()

    println("##################")
    val df3 = df.groupBy("year", "id").agg(sum("value"))
    val df4 = df3.where(col("year") === 2020)
    df4.explain()
  }

  // ############################################################################################################
  @Test
  def test62741589(): Unit = {
    val data =
      """
        |comment|inp_col|inp_val
        |11     |a      |1
        |12     |a      |2
        |15     |b      |5
        |16     |b      |6
        |17     |c      |&b
        |17     |c      |7
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

    val grouped = df1.groupBy("inp_col").agg(collect_list($"inp_val").as("inp_val"))
    grouped.show(false)

//    val w = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    val df2 = df1.withColumn("new_col", when(substring($"inp_val", 1, 1) === "&",
      substring($"inp_val", 2, 1))
      .otherwise(col("inp_val")).as("new_col"))
    df2.show(false)


  }

  // ############################################################################################################
  @Test
  def test62743263(): Unit = {
    val data =
      """
        |CUSTOMER_RATE|STOREID|PUBLICATIONID
        |1.0 | MSB0045024 | AU121879
        |1.0 | MSB0045024 | AU121879
        |1.2 | MBR0000616 | AU121879
        |1.0 | MBR0000616 | AU121879
        |1.0 | MBR0010285 | AU121879
        |1.0 | MSB0045024 | AU133703
        |1.0 | MSB0023370 | AU133703
        |1.3 | MSB0023370 | AU133703
        |1.0 | MSB0045024 | AU157764
        |1.0 | MSB0023370 | AU157764
        |1.0 | MBR0000616 | AU157764
        |1.0 | TAR0000018 | AU157764
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

    val groupByCol = "PUBLICATIONID"

    df1.groupBy(groupByCol).agg(min("CUSTOMER_RATE").as("CUSTOMER_RATE"), max("CUSTOMER_RATE").as("CUSTOMER_RATE"))
  }

  // ############################################################################################################
  @Test
  def test62750796(): Unit = {
    val data =
      """
        |{
        |    "parentArray": [
        |        {
        |            "child 1": 0
        |        },
        |        {
        |            "child 1": 1
        |        }
        |    ]
        |}
      """.stripMargin
    val df = spark.read.option("multiLine", true)
      .json(Seq(data).toDS())
    df.show(false)
    df.printSchema()
    /**
      * +-----------+
      * |parentArray|
      * +-----------+
      * |[[0], [1]] |
      * +-----------+
      *
      * root
      * |-- parentArray: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- child 1: long (nullable = true)
      */

    val p = df.withColumn("parentArray", col("parentArray").cast("array<struct<new_col: long>>"))
    p.show(false)
    p.printSchema()

    /**
      * +-----------+
      * |parentArray|
      * +-----------+
      * |[[0], [1]] |
      * +-----------+
      *
      * root
      * |-- parentArray: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- new_col: long (nullable = true)
      */
  }

  // ############################################################################################################
  @Test
  def test62748980(): Unit = {
    val data =
      """
        |vehicleID   |startDateTimeUtc        |Odometer
        |a           |2019-04-11T16:27:32+0000|10000
        |a           |2019-04-11T16:27:32+0000|15000
        |a           |2019-04-11T16:43:10+0000|null
        |a           |2019-04-11T20:13:52+0000|null
        |a           |2019-04-12T14:50:35+0000|null
        |a           |2019-04-12T18:53:19+0000|20000
        |b           |2019-04-12T19:06:41+0000|350000
        |b           |2019-04-12T19:17:15+0000|370000
        |b           |2019-04-12T19:30:32+0000|null
        |b           |2019-04-12T20:19:41+0000|380000
        |b           |2019-04-12T20:42:26+0000|null
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
      * +---------+------------------------+--------+
      * |vehicleID|startDateTimeUtc        |Odometer|
      * +---------+------------------------+--------+
      * |a        |2019-04-11T16:27:32+0000|10000   |
      * |a        |2019-04-11T16:27:32+0000|15000   |
      * |a        |2019-04-11T16:43:10+0000|null    |
      * |a        |2019-04-11T20:13:52+0000|null    |
      * |a        |2019-04-12T14:50:35+0000|null    |
      * |a        |2019-04-12T18:53:19+0000|20000   |
      * |b        |2019-04-12T19:06:41+0000|350000  |
      * |b        |2019-04-12T19:17:15+0000|370000  |
      * |b        |2019-04-12T19:30:32+0000|null    |
      * |b        |2019-04-12T20:19:41+0000|380000  |
      * |b        |2019-04-12T20:42:26+0000|null    |
      * +---------+------------------------+--------+
      *
      * root
      * |-- vehicleID: string (nullable = true)
      * |-- startDateTimeUtc: string (nullable = true)
      * |-- Odometer: integer (nullable = true)
      */

    val w = Window.partitionBy("vehicleID").orderBy("startDateTimeUtc")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df1.withColumn("NewColumn-CurrentOdometer",
      max("Odometer").over(w))
      .show(false)

    /**
      * +---------+------------------------+--------+-------------------------+
      * |vehicleID|startDateTimeUtc        |Odometer|NewColumn-CurrentOdometer|
      * +---------+------------------------+--------+-------------------------+
      * |a        |2019-04-11T16:27:32+0000|10000   |10000                    |
      * |a        |2019-04-11T16:27:32+0000|15000   |15000                    |
      * |a        |2019-04-11T16:43:10+0000|null    |15000                    |
      * |a        |2019-04-11T20:13:52+0000|null    |15000                    |
      * |a        |2019-04-12T14:50:35+0000|null    |15000                    |
      * |a        |2019-04-12T18:53:19+0000|20000   |20000                    |
      * |b        |2019-04-12T19:06:41+0000|350000  |350000                   |
      * |b        |2019-04-12T19:17:15+0000|370000  |370000                   |
      * |b        |2019-04-12T19:30:32+0000|null    |370000                   |
      * |b        |2019-04-12T20:19:41+0000|380000  |380000                   |
      * |b        |2019-04-12T20:42:26+0000|null    |380000                   |
      * +---------+------------------------+--------+-------------------------+
      */
  }
  // ############################################################################################################
  @Test
  def test62761153(): Unit = {
    val df = Seq(
      (0, Vectors.dense(0.0, 10.0, 0.5), 1, Vectors.dense(0.0, 10.0, 0.5)),
      (1, Vectors.dense(0.0, 10.0, 0.2), 2, Vectors.dense(0.0, 10.0, 0.2))
    ).toDF("id", "vector1", "id2",  "vector2")
    df.show(false)
    df.printSchema()
    /**
      * +---+--------------+---+--------------+
      * |id |vector1       |id2|vector2       |
      * +---+--------------+---+--------------+
      * |0  |[0.0,10.0,0.5]|1  |[0.0,10.0,0.5]|
      * |1  |[0.0,10.0,0.2]|2  |[0.0,10.0,0.2]|
      * +---+--------------+---+--------------+
      *
      * root
      * |-- id: integer (nullable = false)
      * |-- vector1: vector (nullable = true)
      * |-- id2: integer (nullable = false)
      * |-- vector2: vector (nullable = true)
      */

    // vector1.dot(vector2)/(sqrt(vector1.dot(vector1)*sqrt(vector2.dot(vector2))
    val cosine_similarity = udf((vector1: Vector, vector2: Vector) => utils.BLAS.dot(vector1, vector2) /
        (Math.sqrt(utils.BLAS.dot(vector1, vector1))* Math.sqrt(utils.BLAS.dot(vector2, vector2)))
    )
    df.withColumn("cosine", cosine_similarity($"vector1", $"vector2"))
      .show(false)

    /**
      * +---+--------------+---+--------------+------------------+
      * |id |vector1       |id2|vector2       |cosine            |
      * +---+--------------+---+--------------+------------------+
      * |0  |[0.0,10.0,0.5]|1  |[0.0,10.0,0.5]|0.9999999999999999|
      * |1  |[0.0,10.0,0.2]|2  |[0.0,10.0,0.2]|1.0000000000000002|
      * +---+--------------+---+--------------+------------------+
      */

  }

  // ############################################################################################################
  @Test
  def test62764338(): Unit = {
    val data =
      """
        |{
        |	"template": {
        |		"id": "9",
        |		"header": [{
        |				"id": "header",
        |				"value": "Find the Right Marker for the Job"
        |			},
        |			{
        |				"id": "section1-header",
        |				"value": "Desk-Style Dry Erase Markers"
        |			},
        |			{
        |				"id": "section2-header",
        |				"value": "Pen-Style Dry Erase Markers"
        |			},
        |			{
        |				"id": "section3-header",
        |				"value": "Jumbo Washable Markers"
        |			}
        |		],
        |		"paragraph": [{
        |				"id": "description1",
        |				"value": ["Desk-style wipe off easily "]
        |			},
        |			{
        |				"id": "description2",
        |				"value": ["Pen-style "]
        |			},
        |			{
        |				"id": "description3",
        |				"value": ["banners."]
        |			},
        |			{
        |				"id": "description4",
        |				"value": ["posters"]
        |			}
        |		],
        |		"image": [{
        |				"id": "section1-image",
        |				"assetId": "S"
        |			},
        |			{
        |				"id": "section2-image",
        |				"assetId": "A"
        |			},
        |			{
        |				"id": "section3-image",
        |				"assetId": "34"
        |			},
        |			{
        |				"id": "section4-image",
        |				"assetId": "36"
        |			}
        |		]
        |	}
        |}
      """.stripMargin
    val df = spark.read.option("multiLine", true)
      .json(Seq(data).toDS())

    df.show(false)
    df.printSchema()
//
//    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//    |template                                                                                                                                                                                                                                                                                                                                                                                                                    |
//    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//    |[[[header, Find the Right Marker for the Job], [section1-header, Desk-Style Dry Erase Markers], [section2-header, Pen-Style Dry Erase Markers], [section3-header, Jumbo Washable Markers]], 9, [[S, section1-image], [A, section2-image], [34, section3-image], [36, section4-image]], [[description1, [Desk-style wipe off easily ]], [description2, [Pen-style ]], [description3, [banners.]], [description4, [posters]]]]|
//    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//
//    root
//    |-- template: struct (nullable = true)
//    |    |-- header: array (nullable = true)
//    |    |    |-- element: struct (containsNull = true)
//    |    |    |    |-- id: string (nullable = true)
//    |    |    |    |-- value: string (nullable = true)
//    |    |-- id: string (nullable = true)
//    |    |-- image: array (nullable = true)
//    |    |    |-- element: struct (containsNull = true)
//    |    |    |    |-- assetId: string (nullable = true)
//    |    |    |    |-- id: string (nullable = true)
//    |    |-- paragraph: array (nullable = true)
//    |    |    |-- element: struct (containsNull = true)
//    |    |    |    |-- id: string (nullable = true)
//    |    |    |    |-- value: array (nullable = true)
//    |    |    |    |    |-- element: string (containsNull = true)
//

    val p = df.withColumn("template", col("template")
      .cast(
        """
          |struct<
          |header:array<struct<header_id:string, header_value:string>>,
          |id:string,
          |image:array<struct<image_value:string, image_id:string>>,
          |paragraph:array<struct<paragraph_id:string, paragraph_value:array<string>>>
          |>
        """.stripMargin.replaceAll("\n", "")))
      .selectExpr("template.*")
      .withColumn("paragraph", expr("TRANSFORM(paragraph, x -> named_struct('paragraph_id', x.paragraph_id, " +
        "'paragraph_value', x.paragraph_value[0]))"))

    p.show(false)
    p.printSchema()

//
//    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---+--------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
//    |header                                                                                                                                                                                   |id |image                                                                                 |paragraph                                                                                                                   |
//    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---+--------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
//    |[[header, Find the Right Marker for the Job], [section1-header, Desk-Style Dry Erase Markers], [section2-header, Pen-Style Dry Erase Markers], [section3-header, Jumbo Washable Markers]]|9  |[[S, section1-image], [A, section2-image], [34, section3-image], [36, section4-image]]|[[description1, Desk-style wipe off easily ], [description2, Pen-style ], [description3, banners.], [description4, posters]]|
//    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---+--------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
//
//    root
//    |-- header: array (nullable = true)
//    |    |-- element: struct (containsNull = true)
//    |    |    |-- header_id: string (nullable = true)
//    |    |    |-- header_value: string (nullable = true)
//    |-- id: string (nullable = true)
//    |-- image: array (nullable = true)
//    |    |-- element: struct (containsNull = true)
//    |    |    |-- image_value: string (nullable = true)
//    |    |    |-- image_id: string (nullable = true)
//    |-- paragraph: array (nullable = true)
//    |    |-- element: struct (containsNull = false)
//    |    |    |-- paragraph_id: string (nullable = true)
//    |    |    |-- paragraph_value: string (nullable = true)
//


    val p1 = p.select($"id", posexplode_outer($"header")).selectExpr("col.*", "pos as position", "id")
    val p2 = p.select($"id", posexplode_outer($"image")).selectExpr("col.*", "pos as position")
    val p3 = p.select($"id", posexplode_outer($"paragraph")).selectExpr("col.*", "pos as position")

    p1.join(p2, "position")
        .join(p3, "position")
        .show(false)

    /**
      * +--------+---------------+---------------------------------+---+-----------+--------------+------------+---------------------------+
      * |position|header_id      |header_value                     |id |image_value|image_id      |paragraph_id|paragraph_value            |
      * +--------+---------------+---------------------------------+---+-----------+--------------+------------+---------------------------+
      * |2       |section2-header|Pen-Style Dry Erase Markers      |9  |34         |section3-image|description3|banners.                   |
      * |0       |header         |Find the Right Marker for the Job|9  |S          |section1-image|description1|Desk-style wipe off easily |
      * |1       |section1-header|Desk-Style Dry Erase Markers     |9  |A          |section2-image|description2|Pen-style                  |
      * |3       |section3-header|Jumbo Washable Markers           |9  |36         |section4-image|description4|posters                    |
      * +--------+---------------+---------------------------------+---+-----------+--------------+------------+---------------------------+
      */

  }
  // ############################################################################################################
  @Test
  def test62769856(): Unit = {
    val data =
      """
        |{"load":{"employee":{"emp_id":"123","department":"science"}}}
        |{"load":{"employee":{"emp_id":"456"}}}
      """.stripMargin
    val df = spark.read.json(data.split(System.lineSeparator()).toSeq.toDS())
    df.show(false)
    df.printSchema()

    /**
      * +----------------+
      * |load            |
      * +----------------+
      * |[[ science, 123]]|
      * |[[, 456]]       |
      * +----------------+
      *
      * root
      * |-- load: struct (nullable = true)
      * |    |-- employee: struct (nullable = true)
      * |    |    |-- department: string (nullable = true)
      * |    |    |-- emp_id: string (nullable = true)
      */
    df.write.mode(SaveMode.Overwrite).json("/Users/sokale/models/test62769856/test1")

    val df1 = spark.sql("select id, name from values (1, 'foo'), (2, null) T(id, name)")
    df1.show(false)
    df1.printSchema()
    df1.write.mode(SaveMode.Overwrite).json("/Users/sokale/models/test62769856/test2")

    df1
      .printSchema()


//      .write.mode(SaveMode.Overwrite)
//      .json("/Users/sokale/models/test62769856/test2")
  }

  // ############################################################################################################
  @Test
  def test62770920(): Unit = {
    val data =
      """
        |writingTime,time
        |  2020-06-25T13:29:33.415Z,2020-06-25T13:29:33.190Z
      """.stripMargin
    val stringDS1 = data.split(System.lineSeparator())
      .map(_.split("\\,").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
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
      * +-----------------------+----------------------+
      * |writingTime            |time                  |
      * +-----------------------+----------------------+
      * |2020-06-25 18:59:33.415|2020-06-25 18:59:33.19|
      * +-----------------------+----------------------+
      *
      * root
      * |-- writingTime: timestamp (nullable = true)
      * |-- time: timestamp (nullable = true)
      */

    val millis = udf((start: java.sql.Timestamp, end: java.sql.Timestamp) => end.getTime - start.getTime)
    df1.withColumn("date_diff_millis", millis($"time",  $"writingTime"))
      .show(false)

    /**
      * +-----------------------+----------------------+----------------+
      * |writingTime            |time                  |date_diff_millis|
      * +-----------------------+----------------------+----------------+
      * |2020-06-25 18:59:33.415|2020-06-25 18:59:33.19|225             |
      * +-----------------------+----------------------+----------------+
      */

    // casting timestamp to long converts it to seconds (unix epoch)
    df1.withColumn("date_diff_millis", $"time".cast("long") -  $"writingTime".cast("long"))
      .show(false)

    /**
      * +-----------------------+----------------------+----------------+
      * |writingTime            |time                  |date_diff_millis|
      * +-----------------------+----------------------+----------------+
      * |2020-06-25 18:59:33.415|2020-06-25 18:59:33.19|0               |
      * +-----------------------+----------------------+----------------+
      */
  }

  // ############################################################################################################
  object TestObject extends Serializable {
    private var x = 1
    def inc(): Unit = { x = x + 1 }
    def get: Int = x
  }

  @Test
  def test62771094(): Unit = {
    val df = spark.range(1, 21).toDF("num").repartition(5)

    val a = TestObject
    a.inc()

    val transformed = df.map(row => {
      val value: Long = row.getAs[Long]("num")
      value + a.get
    }).toDF("num")

    transformed.explain()
    transformed
        .orderBy("num")
      .show(20)
  }

  // ############################################################################################################
  @Test
  def test62770997(): Unit = {
    val directory = getClass.getResource("/irisData.csv").getPath
    val df = spark.sql("select cast(123.1456 as decimal(5, 2)) as price")
    df.printSchema()
    df.show(false)
    /**
      * root
      * |-- price: decimal(5,2) (nullable = true)
      *
      * +------+
      * |price |
      * +------+
      * |123.15|
      * +------+
      */
    val strings = directory.split("\\/")
    df.write
      .mode(SaveMode.Overwrite)
      .format("avro")
      .save(strings.take(strings.length-1).mkString("/") + "/avro_data/day1")

    val df1 = spark.sql("select cast(123456789012345678.1456 as decimal(20, 2)) as price")
    df1.printSchema()
    df1.show(false)

    /**
      * root
      * |-- price: decimal(20,2) (nullable = true)
      *
      * +---------------------+
      * |price                |
      * +---------------------+
      * |123456789012345678.15|
      * +---------------------+
      */
    df1.write
      .mode(SaveMode.Overwrite)
      .format("avro")
      .save(strings.take(strings.length-1).mkString("/") + "/avro_data/day2")

    spark.read.format("avro")
      .load(strings.take(strings.length-1).mkString("/") + "/avro_data/day1")
      .printSchema()

    /**
      * root
      * |-- price: decimal(5,2) (nullable = true)
      */

    spark.read.format("avro")
      .load(strings.take(strings.length-1).mkString("/") + "/avro_data/day2")
      .printSchema()

    /**
      * root
      * |-- price: decimal(20,2) (nullable = true)
      */

    val path = getClass.getResource("/avro_data").getPath
    println(path)

//    val readDF = spark.read.format("avro")
//      .load(path)
//
//    readDF.show(false)
//    readDF.printSchema()

    /**
      * org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 4.0 failed 1 times,
      *    most recent failure: Lost task 0.0 in stage 4.0 (TID 4, localhost, executor driver): java.lang
      *  .IllegalArgumentException: Unscaled value too large for precision
      * at org.apache.spark.sql.types.Decimal.set(Decimal.scala:79)
      * at org.apache.spark.sql.types.Decimal$.apply(Decimal.scala:468)
      * at org.apach
      */

    /**
      * {
      * "type": "record",
      * "name": "topLevelRecord",
      * "fields": [{
      * "name": "price",
      * "type": {
      * "type": "fixed",
      * "name": "fixed",
      * "namespace": "topLevelRecord.price",
      * "size": 3,
      * "logicalType": "decimal",
      * "precision": 5,
      * "scale": 2
      * }
      * }]
      * }
      */
    import org.apache.avro.Schema
    val schema = new Schema.Parser().parse(new File(getClass.getResource("/avsc/price.avsc").getFile))
    val p = spark
      .read
      .format("avro")
      .option("avroSchema", schema.toString)
      .load(path)
    p.show(false)
    p.printSchema()

  }
  // ############################################################################################################
  @Test
  def test62775199(): Unit = {
    val data =
      """
        |class |  male  |  female
        |1 |  2 |  1
        |2 |  0 |  2
        |3 |  2 |  0
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
      * +-----+----+------+
      * |class|male|female|
      * +-----+----+------+
      * |1    |2   |1     |
      * |2    |0   |2     |
      * |3    |2   |0     |
      * +-----+----+------+
      *
      * root
      * |-- class: integer (nullable = true)
      * |-- male: integer (nullable = true)
      * |-- female: integer (nullable = true)
      */

    val df2 = df1.select($"class",
      when($"male" >= 1, sequence(lit(1), col("male"))).otherwise(array()).as("male"),
      when($"female" >= 1, sequence(lit(1), col("female"))).otherwise(array()).as("female")
    ).withColumn("male", expr("TRANSFORM(male, x -> 'm')"))
      .withColumn("female", expr("TRANSFORM(female, x -> 'f')"))
      .withColumn("gender", explode(concat($"male", $"female")))
      .select("class", "gender")
    df2.show(false)

    /**
      * +-----+------+
      * |class|gender|
      * +-----+------+
      * |1    |m     |
      * |1    |m     |
      * |1    |f     |
      * |2    |f     |
      * |2    |f     |
      * |3    |m     |
      * |3    |m     |
      * +-----+------+
      */
    df2.groupBy("class").agg(collect_list("gender").as("gender"))
      .withColumn("male", expr("size(FILTER(gender, x -> x='m'))"))
      .withColumn("female", expr("size(FILTER(gender, x -> x='f'))"))
      .select("class", "male", "female")
      .orderBy("class")
      .show(false)

    /**
      * +-----+----+------+
      * |class|male|female|
      * +-----+----+------+
      * |1    |2   |1     |
      * |2    |0   |2     |
      * |3    |2   |0     |
      * +-----+----+------+
      */

    // SO = 62778701
    df2.withColumn("current_date",current_timestamp())
      .show(false)

    df2.withColumn("current_date", expr("reflect('java.lang.System', 'currentTimeMillis')"))
      .show(false)

    /**
      * +-----+------+-------------+
      * |class|gender|current_date |
      * +-----+------+-------------+
      * |1    |m     |1594137247247|
      * |1    |m     |1594137247247|
      * |1    |f     |1594137247247|
      * |2    |f     |1594137247272|
      * |2    |f     |1594137247272|
      * |3    |m     |1594137247272|
      * |3    |m     |1594137247272|
      * +-----+------+-------------+
      */

    df2.withColumn("current_date", expr("reflect('java.time.LocalDateTime', 'now')"))
      .show(false)

    /**
      * +-----+------+-----------------------+
      * |class|gender|current_date           |
      * +-----+------+-----------------------+
      * |1    |m     |2020-07-07T21:24:07.377|
      * |1    |m     |2020-07-07T21:24:07.378|
      * |1    |f     |2020-07-07T21:24:07.378|
      * |2    |f     |2020-07-07T21:24:07.398|
      * |2    |f     |2020-07-07T21:24:07.398|
      * |3    |m     |2020-07-07T21:24:07.398|
      * |3    |m     |2020-07-07T21:24:07.398|
      * +-----+------+-----------------------+
      */

    df2.withColumn("current_date", expr("reflect('java.time.LocalDateTime', 'now')").cast("timestamp"))
      .show(false)

    /**
      * +-----+------+-----------------------+
      * |class|gender|current_date           |
      * +-----+------+-----------------------+
      * |1    |m     |2020-07-07 21:29:50.911|
      * |1    |m     |2020-07-07 21:29:50.911|
      * |1    |f     |2020-07-07 21:29:50.911|
      * |2    |f     |2020-07-07 21:29:50.943|
      * |2    |f     |2020-07-07 21:29:50.943|
      * |3    |m     |2020-07-07 21:29:50.943|
      * |3    |m     |2020-07-07 21:29:50.944|
      * +-----+------+-----------------------+
      */
  }

  // ############################################################################################################
  @Test
  def test62789398(): Unit = {

    // Generate parquet
//    val directory = getClass.getResource("/irisData.csv").getPath
//    val df = spark.sql("select cast(123.1456 as decimal(5, 2)) as price")
//    df.printSchema()
//    df.show(false)
//    /**
//      * root
//      * |-- price: decimal(5,2) (nullable = true)
//      *
//      * +------+
//      * |price |
//      * +------+
//      * |123.15|
//      * +------+
//      */
//    val strings = directory.split("\\/")
//    df.write
//      .mode(SaveMode.Overwrite)
//      .parquet(strings.take(strings.length-1).mkString("/") + "/parquet_data")
    // work well
    val df = spark.read
      .parquet(getClass.getResource("/parquet/plain/part-00000-4ece3595-e410-4301-aefd-431cd1debf91-c000.snappy" +
        ".parquet").getPath)
    df.show(false)
    df.printSchema()

    val df1 = spark.read
        .option("compression", "gzip")
      .parquet(getClass.getResource("/parquet/gzip/part-00000-4ece3595-e410-4301-aefd-431cd1debf91-c000.snappy" +
        ".parquet.gz").getPath)
    // Caused by: java.lang.RuntimeException: file:/Users/sokale/alm_repo/stack-overflow/StackOverflowProblemSolving/target/test-classes/parquet/gzip/part-00000-4ece3595-e410-4301-aefd-431cd1debf91-c000.snappy.parquet.gz is not a Parquet file. expected magic number at tail [80, 65, 82, 49] but found [-105, 1, 0, 0]
    df1.show(false)
    df1.printSchema()
  }

  // ############################################################################################################
  @Test
  def test62792975(): Unit = {
    val data =
      """
        |User|Shop|Location| Seller|Quantity|         GroupBYClause
        |   1| ABC|    Loc1|Seller1|      10|        Shop, location
        |   1| ABC|    Loc1|Seller2|      10|        Shop, location
        |   2| ABC|    Loc1|Seller1|      10|Shop, location, Seller
        |   2| ABC|    Loc1|Seller2|      10|Shop, location,Seller
        |   3| BCD|    Loc1|Seller1|      10|       location,Seller
        |   3| BCD|    Loc1|Seller2|      10|              location
        |   3| CDE|    Loc2|Seller3|      10|              location
      """.stripMargin

    val stringDS1 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()
    /**
      * +----+----+--------+-------+--------+----------------------+
      * |User|Shop|Location|Seller |Quantity|GroupBYClause         |
      * +----+----+--------+-------+--------+----------------------+
      * |1   |ABC |Loc1    |Seller1|10      |Shop, location        |
      * |1   |ABC |Loc1    |Seller2|10      |Shop, location        |
      * |2   |ABC |Loc1    |Seller1|10      |Shop, location, Seller|
      * |2   |ABC |Loc1    |Seller2|10      |Shop, location,Seller |
      * |3   |BCD |Loc1    |Seller1|10      |location              |
      * |3   |BCD |Loc1    |Seller2|10      |location              |
      * |3   |CDE |Loc2    |Seller3|10      |location              |
      * +----+----+--------+-------+--------+----------------------+
      *
      * root
      * |-- User: integer (nullable = true)
      * |-- Shop: string (nullable = true)
      * |-- Location: string (nullable = true)
      * |-- Seller: string (nullable = true)
      * |-- Quantity: integer (nullable = true)
      * |-- GroupBYClause: string (nullable = true)
      */

    val isShopLocation = Seq("Shop", "location").map(array_contains($"arr", _)).reduce(_ && _)
    val isShopLocationSeller = Seq("Shop", "location", "Seller").map(array_contains($"arr", _)).reduce(_ && _)
    val isLocation = array_contains($"arr", "location")
    df1.withColumn("arr", split($"GroupBYClause", "\\s*,\\s*"))
      .withColumn("arr",
        when(isShopLocationSeller, expr("array(Shop, location, Seller)"))
          .when(isShopLocation, expr("array(Shop, location)"))
          .when(isLocation, expr("array(location)"))
      ).withColumn("sum_quantity",
      sum("Quantity").over(Window.partitionBy("User","arr")))
      .show(false)

    /**
      * +----+----+--------+-------+--------+----------------------+--------------------+------------+
      * |User|Shop|Location|Seller |Quantity|GroupBYClause         |arr                 |sum_quantity|
      * +----+----+--------+-------+--------+----------------------+--------------------+------------+
      * |1   |ABC |Loc1    |Seller1|10      |Shop, location        |[ABC, Loc1]         |20          |
      * |1   |ABC |Loc1    |Seller2|10      |Shop, location        |[ABC, Loc1]         |20          |
      * |2   |ABC |Loc1    |Seller2|10      |Shop, location,Seller |[ABC, Loc1, Seller2]|10          |
      * |3   |CDE |Loc2    |Seller3|10      |location              |[Loc2]              |10          |
      * |2   |ABC |Loc1    |Seller1|10      |Shop, location, Seller|[ABC, Loc1, Seller1]|10          |
      * |3   |BCD |Loc1    |Seller1|10      |location              |[Loc1]              |20          |
      * |3   |BCD |Loc1    |Seller2|10      |location              |[Loc1]              |20          |
      * +----+----+--------+-------+--------+----------------------+--------------------+------------+
      */

    val columns = Seq("Shop", "location", "Seller").flatMap(f => Seq(lit(f), col(f)))
    df1.withColumn("arr", split($"GroupBYClause", "\\s*,\\s*"))
      .withColumn("map1", map(columns: _*))
      .withColumn("arr", expr("TRANSFORM(arr, x -> map1[x])"))
      .withColumn("sum_quantity",
        sum("Quantity").over(Window.partitionBy("User","arr")))
      .show(false)

    /**
      * +----+----+--------+-------+--------+----------------------+--------------------+--------------------------------------------------+------------+
      * |User|Shop|Location|Seller |Quantity|GroupBYClause         |arr                 |map1                                              |sum_quantity|
      * +----+----+--------+-------+--------+----------------------+--------------------+--------------------------------------------------+------------+
      * |1   |ABC |Loc1    |Seller1|10      |Shop, location        |[ABC, Loc1]         |[Shop -> ABC, location -> Loc1, Seller -> Seller1]|20          |
      * |1   |ABC |Loc1    |Seller2|10      |Shop, location        |[ABC, Loc1]         |[Shop -> ABC, location -> Loc1, Seller -> Seller2]|20          |
      * |2   |ABC |Loc1    |Seller2|10      |Shop, location,Seller |[ABC, Loc1, Seller2]|[Shop -> ABC, location -> Loc1, Seller -> Seller2]|10          |
      * |3   |CDE |Loc2    |Seller3|10      |location              |[Loc2]              |[Shop -> CDE, location -> Loc2, Seller -> Seller3]|10          |
      * |2   |ABC |Loc1    |Seller1|10      |Shop, location, Seller|[ABC, Loc1, Seller1]|[Shop -> ABC, location -> Loc1, Seller -> Seller1]|10          |
      * |3   |BCD |Loc1    |Seller1|10      |location              |[Loc1]              |[Shop -> BCD, location -> Loc1, Seller -> Seller1]|20          |
      * |3   |BCD |Loc1    |Seller2|10      |location              |[Loc1]              |[Shop -> BCD, location -> Loc1, Seller -> Seller2]|20          |
      * +----+----+--------+-------+--------+----------------------+--------------------+--------------------------------------------------+------------+
      */
  }
  // ############################################################################################################
  @Test
  def test62809983(): Unit = {
    val df = spark.sql("select array(array(1,2,3,4),array(4,5,6,7),array(7,8,9,0)) as column3")
    df.show(false)
    df.printSchema()

    /**
      * +------------------------------------------+
      * |column3                                   |
      * +------------------------------------------+
      * |[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 0]]|
      * +------------------------------------------+
      *
      * root
      * |-- column3: array (nullable = false)
      * |    |-- element: array (containsNull = false)
      * |    |    |-- element: integer (containsNull = false)
      */

    df.withColumn("agg_values", expr("TRANSFORM(column3, x -> element_at(x, -2) )"))
      .show(false)

    /**
      * +------------------------------------------+----------+
      * |column3                                   |agg_values|
      * +------------------------------------------+----------+
      * |[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 0]]|[3, 6, 9] |
      * +------------------------------------------+----------+
      */
    // use array_join to get string

    df.withColumn("agg_values", expr("TRANSFORM(column3, x -> element_at(x, -2) )"))
      .withColumn("agg_values", array_join(col("agg_values"), ", "))
      .show(false)

    /**
      * +------------------------------------------+----------+
      * |column3                                   |agg_values|
      * +------------------------------------------+----------+
      * |[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 0]]|3, 6, 9   |
      * +------------------------------------------+----------+
      */
  }
  // ############################################################################################################
  @Test
  def test62812081(): Unit = {
    // SO = 62812081

    val df1 = Seq("01-Jan-2000 01.00.00.001 AM", "01-Jan-2020 02.00.00.001 AM").toDF("DOJ")
    df1.show(false)
    df1.printSchema()

    df1
      .where(to_timestamp($"DOJ", "dd-MMM-yyyy hh.mm.ss.SSS aa")
        > to_timestamp(lit("01-Jan-2005 01.00.00.001 AM"), "dd-MMM-yyyy hh.mm.ss.SSS aa"))
      .show(false)
  }
  // ############################################################################################################
  @Test
  def test62816196(): Unit = {

    val df = spark.read
      .parquet(
        getClass.getResource("/parquet/day/day1/part-00000-4ece3595-e410-4301-aefd-431cd1debf91-c000.snappy.parquet")
          .getPath,
        getClass.getResource("/parquet/day/day2/part-00000-4ece3595-e410-4301-aefd-431cd1debf91-c000.snappy.parquet")
          .getPath
      )
    df.show(false)
    df.printSchema()

    /**
      * +------+
      * |price |
      * +------+
      * |123.15|
      * |123.15|
      * +------+
      *
      * root
      * |-- price: decimal(5,2) (nullable = true)
      */

    df.withColumn("subfolder", element_at(split(input_file_name(), "/"), -2))
      .show(false)

    /**
      * +------+---------+
      * |price |subfolder|
      * +------+---------+
      * |123.15|day1     |
      * |123.15|day2     |
      * +------+---------+
      */

  }
  // ############################################################################################################

  @Test
  def test62805430(): Unit = {

    val customSummer =  new Aggregator[Data, Int, Int] {
      def zero: Int = 0
      def reduce(b: Int, a: Data): Int = b + a.i
      def merge(b1: Int, b2: Int): Int = b1 + b2
      def finish(r: Int): Int = r
      def bufferEncoder: Encoder[Int] = org.apache.spark.sql.Encoders.scalaInt
      def outputEncoder: Encoder[Int] = org.apache.spark.sql.Encoders.scalaInt
    }.toColumn

    val ds = Seq(Data(1), Data(2)).toDS
    val aggregated = ds.select(customSummer).collect
    println(aggregated.mkString(",")) // 3
  }
  // ############################################################################################################

  @Test
  def test62827036(): Unit = {
    val data =
      """
        |PersonId | Education1CollegeName | Education1Degree | Education2CollegeName | Education2Degree |Education3CollegeName | Education3Degree
        | 1 | xyz | MS | abc | Phd | pqr | BS
        |  2 | POR | MS | ABC | Phd | null | null
      """.stripMargin
    val stringDS1 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()

    /**
      * +--------+---------------------+----------------+---------------------+----------------+---------------------+----------------+
      * |PersonId|Education1CollegeName|Education1Degree|Education2CollegeName|Education2Degree|Education3CollegeName|Education3Degree|
      * +--------+---------------------+----------------+---------------------+----------------+---------------------+----------------+
      * |1       |xyz                  |MS              |abc                  |Phd             |pqr                  |BS              |
      * |2       |POR                  |MS              |ABC                  |Phd             |null                 |null            |
      * +--------+---------------------+----------------+---------------------+----------------+---------------------+----------------+
      *
      * root
      * |-- PersonId: integer (nullable = true)
      * |-- Education1CollegeName: string (nullable = true)
      * |-- Education1Degree: string (nullable = true)
      * |-- Education2CollegeName: string (nullable = true)
      * |-- Education2Degree: string (nullable = true)
      * |-- Education3CollegeName: string (nullable = true)
      * |-- Education3Degree: string (nullable = true)
      */

    df1.selectExpr("PersonId",
      "stack(3, Education1CollegeName, Education1Degree, Education2CollegeName, Education2Degree, " +
        "Education3CollegeName, Education3Degree) as (CollegeName, EducationDegree)")
      .where("CollegeName is not null and EducationDegree is not null")
      .show(false)

    /**
      * +--------+-----------+---------------+
      * |PersonId|CollegeName|EducationDegree|
      * +--------+-----------+---------------+
      * |1       |xyz        |MS             |
      * |1       |abc        |Phd            |
      * |1       |pqr        |BS             |
      * |2       |POR        |MS             |
      * |2       |ABC        |Phd            |
      * +--------+-----------+---------------+
      */

  }
  // ############################################################################################################

  @Test
  def test62826527(): Unit = {
    val data =
      """
        |Name|Subject|Marks
        |  Ram|Physics|   80
        |  Sham|English|   90
        |  Ayan|   Math|   70
      """.stripMargin
    val stringDS1 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()
    /**
      * +----+-------+-----+
      * |Name|Subject|Marks|
      * +----+-------+-----+
      * |Ram |Physics|80   |
      * |Sham|English|90   |
      * |Ayan|Math   |70   |
      * +----+-------+-----+
      *
      * root
      * |-- Name: string (nullable = true)
      * |-- Subject: string (nullable = true)
      * |-- Marks: integer (nullable = true)
      */

    val x = "Marks"
    // use functions.col
    df1.withColumn(x, when(col(x) > 75, col(x) + 10).otherwise(col(x)))
      .show()

    /**
      * +----+-------+-----+
      * |Name|Subject|Marks|
      * +----+-------+-----+
      * | Ram|Physics|   90|
      * |Sham|English|  100|
      * |Ayan|   Math|   70|
      * +----+-------+-----+
      */
  }
  // ############################################################################################################

  @Test
  def test62833675(): Unit = {
    // 62833675
    val data1=
      """
        |country|  city | date
        | France|  Paris| 2018-07-01
        |  Spain| Madrid| 2017-06-01
      """.stripMargin
    val stringDS2 = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df_large = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    df_large.show(false)
    /**
      * +-------+------+-------------------+
      * |country|city  |date               |
      * +-------+------+-------------------+
      * |France |Paris |2018-07-01 00:00:00|
      * |Spain  |Madrid|2017-06-01 00:00:00|
      * +-------+------+-------------------+
      */

    val data3=
      """
        |country|  city | filter_date
        | France|  Paris| 2018-07-01
        |  Spain| Madrid| 2017-06-01
      """.stripMargin
    val stringDS3 = data3.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df_small = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS3)
    df_small.show(false)

    /**
      * +-------+------+-------------------+
      * |country|city  |filter_date        |
      * +-------+------+-------------------+
      * |France |Paris |2018-07-01 00:00:00|
      * |Spain  |Madrid|2017-06-01 00:00:00|
      * +-------+------+-------------------+
      */

    df_large.join(broadcast(df_small), df_large("country") === df_small("country") &&
      df_large("city") === df_small("city") && df_large("date") >= df_small("filter_date"), "leftsemi")
      .show(false)

    /**
      * +-------+------+-------------------+
      * |country|city  |date               |
      * +-------+------+-------------------+
      * |France |Paris |2018-07-01 00:00:00|
      * |Spain  |Madrid|2017-06-01 00:00:00|
      * +-------+------+-------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62833371(): Unit = {
    val data =
      """
        |Itemcode
        |
        |DB9450//DB9450/AD9066
        |
        |DA0002/DE2396//DF2345
        |
        |HWC72
        |
        |GG7183/EB6693
        |
        |TA444/B9X8X4:7-2-
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    /**
      * +---------------------+
      * |Itemcode             |
      * +---------------------+
      * |DB9450//DB9450/AD9066|
      * |DA0002/DE2396//DF2345|
      * |HWC72                |
      * |GG7183/EB6693        |
      * |TA444/B9X8X4:7-2-    |
      * +---------------------+
      *
      * root
      * |-- Itemcode: string (nullable = true)
      */

    df.withColumn("item_code", expr("TRANSFORM(split(Itemcode, '/+'), x -> split(x, ':')[0])"))
      .selectExpr("item_code[0] item1", "item_code[1] item2", "item_code[2] item3")
      .show(false)

    /**
      * +------+------+------+
      * |item1 |item2 |item3 |
      * +------+------+------+
      * |DB9450|DB9450|AD9066|
      * |DA0002|DE2396|DF2345|
      * |HWC72 |null  |null  |
      * |GG7183|EB6693|null  |
      * |TA444 |B9X8X4|null  |
      * +------+------+------+
      */
  }
  // ############################################################################################################

  @Test
  def test62826229(): Unit = {
    import org.apache.spark.mllib.linalg.{Matrices => OldMatrices, Matrix => OldMatrix}

    // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    val dm: OldMatrix = OldMatrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))

    println(dm)
    /**
      * 1.0  2.0
      * 3.0  4.0
      * 5.0  6.0
      */
    // /** Gets the (i, j)-th element. */ index starts from 0
    println(dm.apply(0, 1))
    // 2.0
  }
  // ############################################################################################################

  @Test
  def test62823455(): Unit = {
    val data =
      """
        |id     year  month val
        |u_ced    2002   05    v_scyronc
        |u_ced    2002   05    v_ytbtbdv
        |u_ced    2002   05    v_utncasx
        |u_pny    2005   07    v_opcrgae
        |u_pny    2005   07    v_wytnecs
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\s+").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    /**
      * +-----+-----+------+---------+
      * |id,  |year,|month,|val      |
      * +-----+-----+------+---------+
      * |u_ced|2002 |5     |v_scyronc|
      * |u_ced|2002 |5     |v_ytbtbdv|
      * |u_ced|2002 |5     |v_utncasx|
      * |u_pny|2005 |7     |v_opcrgae|
      * |u_pny|2005 |7     |v_wytnecs|
      * +-----+-----+------+---------+
      *
      * root
      * |-- id,: string (nullable = true)
      * |-- year,: integer (nullable = true)
      * |-- month,: integer (nullable = true)
      * |-- val: string (nullable = true)
      */

    df.withColumn("shown", lit(true))
      .withColumnRenamed("val", "val_id")
      .withColumn("val_json", struct(col("shown"), col("val_id")))
      .groupBy("id")
      .agg(collect_list("val_json").as("val_json"))
      .select(col("id"), to_json(col("val_json")).as("val_json"))
      .show(false)

    /**
      * +-----+-------------------------------------------------------------------------------------------------------------+
      * |id   |val_json                                                                                                     |
      * +-----+-------------------------------------------------------------------------------------------------------------+
      * |u_pny|[{"shown":true,"val_id":"v_opcrgae"},{"shown":true,"val_id":"v_wytnecs"}]                                    |
      * |u_ced|[{"shown":true,"val_id":"v_scyronc"},{"shown":true,"val_id":"v_ytbtbdv"},{"shown":true,"val_id":"v_utncasx"}]|
      * +-----+-------------------------------------------------------------------------------------------------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62865986(): Unit = {
    val df =spark.sql("SET").withColumn("rw",expr("row_number() over(order by key)"))
    df.show(false)
    df.printSchema()

    /**
      * +----------------------------+-------------------+---+
      * |key                         |value              |rw |
      * +----------------------------+-------------------+---+
      * |spark.app.id                |local-1594644271573|1  |
      * |spark.app.name              |TestSuite          |2  |
      * |spark.driver.host           |192.168.1.3        |3  |
      * |spark.driver.port           |58420              |4  |
      * |spark.executor.id           |driver             |5  |
      * |spark.master                |local[2]           |6  |
      * |spark.sql.shuffle.partitions|2                  |7  |
      * +----------------------------+-------------------+---+
      *
      * root
      * |-- key: string (nullable = false)
      * |-- value: string (nullable = false)
      * |-- rw: integer (nullable = true)
      */

    val map = df.head().getValuesMap(df.columns)
    println(map)
    println(map("key"))
    println(map("value"))
    println(map("rw"))
    println("Printing using for comprehension")
    map.foreach(println)

    /**
      * Map(key -> spark.app.id, value -> local-1594644271573, rw -> 1)
      * spark.app.id
      * local-1594644271573
      * 1
      * Printing using for comprehension
      * (key,spark.app.id)
      * (value,local-1594644271573)
      * (rw,1)
      */
  }

  // ############################################################################################################

  @Test
  def test62887001(): Unit = {
    val threshold = 5
    val cols = Range(1, 100).map(f => s"$f as col$f").mkString(", ")
    val df1 = spark.sql(s"select $cols")
    df1.show(false)
    df1.printSchema()
    /**
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      * |col1|col2|col3|col4|col5|col6|col7|col8|col9|col10|col11|col12|col13|col14|col15|col16|col17|col18|col19|col20|col21|col22|col23|col24|col25|col26|col27|col28|col29|col30|col31|col32|col33|col34|col35|col36|col37|col38|col39|col40|col41|col42|col43|col44|col45|col46|col47|col48|col49|col50|col51|col52|col53|col54|col55|col56|col57|col58|col59|col60|col61|col62|col63|col64|col65|col66|col67|col68|col69|col70|col71|col72|col73|col74|col75|col76|col77|col78|col79|col80|col81|col82|col83|col84|col85|col86|col87|col88|col89|col90|col91|col92|col93|col94|col95|col96|col97|col98|col99|
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      * |1   |2   |3   |4   |5   |6   |7   |8   |9   |10   |11   |12   |13   |14   |15   |16   |17   |18   |19   |20   |21   |22   |23   |24   |25   |26   |27   |28   |29   |30   |31   |32   |33   |34   |35   |36   |37   |38   |39   |40   |41   |42   |43   |44   |45   |46   |47   |48   |49   |50   |51   |52   |53   |54   |55   |56   |57   |58   |59   |60   |61   |62   |63   |64   |65   |66   |67   |68   |69   |70   |71   |72   |73   |74   |75   |76   |77   |78   |79   |80   |81   |82   |83   |84   |85   |86   |87   |88   |89   |90   |91   |92   |93   |94   |95   |96   |97   |98   |99   |
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      *
      * root
      * |-- col1: integer (nullable = false)
      * |-- col2: integer (nullable = false)
      * |-- col3: integer (nullable = false)
      * |-- col4: integer (nullable = false)
      * ...
      */

    val stringCol = df1.columns.map(c => s"'$c', cast(`$c` as int)").mkString(", ")

    val processedDF = df1.selectExpr(s"stack(${df1.columns.length}, $stringCol) as (name, value)")
      .filter(s"value > $threshold")
    processedDF.show(false)
    /**
      * -----+-----+
      * |name |value|
      * +-----+-----+
      * |col6 |6    |
      * |col7 |7    |
      * |col8 |8    |
      * |col9 |9    |
      * |col10|10   |
      * |col11|11   |
      * |col12|12   |
      * |col13|13   |
      * |col14|14   |
      * |col15|15   |
      * |col16|16   |
      * |col17|17   |
      * |col18|18   |
      * |col19|19   |
      * |col20|20   |
      * |col21|21   |
      * |col22|22   |
      * |col23|23   |
      * |col24|24   |
      * |col25|25   |
      * +-----+-----+
      * only showing top 20 rows
      */
  }

  // ############################################################################################################

  @Test
  def test62907356(): Unit = {
    val table_df = Seq(
      (10, 20, 30, 40, 50),
      (100, 200, 300, 400, 500),
      (111, 222, 333, 444, 555),
      (1123, 2123, 3123, 4123, 5123),
      (1321, 2321, 3321, 4321, 5321)
    ).toDF("col_1", "col_2", "col_3", "col_4", "col_5")
    table_df.show(false)
    table_df.printSchema()

    /**
      * +-----+-----+-----+-----+-----+
      * |col_1|col_2|col_3|col_4|col_5|
      * +-----+-----+-----+-----+-----+
      * |10   |20   |30   |40   |50   |
      * |100  |200  |300  |400  |500  |
      * |111  |222  |333  |444  |555  |
      * |1123 |2123 |3123 |4123 |5123 |
      * |1321 |2321 |3321 |4321 |5321 |
      * +-----+-----+-----+-----+-----+
      *
      * root
      * |-- col_1: integer (nullable = false)
      * |-- col_2: integer (nullable = false)
      * |-- col_3: integer (nullable = false)
      * |-- col_4: integer (nullable = false)
      * |-- col_5: integer (nullable = false)
      */


    import com.som.spark.shared.RichDataFrame.implicits._
    table_df.statSummary()
      .show(false)

    /**
      * +-------+----+---+------+------------------+------------------+-------------------+
      * |columns|max |min|mean  |std               |skewness          |kurtosis           |
      * +-------+----+---+------+------------------+------------------+-------------------+
      * |col_1  |1321|10 |533.0 |634.0634826261484 |0.4334269738367066|-1.7463346405299973|
      * |col_2  |2321|20 |977.2 |1141.1895986206675|0.405051373873868 |-1.7997419516751323|
      * |col_3  |3321|30 |1421.4|1649.399072389699 |0.3979251063785061|-1.8119558312496056|
      * |col_4  |4321|40 |1865.6|2157.926620624529 |0.3950204738145622|-1.816512420634769 |
      * |col_5  |5321|50 |2309.8|2666.5902759891706|0.3935246673563024|-1.81866856281125  |
      * +-------+----+---+------+------------------+------------------+-------------------+
      */

    table_df.statSummary("sum", "count", "25%", "75%")
      .show(false)

    /**
      * +-------+-----+-----+---+----+
      * |columns|sum  |count|25%|75% |
      * +-------+-----+-----+---+----+
      * |col_1  |2665 |5    |100|1123|
      * |col_2  |4886 |5    |200|2123|
      * |col_3  |7107 |5    |300|3123|
      * |col_4  |9328 |5    |400|4123|
      * |col_5  |11549|5    |500|5123|
      * +-------+-----+-----+---+----+
      */

    val count_star = table_df.count()
    table_df.statSummary("count", "approx_count_distinct", "5%", "25%", "50%", "75%", "95%",
    "max", "min", "mean", "std", "SKEWNESS", "KURTOSIS", "VARIANCE")
      .withColumn("count_star", lit(count_star))
      .selectExpr(
        "columns AS Column_Name",
        "COUNT AS number_of_values",
        "approx_count_distinct AS number_of_distinct_values",
        "approx_count_distinct AS distinct_count_with_nan",
        "(approx_count_distinct - 1) AS distinct_count_without_nan",
        "(count == approx_count_distinct) AS is_unique",
        "(count_star - count) AS number_of_missing_values",
        "((count_star - count)/count) AS percentage_of_missing_values",
        "(approx_count_distinct/count) AS percentage_of_unique_values",
        "`5%` AS 05_PCT",
        "`25%` AS 25_PCT",
        "`50%` AS 50_PCT",
        "`75%` AS 75_PCT",
        "`95%` AS 95_PCT",
        "MAX AS max",
        "MIN AS min",
        "MEAN AS mean",
        "STD AS std",
        "SKEWNESS AS skewness",
        "KURTOSIS AS kurtosis",
        "(MAX - MIN) AS range",
        "VARIANCE AS variance"
      ).show(false)

    /**
      * +-----------+----------------+-------------------------+-----------------------+--------------------------+---------+------------------------+----------------------------+---------------------------+------+------+------+------+------+----+---+------+------------------+------------------+-------------------+------+------------------+
      * |Column_Name|number_of_values|number_of_distinct_values|distinct_count_with_nan|distinct_count_without_nan|is_unique|number_of_missing_values|percentage_of_missing_values|percentage_of_unique_values|05_PCT|25_PCT|50_PCT|75_PCT|95_PCT|max |min|mean  |std               |skewness          |kurtosis           |range |variance          |
      * +-----------+----------------+-------------------------+-----------------------+--------------------------+---------+------------------------+----------------------------+---------------------------+------+------+------+------+------+----+---+------+------------------+------------------+-------------------+------+------------------+
      * |col_1      |5               |5                        |5                      |4.0                       |true     |0.0                     |0.0                         |1.0                        |10    |100   |111   |1123  |1321  |1321|10 |533.0 |634.0634826261484 |0.4334269738367066|-1.7463346405299973|1311.0|402036.5          |
      * |col_2      |5               |5                        |5                      |4.0                       |true     |0.0                     |0.0                         |1.0                        |20    |200   |222   |2123  |2321  |2321|20 |977.2 |1141.1895986206675|0.405051373873868 |-1.7997419516751323|2301.0|1302313.7000000002|
      * |col_3      |5               |5                        |5                      |4.0                       |true     |0.0                     |0.0                         |1.0                        |30    |300   |333   |3123  |3321  |3321|30 |1421.4|1649.399072389699 |0.3979251063785061|-1.8119558312496056|3291.0|2720517.3         |
      * |col_4      |5               |5                        |5                      |4.0                       |true     |0.0                     |0.0                         |1.0                        |40    |400   |444   |4123  |4321  |4321|40 |1865.6|2157.926620624529 |0.3950204738145622|-1.816512420634769 |4281.0|4656647.3         |
      * |col_5      |5               |5                        |5                      |4.0                       |true     |0.0                     |0.0                         |1.0                        |50    |500   |555   |5123  |5321  |5321|50 |2309.8|2666.5902759891706|0.3935246673563024|-1.81866856281125  |5271.0|7110703.7         |
      * +-----------+----------------+-------------------------+-----------------------+--------------------------+---------+------------------------+----------------------------+---------------------------+------+------+------+------+------+----+---+------+------------------+------------------+-------------------+------+------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62909508(): Unit = {
    val data =
      """
        |{
        |    "appointmentRef": "Appointment/12213#4200",
        |    "encounterLengh": "2",
        |    "billingAccount": "savingsAccount",
        |    "hospitalization": "{\"preAdmissionIdentifierSystem\":\"https://system123445.html\",\"preAdmissionIdentifierValue\":\"pqr\",\"origin\":\"hospital\",\"admitSourceCode\":\"outp\",\"admitSourceReason\":\"some thing\",\"eid\":200,\"destination\":\"hospital\"}",
        |    "resourceType": "Encounter",
        |    "priority": "abc",
        |    "status": "triaged",
        |    "eid": "200",
        |    "subject": "Patient/435"
        |}
      """.stripMargin

    val ds = Seq(data).toDF()
    ds.show(false)
    ds.printSchema()

    /**
      * +------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
      * +------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |
      * {
      * "appointmentRef": "Appointment/12213#4200",
      * "encounterLengh": "2",
      * "billingAccount": "savingsAccount",
      * "hospitalization": "{\"preAdmissionIdentifierSystem\":\"https://system123445.html\",\"preAdmissionIdentifierValue\":\"pqr\",\"origin\":\"hospital\",\"admitSourceCode\":\"outp\",\"admitSourceReason\":\"some thing\",\"eid\":200,\"destination\":\"hospital\"}",
      * "resourceType": "Encounter",
      * "priority": "abc",
      * "status": "triaged",
      * "eid": "200",
      * "subject": "Patient/435"
      * }
      * |
      * +------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      *
      * root
      * |-- value: string (nullable = true)
      */
    ds.withColumn("value", translate($"value", "\\", ""))
      .show(false)

    /**
      * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
      * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |
      * {
      * "appointmentRef": "Appointment/12213#4200",
      * "encounterLengh": "2",
      * "billingAccount": "savingsAccount",
      * "hospitalization": "{"preAdmissionIdentifierSystem":"https://system123445.html","preAdmissionIdentifierValue":"pqr","origin":"hospital","admitSourceCode":"outp","admitSourceReason":"some thing","eid":200,"destination":"hospital"}",
      * "resourceType": "Encounter",
      * "priority": "abc",
      * "status": "triaged",
      * "eid": "200",
      * "subject": "Patient/435"
      * }
      * |
      * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      */
  }
  // ############################################################################################################

  @Test
  def test62908025(): Unit = {
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    df.show(false)
    df.printSchema()

    /**
      * +---------------------+
      * |features             |
      * +---------------------+
      * |(5,[1,3],[1.0,7.0])  |
      * |[2.0,0.0,3.0,4.0,5.0]|
      * |[4.0,0.0,0.0,6.0,7.0]|
      * +---------------------+
      *
      * root
      * |-- features: vector (nullable = true)
      */


   val shuffleVector = udf((vector: Vector) =>
     Vectors.dense(scala.util.Random.shuffle(mutable.WrappedArray.make[Double](vector.toArray)).toArray)
   )

    val p = df.withColumn("shuffled_vector", shuffleVector($"features"))
    p.show(false)
    p.printSchema()

    /**
      * +---------------------+---------------------+
      * |features             |shuffled_vector      |
      * +---------------------+---------------------+
      * |(5,[1,3],[1.0,7.0])  |[1.0,0.0,0.0,0.0,7.0]|
      * |[2.0,0.0,3.0,4.0,5.0]|[0.0,3.0,2.0,5.0,4.0]|
      * |[4.0,0.0,0.0,6.0,7.0]|[4.0,7.0,6.0,0.0,0.0]|
      * +---------------------+---------------------+
      *
      * root
      * |-- features: vector (nullable = true)
      * |-- shuffled_vector: vector (nullable = true)
      */

    val shuffleVectorToSparse = udf((vector: Vector) =>
      Vectors.dense(scala.util.Random.shuffle(mutable.WrappedArray.make[Double](vector.toArray)).toArray).toSparse
    )

    val p1 = df.withColumn("shuffled_vector", shuffleVectorToSparse($"features"))
    p1.show(false)
    p1.printSchema()

    /**
      * +---------------------+-------------------------------+
      * |features             |shuffled_vector                |
      * +---------------------+-------------------------------+
      * |(5,[1,3],[1.0,7.0])  |(5,[0,3],[1.0,7.0])            |
      * |[2.0,0.0,3.0,4.0,5.0]|(5,[1,2,3,4],[5.0,3.0,2.0,4.0])|
      * |[4.0,0.0,0.0,6.0,7.0]|(5,[1,3,4],[7.0,4.0,6.0])      |
      * +---------------------+-------------------------------+
      *
      * root
      * |-- features: vector (nullable = true)
      * |-- shuffled_vector: vector (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62912383(): Unit = {
    val df = spark.sql("select cast('2020-07-12' as date) as date")
    df.show(false)
    df.printSchema()

    /**
      * +----------+
      * |date      |
      * +----------+
      * |2020-07-15|
      * +----------+
      *
      * root
      * |-- date: date (nullable = true)
      */

    // week starting from SUNDAY and ending SATURDAY
    df.withColumn("week_end", next_day($"date", "SAT"))
      .withColumn("week_start", date_sub($"week_end", 6))
      .show(false)

    /**
      * +----------+----------+----------+
      * |date      |week_end  |week_start|
      * +----------+----------+----------+
      * |2020-07-12|2020-07-18|2020-07-12|
      * +----------+----------+----------+
      */

    // week starting from MONDAY and ending SUNDAY
    df.withColumn("week_end", next_day($"date", "SUN"))
      .withColumn("week_start", date_sub($"week_end", 6))
      .show(false)

    /**
      * +----------+----------+----------+
      * |date      |week_end  |week_start|
      * +----------+----------+----------+
      * |2020-07-12|2020-07-19|2020-07-13|
      * +----------+----------+----------+
      */

    // week starting from TUESDAY and ending MONDAY
    df.withColumn("week_end", next_day($"date", "MON"))
      .withColumn("week_start", date_sub($"week_end", 6))
      .show(false)

    /**
      * +----------+----------+----------+
      * |date      |week_end  |week_start|
      * +----------+----------+----------+
      * |2020-07-12|2020-07-13|2020-07-07|
      * +----------+----------+----------+
      */

  }

  // ############################################################################################################

  @Test
  def test62914863(): Unit = {
    //62914863
    val threshold = 1
    val df1 = spark.createDataFrame(spark.sparkContext.parallelize(Seq(1,1,1,2,2,3).map(f => Row.apply(f))),
      StructType(StructField("my_col", IntegerType) :: Nil))
      .repartition(5, col("my_col") % 3)

    df1.foreachPartition(iter => {
      println("### partition ### " + iter.map(_.toString()).mkString("=="))
    })

    // 5 partitions (2 are empty, 3 has some value)
    /**
      * ### partition ###
      * ### partition ###
      * ### partition ### [1]==[1]==[1]
      * ### partition ### [3]
      * ### partition ### [2]==[2]
      */

    df1.printSchema()
    /**
      * root
      * |-- my_col: integer (nullable = true)
      */

    // goal is to remove empty partitions as well as the partitions having records less than thresholds
    val nulls: Seq[Any] = Seq.fill(df1.columns.length)(null)
    val p = df1.mapPartitions(iter => {
      val seq = iter.toSeq
      val length = seq.length
      if(length > 0) seq.map(r => Row.fromSeq(r.toSeq :+ length)).iterator
      else Iterator(Row.fromSeq(nulls :+ length))
    }
    )(RowEncoder.apply(df1.schema.add(StructField("length", IntegerType))))

    p.show(false)

    p.filter(s"length > $threshold")
      .coalesce(p.rdd.getNumPartitions - p.filter("length=0").count().toInt)
      .foreachPartition(iter => {
        println("### partition ### " + iter.map(_.toString()).mkString("=="))
      })
    p.filter(s"length > $threshold")
      .drop("length").show(false)

    println()
    p.foreachPartition(iter => {
      println("### partition ### " + iter.map(_.toString()).mkString("=="))
    })
  }
  // ############################################################################################################

  @Test
  def test62911204(): Unit = {
    // 62911204
    val frame = Seq("SNNNN","NNNNN","NNSNN","SNSNS","NNNNS").toDF("values")
    frame.withColumn("x", length($"values") - length(substring_index($"values", "S", -1)))
      .show(false)
    /**
      * +------+---+
      * |values|x  |
      * +------+---+
      * |SNNNN |1  |
      * |NNNNN |0  |
      * |NNSNN |3  |
      * |SNSNS |5  |
      * |NNNNS |5  |
      * +------+---+
      */
  }
  // ############################################################################################################

  @Test
  def test62913292(): Unit = {

    val data =
      """
        |user, location, date, date_minus_6
        |id1, loc1, 20100110, 20100104
        |id1, loc1, 20100111, 20100105
        |id1, loc2, 20100111, 20100105
        |id2, loc1, 20100111, 20100105
        |id3, loc1, 20100108, 20100102
      """.stripMargin

    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\,").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    val w = Window.partitionBy("location", "date")
    df.withColumn("num_of_users",
      sum(when($"date" >= $"date_minus_6" && $"date" <= $"date", 1).otherwise(0)).over(w)
    ).select("location", "date", "num_of_users")
      .groupBy("location", "date")
      .agg(sum("num_of_users").as("num_of_users"))
      .orderBy("location", "date")
      .show(false)
  }

  // ############################################################################################################

  @Test
  def test62913771(): Unit = {
    // 62928072
   val df = Seq((0, 1, 1.0), (1, 2, 2.0)).toDF("a", "b", "b")

    df.show(false)
    df.printSchema()

    /**
      * +---+---+---+
      * |a  |b  |b  |
      * +---+---+---+
      * |0  |1  |1.0|
      * |1  |2  |2.0|
      * +---+---+---+
      *
      * root
      * |-- a: integer (nullable = false)
      * |-- b: integer (nullable = false)
      * |-- b: double (nullable = false)
      */
    df.toDF("a", "b", "b2").drop("b2").show(false)
    /**
      * +---+---+
      * |a  |b  |
      * +---+---+
      * |0  |1  |
      * |1  |2  |
      * +---+---+
      */

    // 62928495
    val df3 = Seq(("tom", "jerry"), ("hank", "polo")).toDF("firstname", "lastname")
    df3.show(false)
    df3.printSchema()
    /**
      * +---------+--------+
      * |firstname|lastname|
      * +---------+--------+
      * |tom      |jerry   |
      * |hank     |polo    |
      * +---------+--------+
      *
      * root
      * |-- firstname: string (nullable = true)
      * |-- lastname: string (nullable = true)
      */

    val p = df3.withColumn("fullname", concat(col("firstname"), col("lastname")))
      .as[invoiceColumns]
    p.show(false)
    p.printSchema()
    /**
      * +---------+--------+--------+
      * |firstname|lastname|fullname|
      * +---------+--------+--------+
      * |tom      |jerry   |tomjerry|
      * |hank     |polo    |hankpolo|
      * +---------+--------+--------+
      *
      * root
      * |-- firstname: string (nullable = true)
      * |-- lastname: string (nullable = true)
      * |-- fullname: string (nullable = true)
      */

    val p1 = df3.map{case Row(firstname: String, lastname: String) => new invoiceColumns2(firstname, lastname)}
    p1.show(false)
    p1.printSchema()
    /**
      * +---------+--------+--------+
      * |firstname|lastname|fullname|
      * +---------+--------+--------+
      * |tom      |jerry   |tomjerry|
      * |hank     |polo    |hankpolo|
      * +---------+--------+--------+
      *
      * root
      * |-- firstname: string (nullable = true)
      * |-- lastname: string (nullable = true)
      * |-- fullname: string (nullable = true)
      */


    // 62913771
    // Since scala has only Tuple22 & Product22 defined, you can create dataframe of Tuple22, as below-
    val df1 = Seq.empty[(
      String, String, String, String, String, String, String, String, String, String,
        String, String, String, String, String, String, String, String, String, String,
        String, String)].toDF(Range(1, 23).map(s => s"col$s"): _*)
    df1.show(false)
    /**
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      * |col1|col2|col3|col4|col5|col6|col7|col8|col9|col10|col11|col12|col13|col14|col15|col16|col17|col18|col19|col20|col21|col22|
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      */

    // To create dataframe of Seq[Tuple23[String]], you need to create Product23 and
    val df2 = Seq.empty[Tuple23[
      String, String, String, String, String, String, String, String, String, String,
      String, String, String, String, String, String, String, String, String, String,
      String, String, String]].toDF(Range(1, 24).map(s => s"col$s"): _*)
    df2.show(false)

    /**
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      * |col1|col2|col3|col4|col5|col6|col7|col8|col9|col10|col11|col12|col13|col14|col15|col16|col17|col18|col19|col20|col21|col22|col23|
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      * +----+----+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      */


  }

  // ############################################################################################################

  @Test
  def test62928053(): Unit = {
    val df = spark.sql(
      """
        | select id, emprecords
        |  from values
        | (201, array(named_struct('emprec', array(named_struct('firstName' , 'tom', 'lastName', 'hank'))))),
        | (202, null)
        | T(id, emprecords)
      """.stripMargin)
    df.printSchema()
    df.show(false)

    /**
      * root
      * |-- id: integer (nullable = false)
      * |-- emprecords: array (nullable = true)
      * |    |-- element: struct (containsNull = false)
      * |    |    |-- emprec: array (nullable = false)
      * |    |    |    |-- element: struct (containsNull = false)
      * |    |    |    |    |-- firstName: string (nullable = false)
      * |    |    |    |    |-- lastName: string (nullable = false)
      *
      * +---+-----------------+
      * |id |emprecords       |
      * +---+-----------------+
      * |201|[[[[tom, hank]]]]|
      * |202|null             |
      * +---+-----------------+
      */

    df.withColumn("emprecords",
      expr("ifnull(emprecords, array(named_struct('emprec', array(named_struct('firstName' , null, 'lastName', null)))))"))
      .show(false)

    /**
      * +---+-----------------+
      * |id |emprecords       |
      * +---+-----------------+
      * |201|[[[[tom, hank]]]]|
      * |202|[[[[,]]]]        |
      * +---+-----------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62933135(): Unit = {

    val data = List(
      ("20", "score", "school",  14 ,12),
      ("21", "score", "school",  13 , 13),
      ("22", "rate", "school",  11 ,14),
      ("23", "score", "school",  11 ,14),
      ("24", "rate", "school",  12 ,12),
      ("25", "score", "school", 11 ,14)
    )
    val df = data.toDF("id", "code", "entity", "value1","value2")
    df.show
    /**
      * +---+-----+------+------+------+
      * | id| code|entity|value1|value2|
      * +---+-----+------+------+------+
      * | 20|score|school|    14|    12|
      * | 21|score|school|    13|    13|
      * | 22| rate|school|    11|    14|
      * | 23|score|school|    11|    14|
      * | 24| rate|school|    12|    12|
      * | 25|score|school|    11|    14|
      * +---+-----+------+------+------+
      */

    //this look up data populated from DB.

    val ll = List(
      ("aaaa", 11),
      ("aaa", 12),
      ("aa", 13),
      ("a", 14)
    )
    val codeValudeDf = ll.toDF( "code", "value")
    codeValudeDf.show
    /**
      * +----+-----+
      * |code|value|
      * +----+-----+
      * |aaaa|   11|
      * | aaa|   12|
      * |  aa|   13|
      * |   a|   14|
      * +----+-----+
      */

    val lookUp = spark.sparkContext
      .broadcast(codeValudeDf.map{case Row(code: String, value: Integer) => value -> code}
      .collect().toMap)

    val look_up = udf((value: Integer) => lookUp.value.get(value))

    df.withColumn("value1",
      when($"code" === "score", look_up($"value1")).otherwise($"value1".cast("string")))
      .withColumn("value2",
        when($"code" === "score", look_up($"value2")).otherwise($"value2".cast("string")))
      .show(false)
    /**
      * +---+-----+------+------+------+
      * |id |code |entity|value1|value2|
      * +---+-----+------+------+------+
      * |20 |score|school|a     |aaa   |
      * |21 |score|school|aa    |aa    |
      * |22 |rate |school|11    |14    |
      * |23 |score|school|aaaa  |a     |
      * |24 |rate |school|12    |12    |
      * |25 |score|school|aaaa  |a     |
      * +---+-----+------+------+------+
      */




    // 62932982
    val data1 =
      """
        | a1| b1|  c1| d1| e1
        |  1|  a|foo1|  4|  5
        |   |  b| bar|  4|  6
        |   |  c| mnc|   |  7
      """.stripMargin

    val stringDS = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
//      .option("nullValue", "null")
      .csv(stringDS)
    df1.show(false)
    df1.printSchema()

    /**
      * +----+---+----+----+---+
      * |a1  |b1 |c1  |d1  |e1 |
      * +----+---+----+----+---+
      * |1   |a  |foo1|4   |5  |
      * |null|b  |bar |4   |6  |
      * |null|c  |mnc |null|7  |
      * +----+---+----+----+---+
      *
      * root
      * |-- a1: integer (nullable = true)
      * |-- b1: string (nullable = true)
      * |-- c1: string (nullable = true)
      * |-- d1: integer (nullable = true)
      * |-- e1: integer (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62959748(): Unit = {
    val df = spark.sparkContext.parallelize(List(
      ("Company B","xi2", "2020-07-02T01:07:00.000+0000", "2020-07-02T02:29:00.000+0000"),
      ("Company A","xi1", "2020-07-01T23:55:00.000+0000", "2020-07-02T01:17:00.000+0000"),
      ("Company B","xi2", "2020-07-01T22:31:00.000+0000", "2020-07-01T23:53:00.000+0000"),
      ("Company B","xi1", "2020-07-01T23:07:00.000+0000", "2020-07-02T00:29:00.000+0000"),
      ("Company A","xi1", "2020-07-01T22:19:00.000+0000", "2020-07-01T23:41:00.000+0000"),
      ("Company B","xi1", "2020-07-02T00:07:00.000+0000", "2020-07-02T01:29:00.000+0000"),
      ("Company B","xi1", "2020-07-02T00:55:00.000+0000", "2020-07-02T02:17:00.000+0000"),
      ("Company A","xi1", "2020-07-02T00:19:00.000+0000", "2020-07-02T01:41:00.000+0000"),
      ("Company A","xi2", "2020-07-01T22:55:00.000+0000", "2020-07-02T00:17:00.000+0000"),
      ("Company B","xi2", "2020-07-02T00:43:00.000+0000", "2020-07-02T02:05:00.000+0000"),
      ("Company A","xi2", "2020-07-01T23:31:00.000+0000", "2020-07-02T00:53:00.000+0000"),
      ("Company B","xi1", "2020-07-01T23:19:00.000+0000", "2020-07-02T00:41:00.000+0000"),
      ("Company A","xi2", "2020-07-01T23:43:00.000+0000", "2020-07-02T01:05:00.000+0000"),
      ("Company A","xi2", "2020-07-02T00:31:00.000+0000", "2020-07-02T01:53:00.000+0000"),
      ("Company A","xi2", "2020-07-01T22:43:00.000+0000", "2020-07-02T00:05:00.000+0000")  ))
      .toDF("customer","device_model","start_timestamp","end_timestamp")
      .withColumn("start_timestamp", to_timestamp($"start_timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSZ"))
      .withColumn("end_timestamp", to_timestamp($"end_timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSZ"))
    df.show(false)
    /**
      * +---------+------------+-------------------+-------------------+
      * |customer |device_model|start_timestamp    |end_timestamp      |
      * +---------+------------+-------------------+-------------------+
      * |Company B|xi2         |2020-07-02 06:37:00|2020-07-02 07:59:00|
      * |Company A|xi1         |2020-07-02 05:25:00|2020-07-02 06:47:00|
      * |Company B|xi2         |2020-07-02 04:01:00|2020-07-02 05:23:00|
      * |Company B|xi1         |2020-07-02 04:37:00|2020-07-02 05:59:00|
      * |Company A|xi1         |2020-07-02 03:49:00|2020-07-02 05:11:00|
      * |Company B|xi1         |2020-07-02 05:37:00|2020-07-02 06:59:00|
      * |Company B|xi1         |2020-07-02 06:25:00|2020-07-02 07:47:00|
      * |Company A|xi1         |2020-07-02 05:49:00|2020-07-02 07:11:00|
      * |Company A|xi2         |2020-07-02 04:25:00|2020-07-02 05:47:00|
      * |Company B|xi2         |2020-07-02 06:13:00|2020-07-02 07:35:00|
      * |Company A|xi2         |2020-07-02 05:01:00|2020-07-02 06:23:00|
      * |Company B|xi1         |2020-07-02 04:49:00|2020-07-02 06:11:00|
      * |Company A|xi2         |2020-07-02 05:13:00|2020-07-02 06:35:00|
      * |Company A|xi2         |2020-07-02 06:01:00|2020-07-02 07:23:00|
      * |Company A|xi2         |2020-07-02 04:13:00|2020-07-02 05:35:00|
      * +---------+------------+-------------------+-------------------+
      */

    val intervalInMinutes = 30
    val seconds = intervalInMinutes * 60 // seconds
    val p = df.withColumn("new_start", to_timestamp(floor($"start_timestamp".cast("long")/ seconds ) * seconds))
      .withColumn("splits", sequence(
      $"new_start",
      $"end_timestamp",
      expr(s"interval $intervalInMinutes MINUTE")))
    p.show(false)

    /**
      * +---------+------------+-------------------+-------------------+-------------------+------------------------------------------------------------------------------------+
      * |customer |device_model|start_timestamp    |end_timestamp      |new_start          |splits                                                                              |
      * +---------+------------+-------------------+-------------------+-------------------+------------------------------------------------------------------------------------+
      * |Company B|xi2         |2020-07-02 06:37:00|2020-07-02 07:59:00|2020-07-02 06:30:00|[2020-07-02 06:30:00, 2020-07-02 07:00:00, 2020-07-02 07:30:00]                     |
      * |Company A|xi1         |2020-07-02 05:25:00|2020-07-02 06:47:00|2020-07-02 05:00:00|[2020-07-02 05:00:00, 2020-07-02 05:30:00, 2020-07-02 06:00:00, 2020-07-02 06:30:00]|
      * |Company B|xi2         |2020-07-02 04:01:00|2020-07-02 05:23:00|2020-07-02 04:00:00|[2020-07-02 04:00:00, 2020-07-02 04:30:00, 2020-07-02 05:00:00]                     |
      * |Company B|xi1         |2020-07-02 04:37:00|2020-07-02 05:59:00|2020-07-02 04:30:00|[2020-07-02 04:30:00, 2020-07-02 05:00:00, 2020-07-02 05:30:00]                     |
      * |Company A|xi1         |2020-07-02 03:49:00|2020-07-02 05:11:00|2020-07-02 03:30:00|[2020-07-02 03:30:00, 2020-07-02 04:00:00, 2020-07-02 04:30:00, 2020-07-02 05:00:00]|
      * |Company B|xi1         |2020-07-02 05:37:00|2020-07-02 06:59:00|2020-07-02 05:30:00|[2020-07-02 05:30:00, 2020-07-02 06:00:00, 2020-07-02 06:30:00]                     |
      * |Company B|xi1         |2020-07-02 06:25:00|2020-07-02 07:47:00|2020-07-02 06:00:00|[2020-07-02 06:00:00, 2020-07-02 06:30:00, 2020-07-02 07:00:00, 2020-07-02 07:30:00]|
      * |Company A|xi1         |2020-07-02 05:49:00|2020-07-02 07:11:00|2020-07-02 05:30:00|[2020-07-02 05:30:00, 2020-07-02 06:00:00, 2020-07-02 06:30:00, 2020-07-02 07:00:00]|
      * |Company A|xi2         |2020-07-02 04:25:00|2020-07-02 05:47:00|2020-07-02 04:00:00|[2020-07-02 04:00:00, 2020-07-02 04:30:00, 2020-07-02 05:00:00, 2020-07-02 05:30:00]|
      * |Company B|xi2         |2020-07-02 06:13:00|2020-07-02 07:35:00|2020-07-02 06:00:00|[2020-07-02 06:00:00, 2020-07-02 06:30:00, 2020-07-02 07:00:00, 2020-07-02 07:30:00]|
      * |Company A|xi2         |2020-07-02 05:01:00|2020-07-02 06:23:00|2020-07-02 05:00:00|[2020-07-02 05:00:00, 2020-07-02 05:30:00, 2020-07-02 06:00:00]                     |
      * |Company B|xi1         |2020-07-02 04:49:00|2020-07-02 06:11:00|2020-07-02 04:30:00|[2020-07-02 04:30:00, 2020-07-02 05:00:00, 2020-07-02 05:30:00, 2020-07-02 06:00:00]|
      * |Company A|xi2         |2020-07-02 05:13:00|2020-07-02 06:35:00|2020-07-02 05:00:00|[2020-07-02 05:00:00, 2020-07-02 05:30:00, 2020-07-02 06:00:00, 2020-07-02 06:30:00]|
      * |Company A|xi2         |2020-07-02 06:01:00|2020-07-02 07:23:00|2020-07-02 06:00:00|[2020-07-02 06:00:00, 2020-07-02 06:30:00, 2020-07-02 07:00:00]                     |
      * |Company A|xi2         |2020-07-02 04:13:00|2020-07-02 05:35:00|2020-07-02 04:00:00|[2020-07-02 04:00:00, 2020-07-02 04:30:00, 2020-07-02 05:00:00, 2020-07-02 05:30:00]|
      * +---------+------------+-------------------+-------------------+-------------------+------------------------------------------------------------------------------------+
      */

    p.select($"customer", $"device_model", explode($"splits").as("timeinterval"))
      .groupBy("timeinterval", "customer")
      .pivot("device_model")
      .agg(
        count("device_model")
      )
      .withColumn("xi1", coalesce($"xi1", lit(0)))
      .withColumn("xi2", coalesce($"xi2", lit(0)))
      .orderBy("timeinterval", "customer")
      .show(false)

    /**
      * +-------------------+---------+---+---+
      * |timeinterval       |customer |xi1|xi2|
      * +-------------------+---------+---+---+
      * |2020-07-02 03:30:00|Company A|1  |0  |
      * |2020-07-02 04:00:00|Company A|1  |2  |
      * |2020-07-02 04:00:00|Company B|0  |1  |
      * |2020-07-02 04:30:00|Company A|1  |2  |
      * |2020-07-02 04:30:00|Company B|2  |1  |
      * |2020-07-02 05:00:00|Company A|2  |4  |
      * |2020-07-02 05:00:00|Company B|2  |1  |
      * |2020-07-02 05:30:00|Company A|2  |4  |
      * |2020-07-02 05:30:00|Company B|3  |0  |
      * |2020-07-02 06:00:00|Company A|2  |3  |
      * |2020-07-02 06:00:00|Company B|3  |1  |
      * |2020-07-02 06:30:00|Company A|2  |2  |
      * |2020-07-02 06:30:00|Company B|2  |2  |
      * |2020-07-02 07:00:00|Company A|1  |1  |
      * |2020-07-02 07:00:00|Company B|1  |2  |
      * |2020-07-02 07:30:00|Company B|1  |2  |
      * +-------------------+---------+---+---+
      */

  }

  // ############################################################################################################

  @Test
  def test62957720(): Unit = {
    val df = Seq(("A", "x", "3"), ("A", "y", "1"), ("B", "a", "2"), ("B", "b", "5"), ("C", "v", "2"), ("D", "f", "6"))
    .toDF("id", "manager", "score")
    df.show(false)

    /**
      * +---+-------+-----+
      * |id |manager|score|
      * +---+-------+-----+
      * |A  |x      |3    |
      * |A  |y      |1    |
      * |B  |a      |2    |
      * |B  |b      |5    |
      * |C  |v      |2    |
      * |D  |f      |6    |
      * +---+-------+-----+
      */

    df.createOrReplaceTempView("employee")
    spark.sql("select * from employee where id in (select distinct id from employee order by id limit 3)")
      .show(false)

    /**
      * +---+-------+-----+
      * |id |manager|score|
      * +---+-------+-----+
      * |A  |x      |3    |
      * |A  |y      |1    |
      * |B  |a      |2    |
      * |B  |b      |5    |
      * |C  |v      |2    |
      * +---+-------+-----+
      */
  }

  // ############################################################################################################

  @Test
  def test62956891(): Unit = {
    val df = spark.sql("select 'a' as Y, 3.2 as X1, 4.5 as X2")
    df.show(false)
    df.printSchema()

    /**
      * +---+---+---+
      * |Y  |X1 |X2 |
      * +---+---+---+
      * |a  |3.2|4.5|
      * +---+---+---+
      *
      * root
      * |-- Y: string (nullable = false)
      * |-- X1: decimal(2,1) (nullable = false)
      * |-- X2: decimal(2,1) (nullable = false)
      */
    import org.apache.spark.ml.feature.VectorAssembler
    val features = new VectorAssembler()
      .setInputCols(Array("X1", "X2"))
      .setOutputCol("features")
      .transform(df)
    features.show(false)
    features.printSchema()

    /**
      * +---+---+---+---------+
      * |Y  |X1 |X2 |features |
      * +---+---+---+---------+
      * |a  |3.2|4.5|[3.2,4.5]|
      * +---+---+---+---------+
      *
      * root
      * |-- Y: string (nullable = false)
      * |-- X1: decimal(2,1) (nullable = false)
      * |-- X2: decimal(2,1) (nullable = false)
      * |-- features: vector (nullable = true)
      */

  }
  // ############################################################################################################

  @Test
  def test62953699(): Unit = {
    val df = Seq("#studiolife #aisl", "@user #white #su", "oh! yeah #123 #su.").toDF("tweet")
    df.withColumn("clean_tweet", regexp_replace($"tweet", "[\\W&&[^\\s+]]", ""))
      .show(false)

    /**
      * +------------------+---------------+
      * |tweet             |clean_tweet    |
      * +------------------+---------------+
      * |#studiolife #aisl |studiolife aisl|
      * |@user #white #su  |user white su  |
      * |oh! yeah #123 #su.|oh yeah 123 su |
      * +------------------+---------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62953149(): Unit = {
    val data =
      """
        |id date             key1
        |A1 2020-01-06        K1
        |A1 2020-01-06        K2
        |A1 2020-01-07        K3
        |A1 2020-01-07        K3
        |A1 2020-01-20        K3
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\s+").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df1.show(false)
    df1.printSchema()
    /**
      * +---+-------------------+----+
      * |id |date               |key1|
      * +---+-------------------+----+
      * |A1 |2020-01-06 00:00:00|K1  |
      * |A1 |2020-01-06 00:00:00|K2  |
      * |A1 |2020-01-07 00:00:00|K3  |
      * |A1 |2020-01-07 00:00:00|K3  |
      * |A1 |2020-01-20 00:00:00|K3  |
      * +---+-------------------+----+
      *
      * root
      * |-- id: string (nullable = true)
      * |-- date: timestamp (nullable = true)
      * |-- key1: string (nullable = true)
      */

    val w = Window.partitionBy("id").orderBy("date")
    val w1 = Window.partitionBy("id", "date")
      .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    df1.withColumn("last_date", lag(col("date"), 1).over(w))
      .withColumn("last_date", min(col("last_date")).over(w1))
      .withColumn("last_date", when($"date" =!= $"last_date", $"last_date"))
      .show(false)

    /**
      * +---+-------------------+----+-------------------+
      * |id |date               |key1|last_date          |
      * +---+-------------------+----+-------------------+
      * |A1 |2020-01-06 00:00:00|K1  |null               |
      * |A1 |2020-01-06 00:00:00|K2  |null               |
      * |A1 |2020-01-07 00:00:00|K3  |2020-01-06 00:00:00|
      * |A1 |2020-01-07 00:00:00|K3  |2020-01-06 00:00:00|
      * |A1 |2020-01-20 00:00:00|K3  |2020-01-07 00:00:00|
      * +---+-------------------+----+-------------------+
      */


  }
  // ############################################################################################################

  @Test
  def test62987950(): Unit = {
    val fileName = "log_data.txt"
    val path = getClass.getResource("/text/" + fileName).getPath
    val df = spark.read
      .option("sep", ":")
      .csv(path)
      .toDF("key", "value")
    df.show(false)
    df.printSchema()

    df.groupBy().pivot("key")
      .agg(
        first("value")
      )
      .show(false)
    df.select(when($"key"===lit("dateCreated"), $"value"),
      when($"key"===lit("customerId"), $"value"))
      .show(false)
  }

  // ############################################################################################################

  @Test
  def test62993903(): Unit = {
    val df = spark.sql(
      """
        |select model, year, timestamp
        | from values
        | ('i20', array(2019, 2018, 2017), '2020-07-20 10:42:38.935'),
        |  ('i10', array(2017), '2020-07-20 10:42:38.935')
        |  T(model, year, timestamp)
      """.stripMargin)
    df.show(false)

    /**
      * +-----+------------------+-----------------------+
      * |model|year              |timestamp              |
      * +-----+------------------+-----------------------+
      * |i20  |[2019, 2018, 2017]|2020-07-20 10:42:38.935|
      * |i10  |[2017]            |2020-07-20 10:42:38.935|
      * +-----+------------------+-----------------------+
      */
    df.createOrReplaceTempView("table")
    spark.sql("select timestamp, collect_list(struct(model, year)) as details from table group by timestamp")
      .toJSON
      .show(false)

    /**
      * +-------------------------------------------------------------------------------------------------------------------------+
      * |value                                                                                                                    |
      * +-------------------------------------------------------------------------------------------------------------------------+
      * |{"timestamp":"2020-07-20 10:42:38.935","details":[{"model":"i20","year":[2019,2018,2017]},{"model":"i10","year":[2017]}]}|
      * +-------------------------------------------------------------------------------------------------------------------------+
      */
    df.groupBy("timestamp")
      .agg(collect_list(struct("model", "year")).as("details"))
      .toJSON
      .show(false)

    /**
      * +-------------------------------------------------------------------------------------------------------------------------+
      * |value                                                                                                                    |
      * +-------------------------------------------------------------------------------------------------------------------------+
      * |{"timestamp":"2020-07-20 10:42:38.935","details":[{"model":"i20","year":[2019,2018,2017]},{"model":"i10","year":[2017]}]}|
      * +-------------------------------------------------------------------------------------------------------------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62992805(): Unit = {
    // case- 1
    val data1 =
      """
        |col1 | col2 | col3
        |u1   | w1   | v1
        |u2   | w2   | v2
        |u3   | w3   | v3
      """.stripMargin

    val stringDS1 = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()
    /**
      * +----+----+----+
      * |col1|col2|col3|
      * +----+----+----+
      * |u1  |w1  |v1  |
      * |u2  |w2  |v2  |
      * |u3  |w3  |v3  |
      * +----+----+----+
      *
      * root
      * |-- col1: string (nullable = true)
      * |-- col2: string (nullable = true)
      * |-- col3: string (nullable = true)
      */

    val case1 = Map("u1" -> Seq("w1","w11"), "u2" -> Seq("w2","w22"))

    val p1 = df1.withColumn("case1", typedLit(case1))
      .withColumn("col3",
        when(array_contains(expr("case1[col1]"), $"col2"), concat(lit("x-"), $"col3"))
          .otherwise($"col3")
      )
    p1.show(false)
    p1.printSchema()
    /**
      * +----+----+----+----------------------------------+
      * |col1|col2|col3|case1                             |
      * +----+----+----+----------------------------------+
      * |u1  |w1  |x-v1|[u1 -> [w1, w11], u2 -> [w2, w22]]|
      * |u2  |w2  |x-v2|[u1 -> [w1, w11], u2 -> [w2, w22]]|
      * |u3  |w3  |v3  |[u1 -> [w1, w11], u2 -> [w2, w22]]|
      * +----+----+----+----------------------------------+
      *
      * root
      * |-- col1: string (nullable = true)
      * |-- col2: string (nullable = true)
      * |-- col3: string (nullable = true)
      * |-- case1: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: array (valueContainsNull = true)
      * |    |    |-- element: string (containsNull = true)
      */


    // case-2
    val data2 =
      """
        |col1 | col2 | col3
        |u1   | 2    | v1
        |u1   | 6    | v11
        |u2   | 3    | v3
        |u3   | 4    | v3
      """.stripMargin
    val stringDS2 = data2.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    df2.show(false)
    df2.printSchema()
    /**
      * +----+----+----+
      * |col1|col2|col3|
      * +----+----+----+
      * |u1  |2   |v1  |
      * |u1  |6   |v11 |
      * |u2  |3   |v3  |
      * |u3  |4   |v3  |
      * +----+----+----+
      *
      * root
      * |-- col1: string (nullable = true)
      * |-- col2: integer (nullable = true)
      * |-- col3: string (nullable = true)
      */

    val case2 = Map("u1" -> (1,5), "u2" -> (2, 4))
    val p = df2.withColumn("case2", typedLit(case2))
      .withColumn("col3",
        when(expr("col2 between case2[col1]._1 and case2[col1]._2"), concat(lit("x-"), $"col3"))
          .otherwise($"col3")
      )
    p.show(false)
    p.printSchema()

    /**
      * +----+----+----+----------------------------+
      * |col1|col2|col3|case2                       |
      * +----+----+----+----------------------------+
      * |u1  |2   |x-v1|[u1 -> [1, 5], u2 -> [2, 4]]|
      * |u1  |6   |v11 |[u1 -> [1, 5], u2 -> [2, 4]]|
      * |u2  |3   |x-v3|[u1 -> [1, 5], u2 -> [2, 4]]|
      * |u3  |4   |v3  |[u1 -> [1, 5], u2 -> [2, 4]]|
      * +----+----+----+----------------------------+
      *
      * root
      * |-- col1: string (nullable = true)
      * |-- col2: integer (nullable = true)
      * |-- col3: string (nullable = true)
      * |-- case2: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: struct (valueContainsNull = true)
      * |    |    |-- _1: integer (nullable = false)
      * |    |    |-- _2: integer (nullable = false)
      */
  }

  // ############################################################################################################

  @Test
  def test63011603(): Unit = {
    val data =
      """
        |{
        |	"rates": {
        |		"2019-02-15": {
        |
        |			"AUD": 1.4996,
        |			"GBN": 8.9623,
        |			"BRL": 137.0
        |		},
        |
        |		"2019-01-02": {
        |			"AUD": 1.3996,
        |			"GBN": 8.6623,
        |			"BRL": 135.0
        |		},
        |
        |		"2019-01-03": {
        |			"AUD": 1.2996,
        |			"GBN": 8.7623,
        |			"BRL": 135.0
        |		},
        |
        |		"2019-01-04": {
        |			"AUD": 1.2996,
        |			"GBN": 8.8623,
        |			"BRL": 136.0
        |		}
        |	}
        |
        |}
      """.stripMargin
    val df = spark.read.json(Seq(data).toDS)
    // read using file by below command
//    spark.read.option("multiLine", true).json("path/to/json/file.json")
    df.show(false)
    df.printSchema()
    /**
      * +----------------------------------------------------------------------------------------------------+
      * |rates                                                                                               |
      * +----------------------------------------------------------------------------------------------------+
      * |[[1.3996, 135.0, 8.6623], [1.2996, 135.0, 8.7623], [1.2996, 136.0, 8.8623], [1.4996, 137.0, 8.9623]]|
      * +----------------------------------------------------------------------------------------------------+
      *
      * root
      * |-- rates: struct (nullable = true)
      * |    |-- 2019-01-02: struct (nullable = true)
      * |    |    |-- AUD: double (nullable = true)
      * |    |    |-- BRL: double (nullable = true)
      * |    |    |-- GBN: double (nullable = true)
      * |    |-- 2019-01-03: struct (nullable = true)
      * |    |    |-- AUD: double (nullable = true)
      * |    |    |-- BRL: double (nullable = true)
      * |    |    |-- GBN: double (nullable = true)
      * |    |-- 2019-01-04: struct (nullable = true)
      * |    |    |-- AUD: double (nullable = true)
      * |    |    |-- BRL: double (nullable = true)
      * |    |    |-- GBN: double (nullable = true)
      * |    |-- 2019-02-15: struct (nullable = true)
      * |    |    |-- AUD: double (nullable = true)
      * |    |    |-- BRL: double (nullable = true)
      * |    |    |-- GBN: double (nullable = true)
      */

    val df1 = df.selectExpr("rates.*")
    val stringCol = df1.columns.map(c => s"'$c', `$c`").mkString(", ")

    val processedDF = df1.selectExpr(s"stack(${df1.columns.length}, $stringCol) as (`yyyy-MM-dd`, value)")
      .filter(s"`yyyy-MM-dd` between '2019-01-02' and '2019-01-04'")
      .selectExpr("`yyyy-MM-dd`", "value.*")
    processedDF.show(false)
    processedDF.printSchema()

    /**
      * +----------+------+-----+------+
      * |yyyy-MM-dd|AUD   |BRL  |GBN   |
      * +----------+------+-----+------+
      * |2019-01-02|1.3996|135.0|8.6623|
      * |2019-01-03|1.2996|135.0|8.7623|
      * |2019-01-04|1.2996|136.0|8.8623|
      * +----------+------+-----+------+
      *
      * root
      * |-- yyyy-MM-dd: string (nullable = true)
      * |-- AUD: double (nullable = true)
      * |-- BRL: double (nullable = true)
      * |-- GBN: double (nullable = true)
      */
    // processedDF can be saved as below
//    processedDF.write
//      .option("header", true)
//      .mode(SaveMode.Overwrite)
//      .csv("/path/to/directory")

  }

  // ############################################################################################################

  @Test
  def test63027379(): Unit = {
    val data =
      """
        |Col1  Col2   Col3
        |True  False  False
        |True  True   True
        |False False  True
        |False False  False
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
      .map(_.split("\\s+").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    df2.show(false)
    df2.printSchema()
    /**
      * +-----+-----+-----+
      * |Col1 |Col2 |Col3 |
      * +-----+-----+-----+
      * |true |false|false|
      * |true |true |true |
      * |false|false|true |
      * |false|false|false|
      * +-----+-----+-----+
      *
      * root
      * |-- Col1: boolean (nullable = true)
      * |-- Col2: boolean (nullable = true)
      * |-- Col3: boolean (nullable = true)
      */

    val findCounts = df2.columns.flatMap(c => Seq(col(c), count(c).over(Window.partitionBy(c)).as(s"count_$c")))
    df2.select(findCounts: _*).distinct()
      .show(false)
    /**
      * +-----+----------+-----+----------+-----+----------+
      * |Col1 |count_Col1|Col2 |count_Col2|Col3 |count_Col3|
      * +-----+----------+-----+----------+-----+----------+
      * |false|2         |false|3         |false|2         |
      * |false|2         |false|3         |true |2         |
      * |true |2         |false|3         |false|2         |
      * |true |2         |true |1         |true |2         |
      * +-----+----------+-----+----------+-----+----------+
      */

    // Assuming all the columns in the dataframe have same distinct values
    val columns = df2.columns
    val head = columns.head
    val zeroDF = df2.groupBy(head).agg(count(head).as(s"${head}_count"))
    columns.tail.foldLeft(zeroDF){
      (df, c) => df.join(df2.groupBy(c).agg(count(c).as(s"${c}_count")), col(head) === col(c))
    }.show(false)

    /**
      * +-----+----------+-----+----------+-----+----------+
      * |Col1 |Col1_count|Col2 |Col2_count|Col3 |Col3_count|
      * +-----+----------+-----+----------+-----+----------+
      * |false|2         |false|3         |false|2         |
      * |true |2         |true |1         |true |2         |
      * +-----+----------+-----+----------+-----+----------+
      */
  }
  // ############################################################################################################

  @Test
  def test63030568(): Unit = {
    val data =
      """
        |year_week
        | 2019-W51
        | 2019-W52
        | 2020-W01
        | 2020-W02
        | 2020-W03
        | 2020-W04
        | 2020-W05
        | 2020-W06
        | 2020-W07
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    df2.show(false)
    df2.printSchema()
    /**
      * +---------+
      * |year_week|
      * +---------+
      * |2019-W51 |
      * |2019-W52 |
      * |2020-W01 |
      * |2020-W02 |
      * |2020-W03 |
      * |2020-W04 |
      * |2020-W05 |
      * |2020-W06 |
      * |2020-W07 |
      * +---------+
      *
      * root
      * |-- year_week: string (nullable = true)
      */
    // week starting from monday, concat "-1", for tuesday "-2" etc. => (1 = Monday, ..., 7 = Sunday)
    val p = df2.withColumn("week_start", to_date(concat($"year_week", lit("-1")), "YYYY-'W'ww-u"))
      .withColumn("week_end", next_day($"week_start", "SUN"))
    p.show(false)
    p.printSchema()

    /**
      * +---------+----------+----------+
      * |year_week|week_start|week_end  |
      * +---------+----------+----------+
      * |2019-W51 |2019-12-16|2019-12-22|
      * |2019-W52 |2019-12-23|2019-12-29|
      * |2020-W01 |2019-12-30|2020-01-05|
      * |2020-W02 |2020-01-06|2020-01-12|
      * |2020-W03 |2020-01-13|2020-01-19|
      * |2020-W04 |2020-01-20|2020-01-26|
      * |2020-W05 |2020-01-27|2020-02-02|
      * |2020-W06 |2020-02-03|2020-02-09|
      * |2020-W07 |2020-02-10|2020-02-16|
      * +---------+----------+----------+
      *
      * root
      * |-- year_week: string (nullable = true)
      * |-- week_start: date (nullable = true)
      * |-- week_end: date (nullable = true)
      */

//    ALternative-2
//    val p2 = df2.withColumn("date", to_date($"year_week", "YYYY-'W'ww"))
//      .withColumn("week_start", expr("date_sub(date, day_of_week-1)"))
//      .withColumn("week_end", expr("date_add(date, 7-day_of_week)"))
//    p2.show(false)
  }

  // ############################################################################################################

  @Test
  def test63034784(): Unit = {
    def checkIfDataIsInMemory(df: DataFrame): Boolean = {
      val manager = df.sparkSession.sharedState.cacheManager
      // step 1 - check if the dataframe.cache is issued earlier or not
      if (manager.lookupCachedData(df.queryExecution.logical).nonEmpty) {// cache statement was already issued
        println("Cache statement is already issued on this dataframe")
        // step-2 check if the data is in memory or not
        val cacheData = manager.lookupCachedData(df.queryExecution.logical).get
        cacheData.cachedRepresentation.cacheBuilder.isCachedColumnBuffersLoaded
      } else false
    }

    // test
    val df = spark.read
      .parquet(getClass.getResource("/parquet/plain/part-00000-4ece3595-e410-4301-aefd-431cd1debf91-c000.snappy" +
        ".parquet").getPath)
    println(checkIfDataIsInMemory(df))
    /**
      * false
      */

    df.cache()
    // check if the data is cached
    println(checkIfDataIsInMemory(df))
    /**
      * Cache statement is already issued on this dataframe
      * false
      */

    println(df.count())
    println(checkIfDataIsInMemory(df))

    /**
      * 1
      * Cache statement is already issued on this dataframe
      * true
      */

  }

  // ############################################################################################################

  @Test
  def test63052229(): Unit = {
    Seq("DEL_12345", "1234").toDF("phone_number")
      .withColumn("phone_number", regexp_replace($"phone_number", "^DEL_.*", null))
      .show(false)
  }

  // ############################################################################################################

  @Test
  def test63057336(): Unit = {
    val df = spark.range(2).withColumn("name", lit("foo"))
    df.show(false)
    df.printSchema()
    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |0  |foo |
      * |1  |foo |
      * +---+----+
      *
      * root
      * |-- id: long (nullable = false)
      * |-- name: string (nullable = false)
      */
    val emptyDF = spark.createDataFrame(spark.sparkContext.emptyRDD[Row],df.schema)
    emptyDF.show(false)

    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * +---+----+
      */

    emptyDF.unionByName(df)
      .show(false)
    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |0  |foo |
      * |1  |foo |
      * +---+----+
      */
  }

  // ############################################################################################################

  @Test
  def test63057443(): Unit = {
    val df = spark.range(2).withColumn("name", lit("foo"))
    df.show(false)
    df.printSchema()
    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |0  |foo |
      * |1  |foo |
      * +---+----+
      *
      * root
      * |-- id: long (nullable = false)
      * |-- name: string (nullable = false)
      */
    val df2 = df.filter("id=0")
    df.join(df2, df.columns.toSeq, "leftanti")
      .show(false)

    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |1  |foo |
      * +---+----+
      */

//    SO = 63065405
    val df1=df2
    df2.as("b")
      .join(df1.as("a"), $"a.senderId" === $"b.member_id" && $"a.datepartition".between(
        concat($"b.start_date",lit("-00")), concat($"b.end_date", lit("-00")))
      )
      .selectExpr("a.senderId",
        "b.company_id",
        "ROW_NUMBER() OVER(PARTITION BY a.senderId ORDER BY b.chron_rank) AS rnk")
  }
  // ############################################################################################################

  @Test
  def test63065203(): Unit = {
    val df = Seq(Seq((0, 0.0)), Seq((1, 2.2))).toDF("pricing_data")
    df.show(false)
    df.printSchema()

    /**
      * +------------+
      * |pricing_data|
      * +------------+
      * |[[0, 0.0]]  |
      * |[[1, 2.2]]  |
      * +------------+
      *
      * root
      * |-- pricing_data: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- _1: integer (nullable = false)
      * |    |    |-- _2: double (nullable = false)
      */

    // spark>=2.4
    df.withColumn("pricing_data", expr(
    "TRANSFORM(pricing_data, x -> if(x._1=0 and x._2=0.0, named_struct('_1', null, '_2', null), x))"
    ))
      .show(false)

    /**
      * +------------+
      * |pricing_data|
      * +------------+
      * |[[,]]       |
      * |[[1, 2.2]]  |
      * +------------+
      */

    // spark<2.4
    val dataType = df.schema("pricing_data").dataType
   val replace =  udf((arrayOfStruct: mutable.WrappedArray[Row]) => {
      arrayOfStruct.map(row => {
        val map = row.getValuesMap(row.schema.map(_.name))
        if(map("_1")==0 && map("_2") == 0.0) {
          Row.fromTuple((null, null))
        } else row
      })
    }, dataType)

    df.withColumn("pricing_data", replace($"pricing_data"))
        .show(false)

    /**
      * +------------+
      * |pricing_data|
      * +------------+
      * |[[,]]       |
      * |[[1, 2.2]]  |
      * +------------+
      */
  }

  // ############################################################################################################

  @Test
  def test63095738(): Unit = {
    val df = spark.sql("select map('a', 'b') as col1, map('c', cast(1 as int)) as col2, " +
      "map(1, cast(2.2 as double)) as col3")
    df.printSchema()
    df.show(false)
    /**
      * root
      * |-- col1: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: string (valueContainsNull = false)
      * |-- col2: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: integer (valueContainsNull = false)
      * |-- col3: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: double (valueContainsNull = false)
      *
      * +--------+--------+----------+
      * |col1    |col2    |col3      |
      * +--------+--------+----------+
      * |[a -> b]|[c -> 1]|[d -> 2.2]|
      * +--------+--------+----------+
      */

    val p = df.withColumn("new_col", map_concat($"col1", $"col2", $"col3"))
    p.printSchema()
    p.show(false)

    /**
      * root
      * |-- col1: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: string (valueContainsNull = false)
      * |-- col2: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: integer (valueContainsNull = false)
      * |-- col3: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: double (valueContainsNull = false)
      * |-- new_col: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: string (valueContainsNull = false)
      *
      * +--------+--------+----------+--------------------------+
      * |col1    |col2    |col3      |new_col                   |
      * +--------+--------+----------+--------------------------+
      * |[a -> b]|[c -> 1]|[d -> 2.2]|[a -> b, c -> 1, d -> 2.2]|
      * +--------+--------+----------+--------------------------+
      */

    p.selectExpr("new_col['a']", "new_col['c']", "new_col['d']").printSchema()

    val x = df.withColumn("x", struct($"col1", $"col2", $"col3"))
      x.printSchema()
    x.selectExpr("x.col1['a']", "x.col2['c']", "x.col3['d']").printSchema()

    /**
      * root
      * |-- col1: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: string (valueContainsNull = false)
      * |-- col2: map (nullable = false)
      * |    |-- key: string
      * |    |-- value: integer (valueContainsNull = false)
      * |-- col3: map (nullable = false)
      * |    |-- key: integer
      * |    |-- value: double (valueContainsNull = false)
      * |-- x: struct (nullable = false)
      * |    |-- col1: map (nullable = false)
      * |    |    |-- key: string
      * |    |    |-- value: string (valueContainsNull = false)
      * |    |-- col2: map (nullable = false)
      * |    |    |-- key: string
      * |    |    |-- value: integer (valueContainsNull = false)
      * |    |-- col3: map (nullable = false)
      * |    |    |-- key: integer
      * |    |    |-- value: double (valueContainsNull = false)
      *
      * root
      * |-- x.col1 AS `col1`[a]: string (nullable = true)
      * |-- x.col2 AS `col2`[c]: integer (nullable = true)
      * |-- x.col3 AS `col3`[CAST(d AS INT)]: double (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test63095958(): Unit = {
    val data =
      """
        |class|score
        |A|
        |A|46
        |A|
        |A|
        |A|35
        |A|
        |A|
        |A|
        |A|46
        |A|
        |A|
        |B|78
        |B|
        |B|
        |B|
        |B|
        |B|
        |B|56
        |B|
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
//      .option("nullValue", "null")
      .csv(stringDS2)
    df2.show(false)
    df2.printSchema()
    /**
      * +-----+-----+
      * |class|score|
      * +-----+-----+
      * |A    |null |
      * |A    |46   |
      * |A    |null |
      * |A    |null |
      * |A    |35   |
      * |A    |null |
      * |A    |null |
      * |A    |null |
      * |A    |46   |
      * |A    |null |
      * |A    |null |
      * |B    |78   |
      * |B    |null |
      * |B    |null |
      * |B    |null |
      * |B    |null |
      * |B    |null |
      * |B    |56   |
      * |B    |null |
      * +-----+-----+
      *
      * root
      * |-- class: string (nullable = true)
      * |-- score: integer (nullable = true)
      */

    val w1 = Window.partitionBy("class").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    val w2 = Window.partitionBy("class").rowsBetween(Window.currentRow, Window.unboundedFollowing)
    df2.withColumn("previous", last("score", ignoreNulls = true).over(w1))
      .withColumn("next", first("score", ignoreNulls = true).over(w2))
      .withColumn("new_score", (coalesce($"previous", $"next") + coalesce($"next", $"previous")) / 2)
      .drop("next", "previous")
      .show(false)

    /**
      * +-----+-----+---------+
      * |class|score|new_score|
      * +-----+-----+---------+
      * |A    |null |46.0     |
      * |A    |46   |46.0     |
      * |A    |null |40.5     |
      * |A    |null |40.5     |
      * |A    |35   |35.0     |
      * |A    |null |40.5     |
      * |A    |null |40.5     |
      * |A    |null |40.5     |
      * |A    |46   |46.0     |
      * |A    |null |46.0     |
      * |A    |null |46.0     |
      * |B    |78   |78.0     |
      * |B    |null |67.0     |
      * |B    |null |67.0     |
      * |B    |null |67.0     |
      * |B    |null |67.0     |
      * |B    |null |67.0     |
      * |B    |56   |56.0     |
      * |B    |null |56.0     |
      * +-----+-----+---------+
      */
  }

  // ############################################################################################################

  @Test
  def test63094883(): Unit = {
    val data =
      """
        |Col1 | Col2  | Col  | Col3
        |A    | 0.532 | 0.234 | 2020-01-01 05:00:00
        |B    | 0.242 | 0.224 | 2020-01-01 06:00:00
        |A    | 0.152 | 0.753 | 2020-01-01 08:00:00
        |C    | 0.149 | 0.983 | 2020-01-01 08:00:00
        |A    | 0.635 | 0.429 | 2020-01-01 09:00:00
        |A    | 0.938 | 0.365 | 2020-01-01 10:00:00
        |C    | 0.293 | 0.956 | 2020-01-02 05:00:00
        |A    | 0.294 | 0.234 | 2020-01-02 06:00:00
        |E    | 0.294 | 0.394 | 2020-01-02 07:00:00
        |D    | 0.294 | 0.258 | 2020-01-02 08:00:00
        |A    | 0.687 | 0.666 | 2020-01-03 05:00:00
        |C    | 0.232 | 0.494 | 2020-01-03 06:00:00
        |D    | 0.575 | 0.845 | 2020-01-03 07:00:00
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    df2.show(false)
    df2.printSchema()
    /**
      * +----+-----+-----+-------------------+
      * |Col1|Col2 |Col  |Col3               |
      * +----+-----+-----+-------------------+
      * |A   |0.532|0.234|2020-01-01 05:00:00|
      * |B   |0.242|0.224|2020-01-01 06:00:00|
      * |A   |0.152|0.753|2020-01-01 08:00:00|
      * |C   |0.149|0.983|2020-01-01 08:00:00|
      * |A   |0.635|0.429|2020-01-01 09:00:00|
      * |A   |0.938|0.365|2020-01-01 10:00:00|
      * |C   |0.293|0.956|2020-01-02 05:00:00|
      * |A   |0.294|0.234|2020-01-02 06:00:00|
      * |E   |0.294|0.394|2020-01-02 07:00:00|
      * |D   |0.294|0.258|2020-01-02 08:00:00|
      * |A   |0.687|0.666|2020-01-03 05:00:00|
      * |C   |0.232|0.494|2020-01-03 06:00:00|
      * |D   |0.575|0.845|2020-01-03 07:00:00|
      * +----+-----+-----+-------------------+
      *
      * root
      * |-- Col1: string (nullable = true)
      * |-- Col2: double (nullable = true)
      * |-- Col: double (nullable = true)
      * |-- Col3: timestamp (nullable = true)
      */

    val w = Window.partitionBy("Col1").orderBy("Col3_long").rangeBetween(-7200, Window.currentRow)
    df2.withColumn("Col3_long", $"Col3".cast("long"))
      .withColumn("new_col", when(sum($"Col2").over(w) - $"Col2" =!= lit(0), sum($"Col2").over(w) - $"Col2")
      )
      .show(false)
    /**
      * +----+-----+-----+-------------------+----------+-------------------+
      * |Col1|Col2 |Col  |Col3               |Col3_long |new_col            |
      * +----+-----+-----+-------------------+----------+-------------------+
      * |A   |0.532|0.234|2020-01-01 05:00:00|1577835000|null               |
      * |A   |0.152|0.753|2020-01-01 08:00:00|1577845800|null               |
      * |A   |0.635|0.429|2020-01-01 09:00:00|1577849400|0.15200000000000002|
      * |A   |0.938|0.365|2020-01-01 10:00:00|1577853000|0.7870000000000001 |
      * |A   |0.294|0.234|2020-01-02 06:00:00|1577925000|null               |
      * |A   |0.687|0.666|2020-01-03 05:00:00|1578007800|null               |
      * |D   |0.294|0.258|2020-01-02 08:00:00|1577932200|null               |
      * |D   |0.575|0.845|2020-01-03 07:00:00|1578015000|null               |
      * |B   |0.242|0.224|2020-01-01 06:00:00|1577838600|null               |
      * |C   |0.149|0.983|2020-01-01 08:00:00|1577845800|null               |
      * |C   |0.293|0.956|2020-01-02 05:00:00|1577921400|null               |
      * |C   |0.232|0.494|2020-01-03 06:00:00|1578011400|null               |
      * |E   |0.294|0.394|2020-01-02 07:00:00|1577928600|null               |
      * +----+-----+-----+-------------------+----------+-------------------+
      */

    df2.createOrReplaceTempView("table")
    spark.sql(
      """
        |select *,
        |case when (sum_Col2 - Col2 != 0) then sum_Col2 - Col2 end as real_sum
        |FROM (
        |   select *,
        |     sum(Col2) over (
        |       partition by Col1
        |       order by cast(Col3 as timestamp)
        |       range between interval 2 hours preceding and current row) as sum_Col2
        |   from table) a
      """.stripMargin)
      .show(false)

    /**
      * +----+-----+-----+-------------------+--------+-------------------+
      * |Col1|Col2 |Col  |Col3               |sum_Col2|real_sum           |
      * +----+-----+-----+-------------------+--------+-------------------+
      * |A   |0.532|0.234|2020-01-01 05:00:00|0.532   |null               |
      * |A   |0.152|0.753|2020-01-01 08:00:00|0.152   |null               |
      * |A   |0.635|0.429|2020-01-01 09:00:00|0.787   |0.15200000000000002|
      * |A   |0.938|0.365|2020-01-01 10:00:00|1.725   |0.7870000000000001 |
      * |A   |0.294|0.234|2020-01-02 06:00:00|0.294   |null               |
      * |A   |0.687|0.666|2020-01-03 05:00:00|0.687   |null               |
      * |D   |0.294|0.258|2020-01-02 08:00:00|0.294   |null               |
      * |D   |0.575|0.845|2020-01-03 07:00:00|0.575   |null               |
      * |B   |0.242|0.224|2020-01-01 06:00:00|0.242   |null               |
      * |C   |0.149|0.983|2020-01-01 08:00:00|0.149   |null               |
      * |C   |0.293|0.956|2020-01-02 05:00:00|0.293   |null               |
      * |C   |0.232|0.494|2020-01-03 06:00:00|0.232   |null               |
      * |E   |0.294|0.394|2020-01-02 07:00:00|0.294   |null               |
      * +----+-----+-----+-------------------+--------+-------------------+
      */

  }

  // ############################################################################################################

  @Test
  def test63138051(): Unit = {
    val df = Seq("[[,,hello,yes],[take,no,I,m],[hi,good,,]]").toDF("table")
    df.show(false)
    df.printSchema()
    /**
      * +-----------------------------------------+
      * |table                                    |
      * +-----------------------------------------+
      * |[[,,hello,yes],[take,no,I,m],[hi,good,,]]|
      * +-----------------------------------------+
      *
      * root
      * |-- table: string (nullable = true)
      */

    val  p = df.withColumn("arr", split(
      translate(
        regexp_replace($"table", """\]\s*,\s*\[""", "##"), "][", ""
      ), "##"
    ))

    val processed = p.withColumn("arr", expr("TRANSFORM(arr, x -> split(x, ','))"))

    processed.show(false)
    processed.printSchema()

    /**
      * +-----------------------------------------+----------------------------------------------------+
      * |table                                    |arr                                                 |
      * +-----------------------------------------+----------------------------------------------------+
      * |[[,,hello,yes],[take,no,I,m],[hi,good,,]]|[[, , hello, yes], [take, no, I, m], [hi, good, , ]]|
      * +-----------------------------------------+----------------------------------------------------+
      *
      * root
      * |-- table: string (nullable = true)
      * |-- arr: array (nullable = true)
      * |    |-- element: array (containsNull = true)
      * |    |    |-- element: string (containsNull = true)
      */
    df.withColumn("a", from_json($"table", "array<array<string>>", Map.empty[String, String]))
      .show(false)
  }
  // ############################################################################################################

  @Test
  def test63130450(): Unit = {
    val df = Seq(
      (10, 20, 30, 40, 50),
      (100, 200, 300, 400, 500),
      (111, 222, 333, 444, 555),
      (1123, 2123, 3123, 4123, 5123),
      (1321, 2321, 3321, 4321, 5321)
    ).toDF("col_1", "col_2", "col_3", "col_4", "col_5")

    val columnsToCalculate = Seq("col_2","col_3","col_4")

    import com.som.spark.shared.RichDataFrame.implicits._
    df.selectExpr(columnsToCalculate: _*)
      .statSummary("mean", "count", "25%", "75%", "90%")
      .show(false)

    /**
      * +-------+------+-----+---+----+----+
      * |columns|mean  |count|25%|75% |90% |
      * +-------+------+-----+---+----+----+
      * |col_2  |977.2 |5    |200|2123|2321|
      * |col_3  |1421.4|5    |300|3123|3321|
      * |col_4  |1865.6|5    |400|4123|4321|
      * +-------+------+-----+---+----+----+
      */
  }
  // ############################################################################################################

  @Test
  def test63137437(): Unit = {
    val data = List(
      ("20", "score", "school", "2018-03-31", 14 , 12),
      ("21", "score", "school", "2018-03-31", 13 , 13),
      ("22", "rate", "school", "2018-03-31", 11 , 14),
      ("21", "rate", "school", "2018-03-31", 13 , 12)
    )
    val df = data.toDF("id", "code", "entity", "date", "value1", "value2")
    df.show(false)
    /**
      * +---+-----+------+----------+------+------+
      * |id |code |entity|date      |value1|value2|
      * +---+-----+------+----------+------+------+
      * |20 |score|school|2018-03-31|14    |12    |
      * |21 |score|school|2018-03-31|13    |13    |
      * |22 |rate |school|2018-03-31|11    |14    |
      * |21 |rate |school|2018-03-31|13    |12    |
      * +---+-----+------+----------+------+------+
      */

    val rateDs = List(
      ("21","2018-01-31","2018-06-31", 12 ,"C"),
      ("21","2018-01-31","2018-06-31", 13 ,"D")
    ).toDF("id","start_date","end_date", "map_code","map_val")
    rateDs.show(false)
    /**
      * +---+----------+----------+--------+-------+
      * |id |start_date|end_date  |map_code|map_val|
      * +---+----------+----------+--------+-------+
      * |21 |2018-01-31|2018-06-31|12      |C      |
      * |21 |2018-01-31|2018-06-31|13      |D      |
      * +---+----------+----------+--------+-------+
      */

    val newRateDS = rateDs.withColumn("lookUpMap",
      map_from_entries(collect_list(struct(col("map_code"), col("map_val"))).over(Window.partitionBy("id")))
    )
    newRateDS.show(false)
    /**
      * +---+----------+----------+--------+-------+------------------+
      * |id |start_date|end_date  |map_code|map_val|lookUpMap         |
      * +---+----------+----------+--------+-------+------------------+
      * |21 |2018-01-31|2018-06-31|12      |C      |[12 -> C, 13 -> D]|
      * |21 |2018-01-31|2018-06-31|13      |D      |[12 -> C, 13 -> D]|
      * +---+----------+----------+--------+-------+------------------+
      */

    val  resultDs = df.filter(col("code").equalTo(lit("rate"))).join(broadcast(newRateDS) ,
      rateDs("id") === df("id") && df("date").between(rateDs("start_date"), rateDs("end_date"))
        //.and(rateDs.col("mapping_value").equalTo(df.col("mean")))
      , "left"
    )

    resultDs.withColumn("value1", expr("coalesce(lookUpMap[value1], value1)"))
      .withColumn("value2", expr("coalesce(lookUpMap[value2], value2)"))
      .show(false)

    /**
      * +---+----+------+----------+------+------+----+----------+----------+--------+-------+------------------+
      * |id |code|entity|date      |value1|value2|id  |start_date|end_date  |map_code|map_val|lookUpMap         |
      * +---+----+------+----------+------+------+----+----------+----------+--------+-------+------------------+
      * |22 |rate|school|2018-03-31|11    |14    |null|null      |null      |null    |null   |null              |
      * |21 |rate|school|2018-03-31|D     |C     |21  |2018-01-31|2018-06-31|13      |D      |[12 -> C, 13 -> D]|
      * |21 |rate|school|2018-03-31|D     |C     |21  |2018-01-31|2018-06-31|12      |C      |[12 -> C, 13 -> D]|
      * +---+----+------+----------+------+------+----+----------+----------+--------+-------+------------------+
      */
  }
  // ############################################################################################################

  @Test
  def test63151711(): Unit = {
    val df = spark.sql("select current_date() as record_date, '1' column1")
    df.show(false)
    /**
      * +-----------+-------+
      * |record_date|column1|
      * +-----------+-------+
      * |2020-07-29 |1      |
      * +-----------+-------+
      */

    df.createOrReplaceTempView("table1")
    spark.sql(
      """
        |select recorddate, count(*)
        |from( select record_date as recorddate, column1
        |      from table1
        |      where record_date >= date_sub(current_date(), 1)
        |    )t
        |group by recorddate
        |order by recorddate
        |
      """.stripMargin)
      .show(false)

    /**
      * +----------+--------+
      * |recorddate|count(1)|
      * +----------+--------+
      * |2020-07-29|1       |
      * +----------+--------+
      */
  }

  // ############################################################################################################

  @Test
  def test63156999(): Unit = {
    val df = spark.sql("select array(0, 1, 1, 0, 0, null) as bits1, array(1, 1, 1, 0, 1, null) as bits2")
    df.show(false)
    df.printSchema()

    /**
      * +----------------+----------------+
      * |bits1           |bits2           |
      * +----------------+----------------+
      * |[0, 1, 1, 0, 0,]|[1, 1, 1, 0, 1,]|
      * +----------------+----------------+
      *
      * root
      * |-- bits1: array (nullable = false)
      * |    |-- element: integer (containsNull = true)
      * |-- bits2: array (nullable = false)
      * |    |-- element: integer (containsNull = true)
      */

    df.withColumn("x", expr("aggregate(zip_with(bits1, bits2, (x, y) -> if(x=y, 1, 0)), 0, (acc, x) -> acc + x)"))
      .show(false)

    /**
      * +----------------+----------------+---+
      * |bits1           |bits2           |x  |
      * +----------------+----------------+---+
      * |[0, 1, 1, 0, 0,]|[1, 1, 1, 0, 1,]|3  |
      * +----------------+----------------+---+
      */
  }
  // ############################################################################################################

  @Test
  def test63156474(): Unit = {
    val df = spark.range(1, 10).toDF("col_a")
    val w = Window.orderBy("col_a")
    df.withColumn("sum_till_current_row", sum("col_a").over(w))
      .show(false)
  }

  // ############################################################################################################

  @Test
  def test63158474(): Unit = {
    val origStructType = new StructType().add("in1", LongType, nullable = true).add("in2", StringType, nullable = true)
    val newStructType = origStructType.add("in3", DecimalType(18,5), nullable = true).add("in4", StringType, nullable = true)
    val newColSchema = MapType(LongType, newStructType)

    val m = Map(101L->(101L,"val2"),102L->(102L,"val3"))
    val df = Seq((100L,m)).toDF("id","info")
    df.show(false)
    df.printSchema()
    val typeUDFNewRet = udf((col1: Map[Long,Row]) => {
      col1.mapValues(r => Row.merge(r, Row(null, ""))) //Forced to use null here for another issue
    }, newColSchema)
    spark.udf.register("typeUDFNewRet",typeUDFNewRet)
    df.registerTempTable("op1")
    val df2 = spark.sql("select id, typeUDFNewRet(info) from op1")

    df2.show(false)
    df2.printSchema()

    /**
      * +---+----------------------------------------------+
      * |id |UDF(info)                                     |
      * +---+----------------------------------------------+
      * |100|[101 -> [101, val2,, ], 102 -> [102, val3,, ]]|
      * +---+----------------------------------------------+
      *
      * root
      * |-- id: long (nullable = false)
      * |-- UDF(info): map (nullable = true)
      * |    |-- key: long
      * |    |-- value: struct (valueContainsNull = true)
      * |    |    |-- in1: long (nullable = true)
      * |    |    |-- in2: string (nullable = true)
      * |    |    |-- in3: decimal(18,5) (nullable = true)
      * |    |    |-- in4: string (nullable = true)
      */
  }
  // ############################################################################################################

  @Test
  def test63168767(): Unit = {
    val df = spark.sql("select 2.00000 as array_1, array(1.0, 2.0, 3.0, 4.0) as array_2")
    df.show(false)

    /**
      * +-------+--------------------+
      * |array_1|array_2             |
      * +-------+--------------------+
      * |2.00000|[1.0, 2.0, 3.0, 4.0]|
      * +-------+--------------------+
      */

    df.withColumn("abs_diff",
      expr("TRANSFORM(slice(array_2, 1, size(array_2)-1), x -> abs(x-array_1))"))
      .show(false)

    /**
      * +-------+--------------------+---------------------------+
      * |array_1|array_2             |abs_diff                   |
      * +-------+--------------------+---------------------------+
      * |2.00000|[1.0, 2.0, 3.0, 4.0]|[1.00000, 0.00000, 1.00000]|
      * +-------+--------------------+---------------------------+
      */
  }
  // ############################################################################################################

  @Test
  def test63168532(): Unit = {
    val data1 =
      """
        |user_id| movie_id|timestep
        |   100 |   1000  |20200728
        |   101 |   1001  |20200727
        |   101 |   1002  |20200726
      """.stripMargin

    val stringDS1 = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString("|"))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", "|")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()

    val data2 =
      """
        |movie_id,  title  ,         genre
        |   1000 ,Toy Story,Adventure|Animation|Children
        |   1001 , Jumanji ,Adventure|Children|Fantasy
        |   1002 , Iron Man,Action|Adventure|Sci-Fi
      """.stripMargin
    val stringDS2 = data2.split(System.lineSeparator())
      .map(_.split("\\,").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    df2.show(false)
    df2.printSchema()

    df1.join(df2, "movie_id")
      .withColumn("genre", explode(split(col("genre"), "[|]")))
      .groupBy("user_id")
      .pivot("genre")
      .count()
      .na.fill(0)
      .show(false)

    /**
      * +-------+------+---------+---------+--------+-------+------+
      * |user_id|Action|Adventure|Animation|Children|Fantasy|Sci-Fi|
      * +-------+------+---------+---------+--------+-------+------+
      * |101    |1     |2        |0        |1       |1      |1     |
      * |100    |0     |1        |1        |1       |0      |0     |
      * +-------+------+---------+---------+--------+-------+------+
      */
  }
  // ############################################################################################################

  @Test
  def test63186784(): Unit = {
    val df = spark.sql("select 'abcdef123456abc123' as col")
//    df.withColumn("new_col", translate())
  }

  // ############################################################################################################

  @Test
  def test63194147(): Unit = {
    val data = Seq(
      Row("CUST_2634",1),
      Row("CUST_85",1),
      Row("CUST_976",2),
      Row("CUST_3005",2),
      Row("CUST_1594",10),
      Row("CUST_519",10)

    )
    val schema = StructType(
      List(
        StructField("CUSTOMER", StringType, true),
        StructField("GOTRESPONSE",IntegerType , true)
      )
    )
    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(data),
      schema
    )

    val assembler = new VectorAssembler()
      .setInputCols(Array("GOTRESPONSE"))
      .setOutputCol("GOTRESPONSE_vec")
    val scaler = new MinMaxScaler()
      .setInputCol("GOTRESPONSE_vec")
      .setOutputCol("RESPONSE_RATE")
    val pipeline = new Pipeline().setStages(Array(assembler, scaler))
    pipeline.fit(df).transform(df)
//      .withColumn("RESPONSE_RATE", expr())
      .show(false)

    val spec = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    df.withColumn("RESPONSE_RATE", (col("GOTRESPONSE")-mean("GOTRESPONSE").over(spec)) /
    stddev("GOTRESPONSE").over(spec))
      .show(false)
  }

  // ############################################################################################################

  class Latest(val f: Row => String, val schema: StructType) extends Aggregator[Row, (String, Row), Row] {
    override def zero: (String, Row) = ("0000-00-00", null)
    override def reduce(b: (String, Row), a: Row): (String, Row) = merge(b, (f(a), a))
    override def merge(b1: (String, Row), b2: (String, Row)): (String, Row) = Seq(b1, b2).maxBy(_._1)
    override def finish(reduction: (String, Row)): Row = reduction._2

    override def bufferEncoder: Encoder[(String, Row)] = Encoders.tuple(Encoders.STRING, RowEncoder(schema))
    override def outputEncoder: Encoder[Row] = RowEncoder(schema)
  }
  @Test
  def test63206872(): Unit = {
    val df = Seq(
      ("ham", "2019-01-01", 3L, "Yah"),
      ("cheese", "2018-12-31", 4L, "Woo"),
      ("fish", "2019-01-02", 5L, "Hah"),
      ("grain", "2019-01-01", 6L, "Community"),
      ("grain", "2019-01-02", 7L, "Community"),
      ("ham", "2019-01-04", 3L, "jamón")
    ).toDF("Key", "Date", "Numeric", "Text")

    println("input data:")
    df.show(false)
    df.printSchema()

    println("running latest:")
    df.groupByKey(_.getString(0)).agg(new Latest(_.getString(1), df.schema).toColumn)
      .show(false)

    /**
      * +------+---------------------------------+
      * |value |Latest(org.apache.spark.sql.Row) |
      * +------+---------------------------------+
      * |ham   |[ham, 2019-01-04, 3, jamón]      |
      * |cheese|[cheese, 2018-12-31, 4, Woo]     |
      * |fish  |[fish, 2019-01-02, 5, Hah]       |
      * |grain |[grain, 2019-01-02, 7, Community]|
      * +------+---------------------------------+
      */

    df.groupBy("Key").agg(max(struct("Date", "Numeric", "Text", "key")))
      .show(false)

    /**
      * +------+-----------------------------------------------------------------------------------------------------------------------+
      * |Key   |max(named_struct(NamePlaceholder(), Date, NamePlaceholder(), Numeric, NamePlaceholder(), Text, NamePlaceholder(), key))|
      * +------+-----------------------------------------------------------------------------------------------------------------------+
      * |cheese|[2018-12-31, 4, Woo, cheese]                                                                                           |
      * |fish  |[2019-01-02, 5, Hah, fish]                                                                                             |
      * |grain |[2019-01-02, 7, Community, grain]                                                                                      |
      * |ham   |[2019-01-04, 3, jamón, ham]                                                                                            |
      * +------+-----------------------------------------------------------------------------------------------------------------------+
      */
  }

}

case class BestSellerRank(
                           Ranking: Integer,
                           Category: String
                         )
case class Data(i: Int)

case class invoiceColumns (firstname :String,lastname:String,fullname:String)
case class invoiceColumns2 (firstname :String,lastname:String,fullname:String) {
  def this(firstname :String,lastname:String) = {
    this(firstname, lastname, firstname + lastname)
  }
}