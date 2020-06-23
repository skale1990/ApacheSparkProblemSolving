package com.som.spark.learning

import java.lang.reflect.Method
import java.text.SimpleDateFormat
import java.time.format.DateTimeFormatterBuilder
import java.time.temporal.ChronoField
import java.util.{Collections, Locale}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkFiles
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.classification.RandomForestClassifier
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
import org.apache.spark.sql.catalyst.expressions.{GenericRowWithSchema, Rand}
import org.apache.spark.sql.catalyst.expressions.aggregate.ApproximatePercentile
import org.apache.spark.sql.catalyst.util.{ArrayBasedMapData, GenericArrayData, MapData}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{first, _}
import org.apache.spark.sql.types._
import org.json4s.JsonAST

import scala.collection.{JavaConverters, mutable}
import scala.util.Random

class ProblemSolverJun2020Test extends Serializable {

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

    df1.select($"key", split($"key", "\\.").as("x"))
      .withColumn("bankInfo",
        expr("named_struct('name', x[1], 'cust_id', x[2], 'branch', x[3], 'dist', x[4])"))
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

  // ############################################################################################################

  @Test
  def test62284374(): Unit = {
    val data = """[{"id":1,"name":"abc1"},{"id":2,"name":"abc2"},{"id":3,"name":"abc3"}]"""
    val df = spark.read.json(Seq(data).toDS())
    df.show(false)
    df.printSchema()

    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |1  |abc1|
      * |2  |abc2|
      * |3  |abc3|
      * +---+----+
      *
      * root
      * |-- id: long (nullable = true)
      * |-- name: string (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62283312(): Unit = {
    val rddOfJsonStrings: RDD[String] = spark.sparkContext.parallelize(Seq("""{"foo":1}"""))
    val classSelector: String = "Foo" // could be "Foo" or "Bar", or any other String value

    val ds = classSelector match {
      case foo if classOf[Foo].getSimpleName == foo =>
        val df: DataFrame = spark.read.json(rddOfJsonStrings)
        df.as[Foo]
      case bar if classOf[Bar].getSimpleName == bar =>
        val df: DataFrame = spark.read.json(rddOfJsonStrings)
        df.as[Bar]
      case _ => throw new UnsupportedOperationException
    }

    ds.show(false)

    /**
      * +---+
      * |foo|
      * +---+
      * |1  |
      * +---+
      */
  }

  // ############################################################################################################

  @Test
  def test62280681(): Unit = {
    val data1 =
      """
        |user_id | earnings | start_date | end_date
        |1       | 10     | 2020-06-01 | 2020-06-10
        |2       | 20     | 2020-06-01 | 2020-06-10
        |3       | 30     | 2020-06-01 | 2020-06-10
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
      * +-------+--------+-------------------+-------------------+
      * |user_id|earnings|start_date         |end_date           |
      * +-------+--------+-------------------+-------------------+
      * |1      |10      |2020-06-01 00:00:00|2020-06-10 00:00:00|
      * |2      |20      |2020-06-01 00:00:00|2020-06-10 00:00:00|
      * |3      |30      |2020-06-01 00:00:00|2020-06-10 00:00:00|
      * +-------+--------+-------------------+-------------------+
      *
      * root
      * |-- user_id: integer (nullable = true)
      * |-- earnings: integer (nullable = true)
      * |-- start_date: timestamp (nullable = true)
      * |-- end_date: timestamp (nullable = true)
      */

    val data2 =
      """
        |user_id | profit
        |1       | 100
        |2       | 200
        |5       | 500
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
      * +-------+------+
      * |user_id|profit|
      * +-------+------+
      * |1      |100   |
      * |2      |200   |
      * |5      |500   |
      * +-------+------+
      *
      * root
      * |-- user_id: integer (nullable = true)
      * |-- profit: integer (nullable = true)
      */

    df1.createOrReplaceTempView("prev_table")
    df2.createOrReplaceTempView("new_table")

   val processedDF = spark.sql(
      """
        |SELECT coalesce(p.user_id, n.user_id) as user_id,
        |       (coalesce(earnings,0) + coalesce(profit, 0)) as earnings,
        |        start_date,
        |        end_date
        |FROM prev_table p FULL OUTER JOIN new_table n ON p.user_id=n.user_id
      """.stripMargin)

     processedDF.orderBy("user_id").show(false)

    /**
      * +-------+--------+-------------------+-------------------+
      * |user_id|earnings|start_date         |end_date           |
      * +-------+--------+-------------------+-------------------+
      * |1      |110     |2020-06-01 00:00:00|2020-06-10 00:00:00|
      * |2      |220     |2020-06-01 00:00:00|2020-06-10 00:00:00|
      * |3      |30      |2020-06-01 00:00:00|2020-06-10 00:00:00|
      * |5      |500     |null               |null               |
      * +-------+--------+-------------------+-------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62289699(): Unit = {
    spark.range(2).withColumn("depdate",
      expr("date_add(to_date('1960-01-01'), id)")
    ).show(false)

    /**
      * +---+----------+
      * |id |depdate   |
      * +---+----------+
      * |0  |1960-01-01|
      * |1  |1960-01-02|
      * +---+----------+
      */
  }

  // ############################################################################################################

  @Test
  def test62292756(): Unit = {
    spark.sql(
      """
        |CREATE TABLE IF NOT EXISTS data_source_tab1 (col1 INT, p1 STRING, p2 STRING)
        |  USING PARQUET PARTITIONED BY (p1, p2)
      """.stripMargin).show(false)

    val table = spark.sql("select p2, col1 from values ('bob', 1), ('sam', 2), ('bob', 1) T(p2,col1)")
    table.createOrReplaceTempView("table")
    spark.sql(
      """
        |INSERT INTO data_source_tab1 PARTITION (p1 = 'part1', p2)
        |  SELECT p2, col1 FROM table order by col1
      """.stripMargin).explain(true)

    /**
      * == Optimized Logical Plan ==
      * InsertIntoHadoopFsRelationCommand file:/Users/sokale/alm_repo/stack-overflow/StackOverflowProblemSolving/spark-warehouse/data_source_tab1, Map(p1 -> part1), false, [p1#14, p2#13], Parquet, Map(path -> file:/Users/sokale/alm_repo/stack-overflow/StackOverflowProblemSolving/spark-warehouse/data_source_tab1), Append, CatalogTable(
      * Database: default
      * Table: data_source_tab1
      * Created Time: Wed Jun 10 11:25:12 IST 2020
      * Last Access: Thu Jan 01 05:29:59 IST 1970
      * Created By: Spark 2.4.5
      * Type: MANAGED
      * Provider: PARQUET
      * Location: file:/Users/sokale/alm_repo/stack-overflow/StackOverflowProblemSolving/spark-warehouse/data_source_tab1
      * Partition Provider: Catalog
      * Partition Columns: [`p1`, `p2`]
      * Schema: root
      * -- col1: integer (nullable = true)
      * -- p1: string (nullable = true)
      * -- p2: string (nullable = true)
      * ), org.apache.spark.sql.execution.datasources.CatalogFileIndex@bbb7b43b, [col1, p1, p2]
      * +- Project [cast(p2#1 as int) AS col1#12, part1 AS p1#14, cast(col1#2 as string) AS p2#13]
      * +- Sort [col1#2 ASC NULLS FIRST], true
      * +- LocalRelation [p2#1, col1#2]
      *
      * == Physical Plan ==
      * Execute InsertIntoHadoopFsRelationCommand InsertIntoHadoopFsRelationCommand file:/Users/sokale/alm_repo/stack-overflow/StackOverflowProblemSolving/spark-warehouse/data_source_tab1, Map(p1 -> part1), false, [p1#14, p2#13], Parquet, Map(path -> file:/Users/sokale/alm_repo/stack-overflow/StackOverflowProblemSolving/spark-warehouse/data_source_tab1), Append, CatalogTable(
      * Database: default
      * Table: data_source_tab1
      * Created Time: Wed Jun 10 11:25:12 IST 2020
      * Last Access: Thu Jan 01 05:29:59 IST 1970
      * Created By: Spark 2.4.5
      * Type: MANAGED
      * Provider: PARQUET
      * Location: file:/Users/sokale/alm_repo/stack-overflow/StackOverflowProblemSolving/spark-warehouse/data_source_tab1
      * Partition Provider: Catalog
      * Partition Columns: [`p1`, `p2`]
      * Schema: root
      * -- col1: integer (nullable = true)
      * -- p1: string (nullable = true)
      * -- p2: string (nullable = true)
      * ), org.apache.spark.sql.execution.datasources.CatalogFileIndex@bbb7b43b, [col1, p1, p2]
      * +- *(1) Project [cast(p2#1 as int) AS col1#12, part1 AS p1#14, cast(col1#2 as string) AS p2#13]
      * +- *(1) Sort [col1#2 ASC NULLS FIRST], true, 0
      * +- Exchange rangepartitioning(col1#2 ASC NULLS FIRST, 2)
      * +- LocalTableScan [p2#1, col1#2]
      */
  }

  // ############################################################################################################

  @Test
  def test62305713(): Unit = {
    /**
      * /Json_gzips
      * file content
      * spark-test-data1.json.gz
      * --------------------
      * {"id":1,"name":"abc1"}
      * {"id":2,"name":"abc2"}
      * {"id":3,"name":"abc3"}
      */
    /**
      * file content
      * spark-test-data2.json.gz
      * --------------------
      * {"id":1,"name":"abc1"}
      * {"id":2,"name":"abc2"}
      * {"id":3,"name":"abc3"}
      */
    val path = getClass.getResource("/Json_gzips").getPath
    spark.read.json(path).show(false)

    /**
      * +---+----+
      * |id |name|
      * +---+----+
      * |1  |abc1|
      * |2  |abc2|
      * |3  |abc3|
      * |1  |abc1|
      * |2  |abc2|
      * |3  |abc3|
      * +---+----+
      */
  }

  // ############################################################################################################

  @Test
  def test62306768(): Unit = {
    spark.createDataset(Seq(ExpenseEntry("John", "candy", 0.5),
      ExpenseEntry("Tia", "game", 0.25),
      ExpenseEntry("John", "candy", 0.15),
      ExpenseEntry("Tia", "candy", 0.55))
    ).groupBy("category", "name")
      .agg(sum("amount"))
      .show(false)

    /**
      * +--------+----+--------------------+
      * |category|name|sum(amount)         |
      * +--------+----+--------------------+
      * |candy   |John|0.650000000000000000|
      * |game    |Tia |0.250000000000000000|
      * |candy   |Tia |0.550000000000000000|
      * +--------+----+--------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62317454(): Unit = {
    val data =
      """
        |    source|live
        |      Ford|   Y
        |      Ford|   Y
        |  Caddilac|   Y
        |  Caddilac|   Y
        | Chevrolet|   Y
        | Chevrolet|   Y
        |     Skoda|   Y
        |     Skoda|   Y
        |      Fiat|   Y
        |      Fiat|   Y
        |Alfa Romeo|   Y
        |Alfa Romeo|   Y
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
      * +----------+----+
      * |source    |live|
      * +----------+----+
      * |Ford      |Y   |
      * |Ford      |Y   |
      * |Caddilac  |Y   |
      * |Caddilac  |Y   |
      * |Chevrolet |Y   |
      * |Chevrolet |Y   |
      * |Skoda     |Y   |
      * |Skoda     |Y   |
      * |Fiat      |Y   |
      * |Fiat      |Y   |
      * |Alfa Romeo|Y   |
      * |Alfa Romeo|Y   |
      * +----------+----+
      *
      * root
      * |-- source: string (nullable = true)
      * |-- live: string (nullable = true)
      */

    df.withColumn("category", expr("element_at(array('Business', 'Casual'), row_number() over(partition by source, " +
      "live order by source, live))"))
      .show(false)

    /**
      * +----------+----+--------+
      * |source    |live|category|
      * +----------+----+--------+
      * |Alfa Romeo|Y   |Business|
      * |Alfa Romeo|Y   |Casual  |
      * |Caddilac  |Y   |Business|
      * |Caddilac  |Y   |Casual  |
      * |Chevrolet |Y   |Business|
      * |Chevrolet |Y   |Casual  |
      * |Ford      |Y   |Business|
      * |Ford      |Y   |Casual  |
      * |Skoda     |Y   |Business|
      * |Skoda     |Y   |Casual  |
      * |Fiat      |Y   |Business|
      * |Fiat      |Y   |Casual  |
      * +----------+----+--------+
      */
  }

  // ############################################################################################################

  @Test
  def test62342328(): Unit = {
    val df = Seq("{1.33,0.567,1.897,0,0.78}").toDF("activity")
    df.show(false)
    df.printSchema()
    /**
      * +-------------------------+
      * |activity                 |
      * +-------------------------+
      * |{1.33,0.567,1.897,0,0.78}|
      * +-------------------------+
      *
      * root
      * |-- activity: string (nullable = true)
      */
    val processedDF = df.withColumn("activity",
      split(regexp_replace($"activity", "[^0-9.,]", ""), ",").cast("array<double>"))
    processedDF.show(false)
    processedDF.printSchema()

    /**
      * +-------------------------------+
      * |activity                       |
      * +-------------------------------+
      * |[1.33, 0.567, 1.897, 0.0, 0.78]|
      * +-------------------------------+
      *
      * root
      * |-- activity: array (nullable = true)
      * |    |-- element: double (containsNull = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62344243(): Unit = {
    val data =
      """
        |start_date| end_date |sales_target
        |2020-01-01|2020-12-31|          15
        |2020-04-01|2020-12-31|          11
        |2020-07-01|2020-12-31|           3
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
      * +-------------------+-------------------+------------+
      * |start_date         |end_date           |sales_target|
      * +-------------------+-------------------+------------+
      * |2020-01-01 00:00:00|2020-12-31 00:00:00|15          |
      * |2020-04-01 00:00:00|2020-12-31 00:00:00|11          |
      * |2020-07-01 00:00:00|2020-12-31 00:00:00|3           |
      * +-------------------+-------------------+------------+
      *
      * root
      * |-- start_date: timestamp (nullable = true)
      * |-- end_date: timestamp (nullable = true)
      * |-- sales_target: integer (nullable = true)
      */

      val processedDF = df.withColumn("new_start_date", explode(sequence(to_date($"start_date"), to_date($"end_date"),
        expr("interval 3 month"))))
        .withColumn("new_end_date",
          date_sub(coalesce(lead("new_start_date", 1)
            .over(Window.partitionBy("start_date").orderBy("new_start_date")), to_date($"end_date")), 1)
        )

      processedDF.orderBy("start_date", "new_start_date").show(false)
      processedDF.printSchema()

      /**
        * +-------------------+-------------------+------------+--------------+------------+
        * |start_date         |end_date           |sales_target|new_start_date|new_end_date|
        * +-------------------+-------------------+------------+--------------+------------+
        * |2020-01-01 00:00:00|2020-12-31 00:00:00|15          |2020-01-01    |2020-03-31  |
        * |2020-01-01 00:00:00|2020-12-31 00:00:00|15          |2020-04-01    |2020-06-30  |
        * |2020-01-01 00:00:00|2020-12-31 00:00:00|15          |2020-07-01    |2020-09-30  |
        * |2020-01-01 00:00:00|2020-12-31 00:00:00|15          |2020-10-01    |2020-12-30  |
        * |2020-04-01 00:00:00|2020-12-31 00:00:00|11          |2020-04-01    |2020-06-30  |
        * |2020-04-01 00:00:00|2020-12-31 00:00:00|11          |2020-07-01    |2020-09-30  |
        * |2020-04-01 00:00:00|2020-12-31 00:00:00|11          |2020-10-01    |2020-12-30  |
        * |2020-07-01 00:00:00|2020-12-31 00:00:00|3           |2020-07-01    |2020-09-30  |
        * |2020-07-01 00:00:00|2020-12-31 00:00:00|3           |2020-10-01    |2020-12-30  |
        * +-------------------+-------------------+------------+--------------+------------+
        *
        * root
        * |-- start_date: timestamp (nullable = true)
        * |-- end_date: timestamp (nullable = true)
        * |-- sales_target: integer (nullable = true)
        * |-- new_start_date: date (nullable = false)
        * |-- new_end_date: date (nullable = true)
        */
  }

  // ############################################################################################################

  @Test
  def test62349251(): Unit = {
    val data =
      """
        |transaction_status|amount|category|email_id      |unique_id|acct_no|ciskey
        |posted            |116.26|Family  |abcd@gmail.com|12345678 |51663  |47626220
        |posted            |116.26|Family  |abcd@gmail.com|12345678 |51663  |47626221
        |posted            |116.26|Family  |abcd@gmail.com|12345678 |51663  |47626222
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
      * +------------------+------+--------+--------------+---------+-------+--------+
      * |transaction_status|amount|category|email_id      |unique_id|acct_no|ciskey  |
      * +------------------+------+--------+--------------+---------+-------+--------+
      * |posted            |116.26|Family  |abcd@gmail.com|12345678 |51663  |47626220|
      * |posted            |116.26|Family  |abcd@gmail.com|12345678 |51663  |47626221|
      * |posted            |116.26|Family  |abcd@gmail.com|12345678 |51663  |47626222|
      * +------------------+------+--------+--------------+---------+-------+--------+
      *
      * root
      * |-- transaction_status: string (nullable = true)
      * |-- amount: double (nullable = true)
      * |-- category: string (nullable = true)
      * |-- email_id: string (nullable = true)
      * |-- unique_id: integer (nullable = true)
      * |-- acct_no: integer (nullable = true)
      * |-- ciskey: integer (nullable = true)
      */

    val groupBy = df.columns.filter(_!="ciskey")
    df.groupBy(groupBy.map(col): _*).agg(collect_list($"ciskey").as("accounts"))
      .withColumn("ciskey", element_at($"accounts", 1) )
      .withColumn("customers", expr("TRANSFORM(accounts, " +
        "x -> named_struct('ciskey_no', x, 'ciskey_val', 'IND'))"))
      .withColumn("accounts",
        struct($"acct_no", $"customers"))
      .drop("customers")
      .toJSON
      .show(false)

    /**
      * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |value                                                                                                                                                                                                                                                                                                                          |
      * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |{"transaction_status":"posted","amount":116.26,"category":"Family","email_id":"abcd@gmail.com","unique_id":12345678,"acct_no":51663,"accounts":{"acct_no":51663,"customers":[{"ciskey_no":47626220,"ciskey_val":"IND"},{"ciskey_no":47626221,"ciskey_val":"IND"},{"ciskey_no":47626222,"ciskey_val":"IND"}]},"ciskey":47626220}|
      * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      */
  }
  // ############################################################################################################

  @Test
  def test62356092(): Unit = {

    val getRandom = udf((seed: Long) => new Random(seed).shuffle(seed.toString.toCharArray.iterator).mkString("").toLong)
    Seq(123456, 234567, 345678, 123456, 456789, 123456, 123456, 123456).toDF("phone_number")
      .withColumn("rand_number", getRandom($"phone_number"))
      .show(false)

    /**
      * +------------+-----------+
      * |phone_number|rand_number|
      * +------------+-----------+
      * |123456      |512634     |
      * |234567      |372456     |
      * |345678      |847635     |
      * |123456      |512634     |
      * |456789      |789546     |
      * |123456      |512634     |
      * |123456      |512634     |
      * |123456      |512634     |
      * +------------+-----------+
      */

    val getRandom1 = udf((seed: Long) => new Random(seed).shuffle(seed.toString.toCharArray.iterator).mkString(""))
    Seq(101010, 202230, 30145, 101000).toDF("phone_number")
      .withColumn("rand_number", getRandom1($"phone_number"))
      .show(false)

  }

  // ############################################################################################################

  @Test
  def test62357799(): Unit = {
    val df0 = Seq(
      ("78aa", "2020-04-14", "2020-04-14 19:00:00", "2020-04-14 19:23:59"),
      ("78aa", "2020-04-14", "2020-04-14 19:24:00", "2020-04-14 19:26:59"),
      ("78aa", "2020-04-14", "2020-04-14 19:27:00", "2020-04-14 19:35:59"),
      ("78aa", "2020-04-14", "2020-04-14 19:36:00", "2020-04-14 19:55:00"),
      ("25aa", "2020-04-15", "2020-04-15 08:00:00", "2020-04-15 08:02:59"),
      ("25aa", "2020-04-15", "2020-04-15 11:03:00", "2020-04-15 11:11:59"),
      ("25aa", "2020-04-15", "2020-04-15 11:12:00", "2020-04-15 11:45:59"),
      ("25aa", "2020-04-15", "2020-04-15 11:46:00", "2020-04-15 11:47:00")
    ).toDF("id", "date", "start_time", "end_time")

    df0.withColumn("minutes",
      explode(sequence($"start_time".cast("timestamp"), $"end_time".cast("timestamp"), expr("interval 1 minute"))))
      .show(false)

    /**
      * +----+----------+-------------------+-------------------+-------------------+
      * |id  |date      |start_time         |end_time           |minutes            |
      * +----+----------+-------------------+-------------------+-------------------+
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:00:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:01:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:02:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:03:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:04:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:05:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:06:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:07:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:08:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:09:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:10:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:11:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:12:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:13:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:14:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:15:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:16:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:17:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:18:00|
      * |78aa|2020-04-14|2020-04-14 19:00:00|2020-04-14 19:23:59|2020-04-14 19:19:00|
      * +----+----------+-------------------+-------------------+-------------------+
      * only showing top 20 rows
      */
  }

  // ############################################################################################################

  @Test
  def test62366103(): Unit = {
    val spark_df = Seq((1, 7, "foo"),
      (2, 6, "bar"),
      (3, 4, "foo"),
      (4, 8, "bar"),
      (5, 1, "bar")
    ).toDF("v1", "v2", "id")
    spark_df.show(false)
    spark_df.printSchema()
    spark_df.summary() // default= "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
      .show(false)

    /**
      * +---+---+---+
      * |v1 |v2 |id |
      * +---+---+---+
      * |1  |7  |foo|
      * |2  |6  |bar|
      * |3  |4  |foo|
      * |4  |8  |bar|
      * |5  |1  |bar|
      * +---+---+---+
      *
      * root
      * |-- v1: integer (nullable = false)
      * |-- v2: integer (nullable = false)
      * |-- id: string (nullable = true)
      *
      * +-------+------------------+------------------+----+
      * |summary|v1                |v2                |id  |
      * +-------+------------------+------------------+----+
      * |count  |5                 |5                 |5   |
      * |mean   |3.0               |5.2               |null|
      * |stddev |1.5811388300841898|2.7748873851023217|null|
      * |min    |1                 |1                 |bar |
      * |25%    |2                 |4                 |null|
      * |50%    |3                 |6                 |null|
      * |75%    |4                 |7                 |null|
      * |max    |5                 |8                 |foo |
      * +-------+------------------+------------------+----+
      */

  }

  // ############################################################################################################

  @Test
  def test62368565(): Unit = {
    val data =
      """
        |id |date_1    |date_2
        |0  |2017-01-21|2017-04-01
        |1  |2017-01-22|2017-04-24
        |2  |2017-02-23|2017-04-30
        |3  |2017-02-27|2017-04-30
        |4  |2017-04-23|2017-05-27
        |5  |2017-04-29|2017-06-30
        |6  |2017-06-13|2017-07-05
        |7  |2017-06-13|2017-07-18
        |8  |2017-06-16|2017-07-19
        |9  |2017-07-09|2017-08-02
        |10 |2017-07-18|2017-08-07
        |11 |2017-07-28|2017-08-11
        |12 |2017-07-28|2017-08-13
        |13 |2017-08-04|2017-08-13
        |14 |2017-08-13|2017-08-13
        |15 |2017-08-13|2017-08-13
        |16 |2017-08-13|2017-08-25
        |17 |2017-08-13|2017-09-10
        |18 |2017-08-31|2017-09-21
        |19 |2017-10-03|2017-09-22
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
      * +---+-------------------+-------------------+
      * |id |date_1             |date_2             |
      * +---+-------------------+-------------------+
      * |0  |2017-01-21 00:00:00|2017-04-01 00:00:00|
      * |1  |2017-01-22 00:00:00|2017-04-24 00:00:00|
      * |2  |2017-02-23 00:00:00|2017-04-30 00:00:00|
      * |3  |2017-02-27 00:00:00|2017-04-30 00:00:00|
      * |4  |2017-04-23 00:00:00|2017-05-27 00:00:00|
      * |5  |2017-04-29 00:00:00|2017-06-30 00:00:00|
      * |6  |2017-06-13 00:00:00|2017-07-05 00:00:00|
      * |7  |2017-06-13 00:00:00|2017-07-18 00:00:00|
      * |8  |2017-06-16 00:00:00|2017-07-19 00:00:00|
      * |9  |2017-07-09 00:00:00|2017-08-02 00:00:00|
      * |10 |2017-07-18 00:00:00|2017-08-07 00:00:00|
      * |11 |2017-07-28 00:00:00|2017-08-11 00:00:00|
      * |12 |2017-07-28 00:00:00|2017-08-13 00:00:00|
      * |13 |2017-08-04 00:00:00|2017-08-13 00:00:00|
      * |14 |2017-08-13 00:00:00|2017-08-13 00:00:00|
      * |15 |2017-08-13 00:00:00|2017-08-13 00:00:00|
      * |16 |2017-08-13 00:00:00|2017-08-25 00:00:00|
      * |17 |2017-08-13 00:00:00|2017-09-10 00:00:00|
      * |18 |2017-08-31 00:00:00|2017-09-21 00:00:00|
      * |19 |2017-10-03 00:00:00|2017-09-22 00:00:00|
      * +---+-------------------+-------------------+
      *
      * root
      * |-- id: integer (nullable = true)
      * |-- date_1: timestamp (nullable = true)
      * |-- date_2: timestamp (nullable = true)
      */

    // week
    val weekDiff = 7
    val w = Window.orderBy("id", "date_1", "date_2")
      .rangeBetween(Window.currentRow, Window.unboundedFollowing)
    df.withColumn("count", sum(
      when(datediff($"date_1", $"date_2") <= weekDiff, 1).otherwise(0)
    ).over(w))
      .orderBy("id")
      .show(false)

    /**
      * +---+-------------------+-------------------+-----+
      * |id |date_1             |date_2             |count|
      * +---+-------------------+-------------------+-----+
      * |0  |2017-01-21 00:00:00|2017-04-01 00:00:00|19   |
      * |1  |2017-01-22 00:00:00|2017-04-24 00:00:00|18   |
      * |2  |2017-02-23 00:00:00|2017-04-30 00:00:00|17   |
      * |3  |2017-02-27 00:00:00|2017-04-30 00:00:00|16   |
      * |4  |2017-04-23 00:00:00|2017-05-27 00:00:00|15   |
      * |5  |2017-04-29 00:00:00|2017-06-30 00:00:00|14   |
      * |6  |2017-06-13 00:00:00|2017-07-05 00:00:00|13   |
      * |7  |2017-06-13 00:00:00|2017-07-18 00:00:00|12   |
      * |8  |2017-06-16 00:00:00|2017-07-19 00:00:00|11   |
      * |9  |2017-07-09 00:00:00|2017-08-02 00:00:00|10   |
      * |10 |2017-07-18 00:00:00|2017-08-07 00:00:00|9    |
      * |11 |2017-07-28 00:00:00|2017-08-11 00:00:00|8    |
      * |12 |2017-07-28 00:00:00|2017-08-13 00:00:00|7    |
      * |13 |2017-08-04 00:00:00|2017-08-13 00:00:00|6    |
      * |14 |2017-08-13 00:00:00|2017-08-13 00:00:00|5    |
      * |15 |2017-08-13 00:00:00|2017-08-13 00:00:00|4    |
      * |16 |2017-08-13 00:00:00|2017-08-25 00:00:00|3    |
      * |17 |2017-08-13 00:00:00|2017-09-10 00:00:00|2    |
      * |18 |2017-08-31 00:00:00|2017-09-21 00:00:00|1    |
      * |19 |2017-10-03 00:00:00|2017-09-22 00:00:00|0    |
      * +---+-------------------+-------------------+-----+
      */
  }

  // ############################################################################################################

  @Test
  def test62369670(): Unit = {
    val data =
      """
        |date,channel,ticket_qty,ticket_amount
        |20011231,passenger,500,2500
        |20011231,agent,100,1100
        |20020101,passenger,450,2000
        |20020101,agent,120,1500
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      //      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
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
      * +--------+---------+----------+-------------+
      * |date    |channel  |ticket_qty|ticket_amount|
      * +--------+---------+----------+-------------+
      * |20011231|passenger|500       |2500         |
      * |20011231|agent    |100       |1100         |
      * |20020101|passenger|450       |2000         |
      * |20020101|agent    |120       |1500         |
      * +--------+---------+----------+-------------+
      *
      * root
      * |-- date: integer (nullable = true)
      * |-- channel: string (nullable = true)
      * |-- ticket_qty: integer (nullable = true)
      * |-- ticket_amount: integer (nullable = true)
      */
    df.groupBy("date")
      .pivot("channel")
      .agg(
        first("ticket_qty").as("ticket_qty"),
        first("ticket_amount").as("ticket_amount")
      ).show(false)

    /**
      * +--------+----------------+-------------------+--------------------+-----------------------+
      * |date    |agent_ticket_qty|agent_ticket_amount|passenger_ticket_qty|passenger_ticket_amount|
      * +--------+----------------+-------------------+--------------------+-----------------------+
      * |20011231|100             |1100               |500                 |2500                   |
      * |20020101|120             |1500               |450                 |2000                   |
      * +--------+----------------+-------------------+--------------------+-----------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62383830(): Unit = {
    val cols = Seq("ID", "1_count", "2_count", "3_count", "4_count", "1_per", "2_per", "3_per", "4_per")
    val df1 = Seq(
      (1, 3, 11, 15, 3, 70, 80, 150, 20),
      (2, 19, 5, 15, 3, 150, 80, 200, 43),
      (3, 30, 15, 15, 39, 55, 80, 150, 200),
      (4, 8, 65, 3, 3, 70, 80, 150, 55)
    ) toDF (cols: _*)
    df1.show(false)
    df1.printSchema()
    /**
      * +---+-------+-------+-------+-------+-----+-----+-----+-----+
      * |ID |1_count|2_count|3_count|4_count|1_per|2_per|3_per|4_per|
      * +---+-------+-------+-------+-------+-----+-----+-----+-----+
      * |1  |3      |11     |15     |3      |70   |80   |150  |20   |
      * |2  |19     |5      |15     |3      |150  |80   |200  |43   |
      * |3  |30     |15     |15     |39     |55   |80   |150  |200  |
      * |4  |8      |65     |3      |3      |70   |80   |150  |55   |
      * +---+-------+-------+-------+-------+-----+-----+-----+-----+
      *
      * root
      * |-- ID: integer (nullable = false)
      * |-- 1_count: integer (nullable = false)
      * |-- 2_count: integer (nullable = false)
      * |-- 3_count: integer (nullable = false)
      * |-- 4_count: integer (nullable = false)
      * |-- 1_per: integer (nullable = false)
      * |-- 2_per: integer (nullable = false)
      * |-- 3_per: integer (nullable = false)
      * |-- 4_per: integer (nullable = false)
      */

    val (countCols, perCols) = cols.filter(_ != "ID").partition(_.endsWith("count"))
    val struct = countCols.zip(perCols).map { case (countCol, perCol) =>
      expr(s"named_struct('count', $countCol, 'per', $perCol, " +
        s"'slot', cast(substring_index('$countCol', '_', 1) as int))")
    }

    val processedDf = df1.select($"ID", array(struct: _*).as("count_per"))
        .withColumn("count_per_p", coalesce(
          expr("FILTER(count_per, x -> x.count > 10 and x.per < 100)[0]"),
          expr("named_struct('count', 0,'per', 0, 'slot', 0)")
        ))
        .selectExpr("ID", "count_per_p.*")
    processedDf.show(false)
    processedDf.printSchema()

    /**
      * +---+-----+---+----+
      * |ID |count|per|slot|
      * +---+-----+---+----+
      * |1  |11   |80 |2   |
      * |2  |0    |0  |0   |
      * |3  |30   |55 |1   |
      * |4  |65   |80 |2   |
      * +---+-----+---+----+
      *
      * root
      * |-- ID: integer (nullable = false)
      * |-- count: integer (nullable = false)
      * |-- per: integer (nullable = false)
      * |-- slot: integer (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62385976(): Unit = {
    /** Input data */
    val inputDf = Seq(
      (1, "Question1Text", "Yes", "abcde1", 0, List("x1", "y1")),
      (2, "Question2Text", "No", "abcde1", 0, List("x1", "y1")),
      (3, "Question3Text", "3", "abcde1", 0, List("x1", "y1")),
      (1, "Question1Text", "No", "abcde2", 0, List("x2", "y2")),
      (2, "Question2Text", "Yes", "abcde2", 0, List("x2", "y2"))
    ).toDF("Qid", "Question", "AnswerText", "ParticipantID", "Assessment", "GeoTag")
    println("Input:")
    inputDf.show(false)
    inputDf.printSchema()

    /**
      * Input:
      * +---+-------------+----------+-------------+----------+--------+
      * |Qid|Question     |AnswerText|ParticipantID|Assessment|GeoTag  |
      * +---+-------------+----------+-------------+----------+--------+
      * |1  |Question1Text|Yes       |abcde1       |0         |[x1, y1]|
      * |2  |Question2Text|No        |abcde1       |0         |[x1, y1]|
      * |3  |Question3Text|3         |abcde1       |0         |[x1, y1]|
      * |1  |Question1Text|No        |abcde2       |0         |[x2, y2]|
      * |2  |Question2Text|Yes       |abcde2       |0         |[x2, y2]|
      * +---+-------------+----------+-------------+----------+--------+
      *
      * root
      * |-- Qid: integer (nullable = false)
      * |-- Question: string (nullable = true)
      * |-- AnswerText: string (nullable = true)
      * |-- ParticipantID: string (nullable = true)
      * |-- Assessment: integer (nullable = false)
      * |-- GeoTag: array (nullable = true)
      * |    |-- element: string (containsNull = true)
      */

    inputDf
      .groupBy($"ParticipantID", $"Assessment", $"GeoTag")
      .pivot("Qid")
      .agg(
        first($"AnswerText").as("Q")
      )
      .orderBy($"ParticipantID")
      .show(false)

    /**
      * +-------------+----------+--------+---+---+----+
      * |ParticipantID|Assessment|GeoTag  |1  |2  |3   |
      * +-------------+----------+--------+---+---+----+
      * |abcde1       |0         |[x1, y1]|Yes|No |3   |
      * |abcde2       |0         |[x2, y2]|No |Yes|null|
      * +-------------+----------+--------+---+---+----+
      */
  }

  // ############################################################################################################

  @Test
  def test62391444(): Unit = {
    val data =
      """
        |cod_cli|article_name|rank
        |123    |art_1       |1
        |123    |art_2       |2
        |123    |art_3       |3
        |456    |art_4       |1
        |456    |art_5       |2
        |456    |art_6       |3
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
      * +-------+------------+----+
      * |cod_cli|article_name|rank|
      * +-------+------------+----+
      * |123    |art_1       |1   |
      * |123    |art_2       |2   |
      * |123    |art_3       |3   |
      * |456    |art_4       |1   |
      * |456    |art_5       |2   |
      * |456    |art_6       |3   |
      * +-------+------------+----+
      *
      * root
      * |-- cod_cli: integer (nullable = true)
      * |-- article_name: string (nullable = true)
      * |-- rank: integer (nullable = true)
      */

    df.groupBy("cod_cli")
      .pivot("rank")
      .agg(first("article_name"))
      .select($"cod_cli", $"1".as("Product 1"), $"2".as("Product 2"), $"3".as("Product 3"))
      .withColumn("Product 1", to_json(expr("named_struct('cod_art', `Product 1`, 'rank', '1')")))
      .withColumn("Product 2", to_json(expr("named_struct('cod_art', `Product 2`, 'rank', '2')")))
      .withColumn("Product 3", to_json(expr("named_struct('cod_art', `Product 3`, 'rank', '3')")))
      .show(false)

    /**
      * +-------+------------------------------+------------------------------+------------------------------+
      * |cod_cli|Product 1                     |Product 2                     |Product 3                     |
      * +-------+------------------------------+------------------------------+------------------------------+
      * |123    |{"cod_art":"art_1","rank":"1"}|{"cod_art":"art_2","rank":"2"}|{"cod_art":"art_3","rank":"3"}|
      * |456    |{"cod_art":"art_4","rank":"1"}|{"cod_art":"art_5","rank":"2"}|{"cod_art":"art_6","rank":"3"}|
      * +-------+------------------------------+------------------------------+------------------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62389969(): Unit = {
    val inputDf = Seq(
      ("Warsaw", "Poland", "1 764 615"),
      ("Cracow", "Poland", "769 498"),
      ("Paris", "France", "2 206 488"),
      ("Villeneuve-Loubet", "France", "15 020"),
      ("Pittsburgh PA", "United States", "302 407"),
      ("Chicago IL", "United States", "2 716 000"),
      ("Milwaukee WI", "United States", "595 351"),
      ("Vilnius", "Lithuania", "580 020"),
      ("Stockholm", "Sweden", "972 647"),
      ("Goteborg", "Sweden", "580 020")
    ).toDF("name", "country", "population")
    println("Input:")
    inputDf.show(false)
    /**
      * Input:
      * +-----------------+-------------+----------+
      * |name             |country      |population|
      * +-----------------+-------------+----------+
      * |Warsaw           |Poland       |1 764 615 |
      * |Cracow           |Poland       |769 498   |
      * |Paris            |France       |2 206 488 |
      * |Villeneuve-Loubet|France       |15 020    |
      * |Pittsburgh PA    |United States|302 407   |
      * |Chicago IL       |United States|2 716 000 |
      * |Milwaukee WI     |United States|595 351   |
      * |Vilnius          |Lithuania    |580 020   |
      * |Stockholm        |Sweden       |972 647   |
      * |Goteborg         |Sweden       |580 020   |
      * +-----------------+-------------+----------+
      */

    val topPopulation = inputDf
      .withColumn("population", regexp_replace($"population", " ", "").cast("Integer"))
      .withColumn("population_name", struct($"population", $"name"))
      .groupBy("country")
      .agg(max("population_name").as("population_name"))
      .selectExpr("country", "population_name.*")
    topPopulation.show(false)
    topPopulation.printSchema()

    /**
      * +-------------+----------+----------+
      * |country      |population|name      |
      * +-------------+----------+----------+
      * |France       |2206488   |Paris     |
      * |Poland       |1764615   |Warsaw    |
      * |Lithuania    |580020    |Vilnius   |
      * |Sweden       |972647    |Stockholm |
      * |United States|2716000   |Chicago IL|
      * +-------------+----------+----------+
      *
      * root
      * |-- country: string (nullable = true)
      * |-- population: integer (nullable = true)
      * |-- name: string (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62395762(): Unit = {
    val data =
      """
        |   name|               value
        |   null|                null
        |   null|                null
        |   null|                null
        |   null|                null
        |   null|                null
        |   null|                null
        |   null|                null
        |   null|                null
        |   null|                null
        |   null|                null
        |     id|                null
        |   name|                null
        |    age|                null
        |   food|                null
        |   null|                   1
        |   null|                 Joe
        |   null|                  47
        |   null|               pizza
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
      * +----+-----+
      * |name|value|
      * +----+-----+
      * |null|null |
      * |null|null |
      * |null|null |
      * |null|null |
      * |null|null |
      * |null|null |
      * |null|null |
      * |null|null |
      * |null|null |
      * |null|null |
      * |id  |null |
      * |name|null |
      * |age |null |
      * |food|null |
      * |null|1    |
      * |null|Joe  |
      * |null|47   |
      * |null|pizza|
      * +----+-----+
      *
      * root
      * |-- name: string (nullable = true)
      * |-- value: string (nullable = true)
      */

    df.select(map_from_arrays(collect_list("name").as("name"),
      collect_list("value").as("value")).as("map"))
      .select(explode_outer($"map").as(Seq("name", "value")))
      .show(false)

    /**
      * +----+-----+
      * |name|value|
      * +----+-----+
      * |id  |1    |
      * |name|Joe  |
      * |age |47   |
      * |food|pizza|
      * +----+-----+
      */

  }

  // ############################################################################################################

  @Test
  def test62395763(): Unit = {
    val path  = getClass.getResource("/header_footer_file.txt").getPath
    /**
      * File content - header_footer_file.txt
      * ---------------------------------------
      * 04/11/2020
      *
      * name;age;id
      * asdildsh;12;1
      * ram;13;2
      * oma;23;3
      * radahea;14;4
      * hellohow
      */
    val stringDS = spark.read.text(path).as(Encoders.STRING)
        .filter(s => s.contains(";"))
     stringDS.show(false)
    /**
      * +-------------+
      * |value        |
      * +-------------+
      * |name;age;id  |
      * |asdildsh;12;1|
      * |ram;13;2     |
      * |oma;23;3     |
      * |radahea;14;4 |
      * +-------------+
      */
    val df = spark.read
      .option("sep", ";")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)

    df.show(false)
    df.printSchema()

    /**
      * +--------+---+---+
      * |name    |age|id |
      * +--------+---+---+
      * |asdildsh|12 |1  |
      * |ram     |13 |2  |
      * |oma     |23 |3  |
      * |radahea |14 |4  |
      * +--------+---+---+
      *
      * root
      * |-- name: string (nullable = true)
      * |-- age: integer (nullable = true)
      * |-- id: integer (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62401222(): Unit = {
    val df = spark.sql("select Data, ID from values(array(1, 2, 3, 4), 22) T(Data, ID)")
    df.show(false)
    df.printSchema()
  }

  // ############################################################################################################

  @Test
  def test62406043(): Unit = {
    val df = Seq(
      ("state1", 1), ("state1", 2), ("state1", 3), ("state1", 4), ("state1", 5),
      ("state2", 1), ("state2", 2), ("state2", 3), ("state2", 4), ("state2", 5),
      ("state3", 1), ("state3", 2), ("state3", 3), ("state3", 4), ("state3", 5),
      ("state4", 1), ("state4", 2), ("state4", 3), ("state4", 4), ("state4", 5),
      ("state5", 1), ("state5", 2), ("state5", 3), ("state5", 4), ("state5", 5)
    ).toDF("state", "voter_id")

    // sample 3 voters for each state

    val voterIdsToSample: Double = 3 // put the records to sample for each stat
    // find distinct state
   val stateMap = df.groupBy("state").count().collect()
      .map(r => (r.getAs[String]("state"), r.getAs[Long]("count"))).toMap

    val fractions = collection.mutable.Map(stateMap.mapValues(voterIdsToSample/_).toSeq: _*)

    val sampleDF = df.rdd.map(r => (r.getAs[String]("state"), r.getAs[Int]("voter_id")))
      .sampleByKeyExact(withReplacement = false, fractions = fractions)
      .toDF("state", "voter_id")

    sampleDF.show(100, false)
    sampleDF.printSchema()

    /**
      * +------+--------+
      * |state |voter_id|
      * +------+--------+
      * |state1|3       |
      * |state1|4       |
      * |state1|5       |
      * |state2|1       |
      * |state2|2       |
      * |state2|4       |
      * |state3|1       |
      * |state3|3       |
      * |state3|5       |
      * |state4|2       |
      * |state4|4       |
      * |state4|5       |
      * |state5|3       |
      * |state5|4       |
      * |state5|5       |
      * +------+--------+
      *
      * root
      * |-- state: string (nullable = true)
      * |-- voter_id: integer (nullable = false)
      */
  }

  // ############################################################################################################

  @Test
  def test62406545(): Unit = {
    val table1 = Seq(
      ("o1", "i1", 1, 0.6),
      ("o1", "i1", 2, 0.4)
    ).toDF("outlet", "item", "day", "ratio")
    table1.show(false)
    /**
      * +------+----+---+-----+
      * |outlet|item|day|ratio|
      * +------+----+---+-----+
      * |o1    |i1  |1  |0.6  |
      * |o1    |i1  |2  |0.4  |
      * +------+----+---+-----+
      */

    val table2 = Seq(
      ("o1", "i1", 4, 5, 6, 8)
    ).toDF("outlet", "item", "week1", "week2", "week3", "week4")
    table2.show(false)
    /**
      * +------+----+-----+-----+-----+-----+
      * |outlet|item|week1|week2|week3|week4|
      * +------+----+-----+-----+-----+-----+
      * |o1    |i1  |4    |5    |6    |8    |
      * +------+----+-----+-----+-----+-----+
      */

    table1.join(table2, Seq("outlet", "item"))
      .groupBy("outlet", "item")
      .pivot("day")
      .agg(
        first($"week1" * $"ratio").as("week1"),
        first($"week2" * $"ratio").as("week2"),
        first($"week3" * $"ratio").as("week3"),
        first($"week4" * $"ratio").as("week4")
      ).show(false)

    /**
      * +------+----+-------+-------+------------------+-------+-------+-------+------------------+-------+
      * |outlet|item|1_week1|1_week2|1_week3           |1_week4|2_week1|2_week2|2_week3           |2_week4|
      * +------+----+-------+-------+------------------+-------+-------+-------+------------------+-------+
      * |o1    |i1  |2.4    |3.0    |3.5999999999999996|4.8    |1.6    |2.0    |2.4000000000000004|3.2    |
      * +------+----+-------+-------+------------------+-------+-------+-------+------------------+-------+
      */
  }

  // ############################################################################################################

  @Test
  def test62407342(): Unit = {
    val data =
      """
        |Name    |  Place    |     Department | Experience
        |
        |Ram      | Ramgarh      |  Sales      |  14
        |
        |Lakshman | Lakshmanpur  |Operations   |
        |
        |Sita     | Sitapur      |             |  14
        |
        |Ravan   |              |              |  25
      """.stripMargin

    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
//      .option("nullValue", "null")
      .csv(stringDS)

    df.show(false)
    df.printSchema()
    /**
      * +--------+-----------+----------+----------+
      * |Name    |Place      |Department|Experience|
      * +--------+-----------+----------+----------+
      * |Ram     |Ramgarh    |Sales     |14        |
      * |Lakshman|Lakshmanpur|Operations|null      |
      * |Sita    |Sitapur    |null      |14        |
      * |Ravan   |null       |null      |25        |
      * +--------+-----------+----------+----------+
      *
      * root
      * |-- Name: string (nullable = true)
      * |-- Place: string (nullable = true)
      * |-- Department: string (nullable = true)
      * |-- Experience: integer (nullable = true)
      */

    val x = df.withColumn("Not_null_columns_values",
      to_json(struct(df.columns.map(col): _*)))
    x.show(false)
    x.printSchema()

    /**
      * +--------+-----------+----------+----------+---------------------------------------------------------------------+
      * |Name    |Place      |Department|Experience|Not_null_columns_values                                              |
      * +--------+-----------+----------+----------+---------------------------------------------------------------------+
      * |Ram     |Ramgarh    |Sales     |14        |{"Name":"Ram","Place":"Ramgarh","Department":"Sales","Experience":14}|
      * |Lakshman|Lakshmanpur|Operations|null      |{"Name":"Lakshman","Place":"Lakshmanpur","Department":"Operations"}  |
      * |Sita    |Sitapur    |null      |14        |{"Name":"Sita","Place":"Sitapur","Experience":14}                    |
      * |Ravan   |null       |null      |25        |{"Name":"Ravan","Experience":25}                                     |
      * +--------+-----------+----------+----------+---------------------------------------------------------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62420684(): Unit = {
    val data1 =
      """
        |id | name
        |1  | Alex
        |2  | Bob
        |3  | Chris
        |4  | Kevin
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
      * +---+-----+
      * |id |name |
      * +---+-----+
      * |1  |Alex |
      * |2  |Bob  |
      * |3  |Chris|
      * |4  |Kevin|
      * +---+-----+
      *
      * root
      * |-- id: integer (nullable = true)
      * |-- name: string (nullable = true)
      */

    val df2 =
      spark.sql(
        """
          |select id, friends from values
          | (1, array(named_struct('id', 2, 'score', 49), named_struct('id', 3, 'score', 15))),
          | (2, array(named_struct('id', 4, 'score', 61), named_struct('id', 2, 'score', 49), named_struct('id', 3,
          | 'score', 4)))
          | T(id, friends)
        """.stripMargin)
    df2.show(false)
    df2.printSchema()
    /**
      * +---+--------------------------+
      * |id |friends                   |
      * +---+--------------------------+
      * |1  |[[2, 49], [3, 15]]        |
      * |2  |[[4, 61], [2, 49], [3, 4]]|
      * +---+--------------------------+
      *
      * root
      * |-- id: integer (nullable = false)
      * |-- friends: array (nullable = false)
      * |    |-- element: struct (containsNull = false)
      * |    |    |-- id: integer (nullable = false)
      * |    |    |-- score: integer (nullable = false)
      */

    // if df1 has big data
    val exploded = df2.select($"id", explode(expr("friends.id")).as("friend_id"))
      exploded.join(df1, exploded("friend_id")===df1("id"))
      .groupBy(exploded("id"))
      .agg(collect_list($"name").as("friends"))
      .show(false)
    /**
      * +---+-------------------+
      * |id |friends            |
      * +---+-------------------+
      * |2  |[Bob, Chris, Kevin]|
      * |1  |[Bob, Chris]       |
      * +---+-------------------+
      */

    // if df1 is small
    val b = spark.sparkContext.broadcast(df1.collect().map{case Row(id: Int, name: String) => id -> name}.toMap)

    val getFriendsName = udf((idArray: mutable.WrappedArray[Int]) => idArray.map(b.value(_)))

    df2.withColumn("friends", getFriendsName(expr("friends.id")))
      .show(false)

    /**
      * +---+-------------------+
      * |id |friends            |
      * +---+-------------------+
      * |1  |[Bob, Chris]       |
      * |2  |[Kevin, Bob, Chris]|
      * +---+-------------------+
      */

  }

  // ############################################################################################################

  @Test
  def test62423032(): Unit = {
    val data1 =
      """
        |name| age| degree| dept
        |aaa | 20| ece |null
        |bbb |20 |it |null
        |ccc |30 |mech| null
      """.stripMargin

    val stringDS1 = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val soruce_df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    soruce_df.show(false)
    soruce_df.printSchema()

    /**
      * +----+---+------+----+
      * |name|age|degree|dept|
      * +----+---+------+----+
      * |aaa |20 |ece   |null|
      * |bbb |20 |it    |null|
      * |ccc |30 |mech  |null|
      * +----+---+------+----+
      *
      * root
      * |-- name: string (nullable = true)
      * |-- age: integer (nullable = true)
      * |-- degree: string (nullable = true)
      * |-- dept: string (nullable = true)
      */

    val data2 =
      """
        |name| age |degree |dept
        |aaa  |20| ece |null
        |bbb |20 |it| null
      """.stripMargin

    val stringDS2 = data2.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val target_df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    target_df.show(false)
    target_df.printSchema()

    /**
      * +----+---+------+----+
      * |name|age|degree|dept|
      * +----+---+------+----+
      * |aaa |20 |ece   |null|
      * |bbb |20 |it    |null|
      * +----+---+------+----+
      *
      * root
      * |-- name: string (nullable = true)
      * |-- age: integer (nullable = true)
      * |-- degree: string (nullable = true)
      * |-- dept: string (nullable = true)
      */

    soruce_df.join(target_df,Seq("name","age","degree"),"leftanti").show(false)

    /**
      * +----+---+------+----+
      * |name|age|degree|dept|
      * +----+---+------+----+
      * |ccc |30 |mech  |null|
      * +----+---+------+----+
      */

    soruce_df.join(target_df,Seq("name","age","degree","dept"),"leftanti").show(false)
    /**
      * +----+---+------+----+
      * |name|age|degree|dept|
      * +----+---+------+----+
      * |bbb |20 |it    |null|
      * |aaa |20 |ece   |null|
      * |ccc |30 |mech  |null|
      * +----+---+------+----+
      */

    // Equality test that is safe for null values.
    soruce_df.join(target_df, soruce_df("name") <=> target_df("name") && soruce_df("age") <=> target_df("age") &&
      soruce_df("degree") <=> target_df("degree") && soruce_df("dept") <=> target_df("dept")
      ,"leftanti").show(false)

    /**
      * +----+---+------+----+
      * |name|age|degree|dept|
      * +----+---+------+----+
      * |ccc |30 |mech  |null|
      * +----+---+------+----+
      */

  }
  // ############################################################################################################

  @Test
  def test62428746(): Unit = {
    val df =
      spark.sql(
        """
          |select id, size, variantID from values
          | (1, array(10, 20), array(150, 160)),
          | (2, array(2), array(1)),
          | (3, array(null), array(null))
          | T(id, size, variantID)
        """.stripMargin)
    df.show(false)
    df.printSchema()
    /**
      * +---+--------+----------+
      * |id |size    |variantID |
      * +---+--------+----------+
      * |1  |[10, 20]|[150, 160]|
      * |2  |[2]     |[1]       |
      * |3  |[]      |[]        |
      * +---+--------+----------+
      *
      * root
      * |-- id: integer (nullable = false)
      * |-- size: array (nullable = false)
      * |    |-- element: integer (containsNull = true)
      * |-- variantID: array (nullable = false)
      * |    |-- element: integer (containsNull = true)
      */

    val p = df.withColumn("sizeMap", arrays_zip($"size", $"variantID"))
      .withColumn("sizeMap", expr("TRANSFORM(sizeMap, x -> concat_ws('|', x.size, x.variantID))"))
    p.show(false)
    p.printSchema()

    /**
      * +---+--------+----------+----------------+
      * |id |size    |variantID |sizeMap         |
      * +---+--------+----------+----------------+
      * |1  |[10, 20]|[150, 160]|[10|150, 20|160]|
      * |2  |[2]     |[1]       |[2|1]           |
      * |3  |[]      |[]        |[]              |
      * +---+--------+----------+----------------+
      *
      * root
      * |-- id: integer (nullable = false)
      * |-- size: array (nullable = false)
      * |    |-- element: integer (containsNull = true)
      * |-- variantID: array (nullable = false)
      * |    |-- element: integer (containsNull = true)
      * |-- sizeMap: array (nullable = false)
      * |    |-- element: string (containsNull = false)
      */
  }

  // ############################################################################################################

  @Test
  def test62432982(): Unit = {
    val Data = spark.range(2).withColumn("Ip", lit(10))
    val myUdf1 = udf((Number: Long) => ((Number) >> 24) & 255)
    val myUdf2 = udf((Number: Long) => ((Number) >> 16) & 255)
    val myUdf3 = udf((Number: Long) => ((Number) >> 8) & 255)
    val myUdf4 = udf((Number: Long) => (Number) & 255)

    val df=Data.withColumn("bitwise 1", myUdf1(Data("Ip")))
      .withColumn("bitwise 2", myUdf2(Data("Ip")))
      .withColumn("bitwise 3", myUdf3(Data("Ip")))
      .withColumn("bitwise 4", myUdf4(Data("Ip")))

    val FinalDF =  df.withColumn("FinalIp",concat(col("bitwise 1"),lit("."),col("bitwise 2"),lit("."),col("bitwise 3"),lit("."),col("bitwise 4")))
      .drop("bitwise 1").drop("bitwise 2").drop("bitwise 3").drop("bitwise 4")
    FinalDF.show(false)

    /**
      * +---+---+--------+
      * |id |Ip |FinalIp |
      * +---+---+--------+
      * |0  |10 |0.0.0.10|
      * |1  |10 |0.0.0.10|
      * +---+---+--------+
      */

    spark.range(2).withColumn("Ip", lit(10))
      .withColumn("FinalIp",
        concat_ws(".", expr("shiftRight(Ip, 24) & 255"), expr("shiftRight(Ip, 16) & 255"),
          expr("shiftRight(Ip, 8) & 255"), expr("Ip & 255"))
      ).show(false)

    /**
      * +---+---+--------+
      * |id |Ip |FinalIp |
      * +---+---+--------+
      * |0  |10 |0.0.0.10|
      * |1  |10 |0.0.0.10|
      * +---+---+--------+
      */
  }

  // ############################################################################################################

  @Test
  def test62432035(): Unit = {
    val df = spark.sql(
      """
        |select elements from values (array('a-1', 'a-2')) T(elements)
      """.stripMargin)

    // spark >= 2.4.0
    df.withColumn("p", expr("TRANSFORM(elements, x -> named_struct('id', x, 'v', 'In'))"))
      .withColumn("p_json", to_json($"p"))
      .show(false)

//
//    +----------+----------------------+---------------------------------------------+
//    |elements  |p                     |p_json                                       |
//    +----------+----------------------+---------------------------------------------+
//    |[a-1, a-2]|[[a-1, In], [a-2, In]]|[{"id":"a-1","v":"In"},{"id":"a-2","v":"In"}]|
//    +----------+----------------------+---------------------------------------------+
//

  }

  // ############################################################################################################

  @Test
  def test62446596(): Unit = {

    val df1 = spark.sql("select col1, col2 from values (null, 1), (2, null), (null, null), (1,2) T(col1, col2)")
    /**
      * +----+----+
      * |col1|col2|
      * +----+----+
      * |null|1   |
      * |2   |null|
      * |null|null|
      * |1   |2   |
      * +----+----+
      */

    df1.show(false)
    df1.filter(df1.columns.map(col(_).isNull).reduce(_ || _)).show(false)

    /**
      * +----+----+
      * |col1|col2|
      * +----+----+
      * |null|1   |
      * |2   |null|
      * |null|null|
      * +----+----+
      */
  }

  // ############################################################################################################

  @Test
  def test62441435(): Unit = {
    // [^0-9a-zA-Z]+ => this will remove all special chars
    spark.range(2).withColumn("str", lit("abc%xyz_12$q"))
      .withColumn("replace", regexp_replace($"str", "[^0-9a-zA-Z]+", "_"))
      .show(false)

    /**
      * +---+------------+------------+
      * |id |str         |replace     |
      * +---+------------+------------+
      * |0  |abc%xyz_12$q|abc_xyz_12_q|
      * |1  |abc%xyz_12$q|abc_xyz_12_q|
      * +---+------------+------------+
      */

    // if you don't want to remove some special char like $ etc, include it [^0-9a-zA-Z$]+
    spark.range(2).withColumn("str", lit("abc%xyz_12$q"))
      .withColumn("replace", regexp_replace($"str", "[^0-9a-zA-Z$]+", "_"))
      .show(false)

    /**
      * +---+------------+------------+
      * |id |str         |replace     |
      * +---+------------+------------+
      * |0  |abc%xyz_12$q|abc_xyz_12$q|
      * |1  |abc%xyz_12$q|abc_xyz_12$q|
      * +---+------------+------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62446585(): Unit = {
    val inputDf = Seq(
      (1, "Mr"),
      (1, "Mme"),
      (1, "Mr"),
      (1, null),
      (1, null),
      (1, null),
      (2, "Mr"),
      (3, null)).toDF("UNIQUE_GUEST_ID", "PREFIX")
    println("Input:")
    inputDf.show(false)
    /**
      * Input:
      * +---------------+------+
      * |UNIQUE_GUEST_ID|PREFIX|
      * +---------------+------+
      * |1              |Mr    |
      * |1              |Mme   |
      * |1              |Mr    |
      * |1              |null  |
      * |1              |null  |
      * |1              |null  |
      * |2              |Mr    |
      * |3              |null  |
      * +---------------+------+
      */

    inputDf
      .groupBy($"UNIQUE_GUEST_ID", $"PREFIX").agg(count($"PREFIX").as("count"))
      .groupBy($"UNIQUE_GUEST_ID")
      .agg(max( struct( $"count", $"PREFIX")).as("max"))
      .selectExpr("UNIQUE_GUEST_ID", "max.PREFIX")
      .show(false)

    /**
      * +---------------+------+
      * |UNIQUE_GUEST_ID|PREFIX|
      * +---------------+------+
      * |2              |Mr    |
      * |1              |Mr    |
      * |3              |null  |
      * +---------------+------+
      */
  }

  // ############################################################################################################

  @Test
  def test62446423(): Unit = {
    val data =
      """
        |{
        |"timestampField":"08.06.2020 12:03:50"
        |}
      """.stripMargin
    val df = spark.read.option("multiLine", true).json(Seq(data).toDS())
    df.show(false)
    df.printSchema()
    /**
      * +-------------------+
      * |timestampField     |
      * +-------------------+
      * |08.06.2020 12:03:50|
      * +-------------------+
      *
      * root
      * |-- timestampField: string (nullable = true)
      */

    val df1 = spark.read
        .schema(StructType(StructField("timestampField", DataTypes.TimestampType) :: Nil))
      .option("multiLine", true)
      .option("timestampFormat", "MM.dd.yyyy HH:mm:ss")
      .json(Seq(data) toDS())
    df1.show(false)
    df1.printSchema()

    /**
      * +-------------------+
      * |timestampField     |
      * +-------------------+
      * |2020-08-06 12:03:50|
      * +-------------------+
      *
      * root
      * |-- timestampField: timestamp (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62447474(): Unit = {
    val data =
      """
        |Customer, Product
        | A, <XmlData ProductName="123">x</XmlData><XmlData ProductName="1452">y</XmlData>
        | B, <XmlData ProductName="123">x</XmlData>
        | C, <XmlData ProductName="123">x</XmlData><XmlData ProductName="1452">y</XmlData>
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\,").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df.printSchema()
    df.show(false)

    val regexp_extractAll = udf((xml: String, exp: String) =>
      exp.r.findAllMatchIn(xml).map(_.group(1)).mkString(";")
    )

    df.withColumn("ProductName", regexp_extractAll($"Product", lit("""ProductName="(\d+)"""")))
      .show(false)

    /**
      * +--------+-----------------------------------------------------------------------------+--------+
      * |Customer|Product                                                                      |ProductName       |
      * +--------+-----------------------------------------------------------------------------+--------+
      * |A       |<XmlData ProductName="123">x</XmlData><XmlData ProductName="1452">y</XmlData>|123;1452|
      * |B       |<XmlData ProductName="123">x</XmlData>                                       |123     |
      * |C       |<XmlData ProductName="123">x</XmlData><XmlData ProductName="1452">y</XmlData>|123;1452|
      * +--------+-----------------------------------------------------------------------------+--------+
      */


    val data1 =
      """
        |a||b|c
        |z||y|x
      """.stripMargin
    data1.split(System.lineSeparator()).toSeq.toDS().map(_.replaceAll("\\|\\|", "|")).show(false)

    /**
      * +------+
      * |value |
      * +------+
      * |      |
      * |a|b|c |
      * |z|y|x |
      * |      |
      * +------+
      */
    spark.read
      .option("sep", "|")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(data1.split(System.lineSeparator()).toSeq.toDS().map(_.replaceAll("\\|\\|", "|")))
      .show(false)

    /**
      * +---+---+---+
      * |a  |b  |c  |
      * +---+---+---+
      * |z  |y  |x  |
      * +---+---+---+
      */
  }

  // ############################################################################################################

  @Test
  def test62452724(): Unit = {
    val data =
      """
        |id| f_1 | f_2
        | 1| null| null
        | 2|123  | null
        | 3|124  |127
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
    df.printSchema()
    df.show(false)

    /**
      * root
      * |-- id: integer (nullable = true)
      * |-- f_1: integer (nullable = true)
      * |-- f_2: integer (nullable = true)
      *
      * +---+----+----+
      * |id |f_1 |f_2 |
      * +---+----+----+
      * |1  |null|null|
      * |2  |123 |null|
      * |3  |124 |127 |
      * +---+----+----+
      */

    df.withColumn("array", array(df.columns.filter(_.startsWith("f")).map(col): _*))
      .withColumn("combined", expr("FILTER(array, x -> x is not null)"))
      .show(false)

    /**
      * +---+----+----+----------+----------+
      * |id |f_1 |f_2 |array     |combined  |
      * +---+----+----+----------+----------+
      * |1  |null|null|[,]       |[]        |
      * |2  |123 |null|[123,]    |[123]     |
      * |3  |124 |127 |[124, 127]|[124, 127]|
      * +---+----+----+----------+----------+
      */
  }

  // ############################################################################################################

  @Test
  def test62460502(): Unit = {
    val end = 100 // change this as required
    val ds = spark.sql(s"select value from values (sequence(0, $end)) T(value)")
      .selectExpr("explode(value) as value").selectExpr("(value * rand()) value")
      .as(Encoders.DOUBLE)

    ds.show(false)
    ds.printSchema()
    /**
      * +-------------------+
      * |value              |
      * +-------------------+
      * |0.0                |
      * |0.6598598027815629 |
      * |0.34305452447822704|
      * |0.2421654251914631 |
      * |3.1937041196518896 |
      * |0.9120972627613766 |
      * |3.307431250924596  |
      *
      * root
      * |-- value: double (nullable = false)
      */
  }
  // ############################################################################################################

  @Test
  def test62463467(): Unit = {
    val data =
      """
        |person_id  | order_id  |     order_ts         |order_amt
        |   1       |     1     |  2020-01-01 10:10:10 |    10
        |   1       |     2     |  2020-01-01 10:15:15 |    15
        |   2       |     3     |  2020-01-01 10:10:10 |    0
        |   2       |     4     |  2020-01-01 10:15:15 |    15
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
    df.printSchema()
    df.show(false)

    /**
      * root
      * |-- person_id: integer (nullable = true)
      * |-- order_id: integer (nullable = true)
      * |-- order_ts: timestamp (nullable = true)
      * |-- order_amt: integer (nullable = true)
      *
      * +---------+--------+-------------------+---------+
      * |person_id|order_id|order_ts           |order_amt|
      * +---------+--------+-------------------+---------+
      * |1        |1       |2020-01-01 10:10:10|10       |
      * |1        |2       |2020-01-01 10:15:15|15       |
      * |2        |3       |2020-01-01 10:10:10|0        |
      * |2        |4       |2020-01-01 10:15:15|15       |
      * +---------+--------+-------------------+---------+
      */

    // SPark DSL
    df.withColumn("latest", max($"order_ts").over(Window.partitionBy("person_id")))
      .withColumn("valid_order", when(unix_timestamp($"latest") - unix_timestamp($"order_ts") =!= 0, lit("N"))
          .otherwise(lit("Y"))
      )
      .show(false)

    /**
      * +---------+--------+-------------------+---------+-------------------+-----------+
      * |person_id|order_id|order_ts           |order_amt|latest             |valid_order|
      * +---------+--------+-------------------+---------+-------------------+-----------+
      * |2        |3       |2020-01-01 10:10:10|0        |2020-01-01 10:15:15|N          |
      * |2        |4       |2020-01-01 10:15:15|15       |2020-01-01 10:15:15|Y          |
      * |1        |1       |2020-01-01 10:10:10|10       |2020-01-01 10:15:15|N          |
      * |1        |2       |2020-01-01 10:15:15|15       |2020-01-01 10:15:15|Y          |
      * +---------+--------+-------------------+---------+-------------------+-----------+
      */

    // Spark SQL
    df.createOrReplaceTempView("order_table")
    spark.sql(
      """
        |select person_id, order_id, order_ts, order_amt, latest,
        | case when (unix_timestamp(latest) - unix_timestamp(order_ts) != 0) then 'N' else 'Y' end as  valid_order
        | from
        | (select person_id, order_id, order_ts, order_amt, max(order_ts) over (partition by person_id) as latest FROM order_table) a
      """.stripMargin)
      .show(false)

    /**
      * +---------+--------+-------------------+---------+-------------------+-----------+
      * |person_id|order_id|order_ts           |order_amt|latest             |valid_order|
      * +---------+--------+-------------------+---------+-------------------+-----------+
      * |2        |3       |2020-01-01 10:10:10|0        |2020-01-01 10:15:15|N          |
      * |2        |4       |2020-01-01 10:15:15|15       |2020-01-01 10:15:15|Y          |
      * |1        |1       |2020-01-01 10:10:10|10       |2020-01-01 10:15:15|N          |
      * |1        |2       |2020-01-01 10:15:15|15       |2020-01-01 10:15:15|Y          |
      * +---------+--------+-------------------+---------+-------------------+-----------+
      */

  }
  // ############################################################################################################

  @Test
  def test62467239(): Unit = {
    val data =
      """
        |order_id       |   product_id|quantity
        |A              |X            |       5
        |A              |Y            |       1
        |A              |Z            |       3
        |A              |X            |      -1
        |A              |Z            |      -1
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
    df.printSchema()
    df.show(false)

    /**
      * root
      * |-- order_id: string (nullable = true)
      * |-- product_id: string (nullable = true)
      * |-- quantity: integer (nullable = true)
      *
      * +--------+----------+--------+
      * |order_id|product_id|quantity|
      * +--------+----------+--------+
      * |A       |X         |5       |
      * |A       |Y         |1       |
      * |A       |Z         |3       |
      * |A       |X         |-1      |
      * |A       |Z         |-1      |
      * +--------+----------+--------+
      */

    df.withColumn("is_negative", $"quantity" < 0)
      .withColumn("position_number", row_number()
      .over(Window.partitionBy($"order_id", $"is_negative").orderBy("product_id")))
      .withColumn("position_number",
        when($"is_negative", max(expr("if(is_negative, 0, position_number)"))
          .over(Window.partitionBy("order_id", "product_id")) + 1000)
          .otherwise($"position_number")
      )
      .show(false)

    /**
      * +--------+----------+--------+-----------+---------------+
      * |order_id|product_id|quantity|is_negative|position_number|
      * +--------+----------+--------+-----------+---------------+
      * |A       |Y         |1       |false      |2              |
      * |A       |Z         |3       |false      |3              |
      * |A       |Z         |-1      |true       |1003           |
      * |A       |X         |5       |false      |1              |
      * |A       |X         |-1      |true       |1001           |
      * +--------+----------+--------+-----------+---------------+
      */
  }
  // ############################################################################################################

  @Test
  def test62451164(): Unit = {
    val df = spark.range(2).withColumnRenamed("id", "day")
      .withColumn("table_row", expr("array(named_struct('DATE', 'sample_date'," +
        " 'ADMISSION_NUM', 'sample_adm_num', 'SOURCE_CODE', 'sample_source_code'))"))
    df.show(false)
    df.printSchema()

//
//   +---+---------------------------------------------------+
//   |day|table_row                                          |
//   +---+---------------------------------------------------+
//   |0  |[[sample_date, sample_adm_num, sample_source_code]]|
//   |1  |[[sample_date, sample_adm_num, sample_source_code]]|
//   +---+---------------------------------------------------+
//
//   root
//   |-- day: long (nullable = false)
//   |-- table_row: array (nullable = false)
//   |    |-- element: struct (containsNull = false)
//   |    |    |-- DATE: string (nullable = false)
//   |    |    |-- ADMISSION_NUM: string (nullable = false)
//   |    |    |-- SOURCE_CODE: string (nullable = false)
//

    def keepColumnInarray(columnsToKeep: Seq[String], rows: mutable.WrappedArray[Row]) = {
      rows.map(r => {
        new GenericRowWithSchema(r.getValuesMap(columnsToKeep).values.toArray,
          StructType(r.schema.filter(s => columnsToKeep.contains(s.name))))
      })
    }

    val keepColumns = udf((columnsToKeep: Seq[String], rows: mutable.WrappedArray[Row]) =>
      keepColumnInarray(columnsToKeep, rows)
      , ArrayType(StructType(StructField("DATE", StringType) :: Nil)))

    val processedDF = df
      .withColumn("table_row_new", keepColumns(array(lit("DATE")), col("table_row")))
    processedDF.show(false)
    processedDF.printSchema()

//
//    +---+---------------------------------------------------+---------------+
//    |day|table_row                                          |table_row_new  |
//    +---+---------------------------------------------------+---------------+
//    |0  |[[sample_date, sample_adm_num, sample_source_code]]|[[sample_date]]|
//    |1  |[[sample_date, sample_adm_num, sample_source_code]]|[[sample_date]]|
//    +---+---------------------------------------------------+---------------+
//
//    root
//    |-- day: long (nullable = false)
//    |-- table_row: array (nullable = false)
//    |    |-- element: struct (containsNull = false)
//    |    |    |-- DATE: string (nullable = false)
//    |    |    |-- ADMISSION_NUM: string (nullable = false)
//    |    |    |-- SOURCE_CODE: string (nullable = false)
//    |-- table_row_new: array (nullable = true)
//    |    |-- element: struct (containsNull = true)
//    |    |    |-- DATE: string (nullable = true)
//
  }
  // ############################################################################################################

  @Test
  def test62483526(): Unit = {

    val dataset =Seq(1.0, 2.0).toDF("id")
    dataset.show(false)

    val assembler = new VectorAssembler()
      .setInputCols(Array("id"))
      .setOutputCol("features")

    val output = assembler.transform(dataset)
    println("Assembled columns ")
    output.select("id").show(false)
    output.printSchema()

    /**
      * Assembled columns
      * +---+
      * |id |
      * +---+
      * |1.0|
      * |2.0|
      * +---+
      *
      * root
      * |-- id: double (nullable = false)
      * |-- features: vector (nullable = true)
      */
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("vScaled")
      .setMax(1)
      .setMin(0)
    val ScalarModel =scaler.fit(output)
    val scalarData =ScalarModel.transform(output)

    scalarData.select("vScaled").show()
    /**
      * +-------+
      * |vScaled|
      * +-------+
      * |  [0.0]|
      * |  [1.0]|
      * +-------+
      */

    val ScaledCol: List[Row] = scalarData.select("vScaled").collect.toList
//    var listofScaledCol: List[Double] = ScaledCol.map(r => r.getDouble(0))
    var listofScaledCol: List[Double] = ScaledCol.map(_.getAs[Vector]("vScaled")(0))
    print(listofScaledCol)
    // List(0.0, 1.0)
  }

  // ############################################################################################################

  @Test
  def test62486780(): Unit = {
    val data =
      """
        |id | group | text
        |1  | a     |  hey there
        |2  | c     |  no you can
        |3  | a     |  yes yes yes
        |4  | b     |  yes or no
        |5  | b     |  you need to say yes.
        |6  | a     |  YES you can
        |7  | d     |  yes!
        |8  | c     |  no&
        |9  | b     |  ok
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
    df.printSchema()
    df.show(false)

    /**
      * root
      * |-- id: integer (nullable = true)
      * |-- group: string (nullable = true)
      * |-- text: string (nullable = true)
      *
      * +---+-----+--------------------+
      * |id |group|text                |
      * +---+-----+--------------------+
      * |1  |a    |hey there           |
      * |2  |c    |no you can          |
      * |3  |a    |yes yes yes         |
      * |4  |b    |yes or no           |
      * |5  |b    |you need to say yes.|
      * |6  |a    |yes you can         |
      * |7  |d    |yes!                |
      * |8  |c    |no&                 |
      * |9  |b    |ok                  |
      * +---+-----+--------------------+
      */

    df.createOrReplaceTempView("my_table")
    val sql_q = spark.sql(
      """
        |select group, sum(
        |  case when (text rlike '(?i)^.*yes.*$') then 1 else 0 end
        | ) as count
        |from my_table group by group
      """.stripMargin)
    sql_q.show(false)

    /**
      * +-----+-----+
      * |group|count|
      * +-----+-----+
      * |a    |2    |
      * |c    |0    |
      * |d    |1    |
      * |b    |2    |
      * +-----+-----+
      */

  }
  // ############################################################################################################

  @Test
  def test62494403(): Unit = {
    val data =
      """
        |userId|movieId|rating| ts       |ratingtimestamp
        |     1|    296|   5.0|1147880044|           null
        |     1|    306|   3.5|1147868817|           null
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
    df.printSchema()
    df.show(false)
    /**
      * root
      * |-- userId: integer (nullable = true)
      * |-- movieId: integer (nullable = true)
      * |-- rating: double (nullable = true)
      * |-- ts: integer (nullable = true)
      * |-- ratingtimestamp: string (nullable = true)
      *
      * +------+-------+------+----------+---------------+
      * |userId|movieId|rating|ts        |ratingtimestamp|
      * +------+-------+------+----------+---------------+
      * |1     |296    |5.0   |1147880044|null           |
      * |1     |306    |3.5   |1147868817|null           |
      * +------+-------+------+----------+---------------+
      */

    val p = df.withColumn("ratingString",from_unixtime($"ts" ,"MM/dd/yyyy HH:mm:ss"))
        .withColumn("ratingtimestamp", unix_timestamp($"ratingString", "MM/dd/yyyy HH:mm:ss").cast("timestamp"))
        .withColumn("ts_new", $"ts".cast("long").cast("timestamp"))
    p.show(false)
    p.printSchema()

    /**
      * +------+-------+------+----------+-------------------+-------------------+-------------------+
      * |userId|movieId|rating|ts        |ratingtimestamp    |ratingString       |ts_new             |
      * +------+-------+------+----------+-------------------+-------------------+-------------------+
      * |1     |296    |5.0   |1147880044|2006-05-17 21:04:04|05/17/2006 21:04:04|2006-05-17 21:04:04|
      * |1     |306    |3.5   |1147868817|2006-05-17 17:56:57|05/17/2006 17:56:57|2006-05-17 17:56:57|
      * +------+-------+------+----------+-------------------+-------------------+-------------------+
      *
      * root
      * |-- userId: integer (nullable = true)
      * |-- movieId: integer (nullable = true)
      * |-- rating: double (nullable = true)
      * |-- ts: integer (nullable = true)
      * |-- ratingtimestamp: timestamp (nullable = true)
      * |-- ratingString: string (nullable = true)
      * |-- ts_new: timestamp (nullable = true)
      */

    df.withColumn("ts_new", $"ts".cast("long").cast("timestamp")).show(false)

    // default format - yyyy-MM-dd HH:mm:ss
    /**
      * +------+-------+------+----------+---------------+-------------------+
      * |userId|movieId|rating|ts        |ratingtimestamp|ts_new             |
      * +------+-------+------+----------+---------------+-------------------+
      * |1     |296    |5.0   |1147880044|null           |2006-05-17 21:04:04|
      * |1     |306    |3.5   |1147868817|null           |2006-05-17 17:56:57|
      * +------+-------+------+----------+---------------+-------------------+
      */

    df.withColumn("ts_new", from_unixtime($"ts".cast("long").cast("timestamp").cast("long"),
      "MM/dd/yyyy HH:mm:ss")).show(false)

    /**
      * +------+-------+------+----------+---------------+-------------------+
      * |userId|movieId|rating|ts        |ratingtimestamp|ts_new             |
      * +------+-------+------+----------+---------------+-------------------+
      * |1     |296    |5.0   |1147880044|null           |05/17/2006 21:04:04|
      * |1     |306    |3.5   |1147868817|null           |05/17/2006 17:56:57|
      * +------+-------+------+----------+---------------+-------------------+
      */
  }
  // ############################################################################################################

  @Test
  def test62499029(): Unit = {
    val df = Seq("corona", "corona?", "this is corona", "coronavirus", "corona's", "is this corona?")
      .toDF("text")
      .withColumn("dates", monotonically_increasing_id())
    df.show(false)
    df.printSchema()

    /**
      * +---------------+-----+
      * |text           |dates|
      * +---------------+-----+
      * |corona         |0    |
      * |corona?        |1    |
      * |this is corona |2    |
      * |coronavirus    |3    |
      * |corona's       |4    |
      * |is this corona?|5    |
      * +---------------+-----+
      *
      * root
      * |-- text: string (nullable = true)
      * |-- dates: long (nullable = false)
      */
    df.createOrReplaceTempView("my_table")
    spark.sql(
      """
        | select dates, sum(
        |         case when (text rlike '\\bcorona\\b') then 1
        |         else 0 end
        |        ) as check
        |        from my_table group by dates
      """.stripMargin)
      .show(false)

    /**
      * +-----+-----+
      * |dates|check|
      * +-----+-----+
      * |2    |1    |
      * |4    |1    |
      * |5    |1    |
      * |0    |1    |
      * |1    |1    |
      * |3    |0    |
      * +-----+-----+
      */

  }

  // ############################################################################################################

  @Test
  def test62506495(): Unit = {
    val df = Seq((1, 2.0, "shared")).toDF("id", "shared", "shared")
    df.show(false)
    df.printSchema()
    /**
      * +---+------+------+
      * |id |shared|shared|
      * +---+------+------+
      * |1  |2.0   |shared|
      * +---+------+------+
      *
      * root
      * |-- id: integer (nullable = false)
      * |-- shared: double (nullable = false)
      * |-- shared: string (nullable = true)
      */

    // 1. get all the duplicated column names
    val findDupCols = (cols: Array[String]) => cols.map((_ , 1)).groupBy(_._1).filter(_._2.length > 1).keys.toSeq
    val dupCols = findDupCols(df.columns)
    println(dupCols.mkString(", "))

    val renamedDF = df
      // 2 rename duplicate cols like shared => shared:string, shared:int
      .toDF(df.schema
        .map{case StructField(name, dt, _, _) =>
          if(dupCols.contains(name)) s"$name:${dt.simpleString}" else name}: _*)
    // 3. create struct of all cols
    val structCols = df.schema.map(f => f.name -> f  ).groupBy(_._1)
      .map{case(name, seq) =>
        if (seq.length > 1)
          struct(
            seq.map { case (_, StructField(fName, dt, _, _)) =>
              expr(s"`$fName:${dt.simpleString}` as ${dt.simpleString}")
            }: _*
          ).as(name)
        else col(name)
      }.toSeq
     val structDF = renamedDF.select(structCols: _*)

    structDF.show(false)
    structDF.printSchema()

    /**
      * +-------------+---+
      * |shared       |id |
      * +-------------+---+
      * |[2.0, shared]|1  |
      * +-------------+---+
      *
      * root
      * |-- shared: struct (nullable = false)
      * |    |-- double: double (nullable = false)
      * |    |-- string: string (nullable = true)
      * |-- id: integer (nullable = false)
      */

    // Use the datafrmae without losing any columns
    structDF.selectExpr("id", "shared.double as shared").show(false)
    /**
      * +---+------+
      * |id |shared|
      * +---+------+
      * |1  |2.0   |
      * +---+------+
      */


  }

  // ############################################################################################################

  // same SO = 62488717
  @Test
  def test62502681(): Unit = {
    val dt_formats= Seq("dd-MMM-yyyy", "MMM-dd-yyyy", "yyyy-MM-dd","MM/dd/yy","dd-MM-yy","dd-MM-yyyy","yyyy/MM/dd","dd/MM/yyyy")

    val newDF =  Seq("20-Mar-2020").toDF()
      .withColumn("ETD", coalesce(dt_formats.map(fmt => to_date($"ETD", fmt)):_*))
  }

  // ############################################################################################################

  @Test
  def test62510279(): Unit = {
    val data =
      """
        |{"Name":"Ram","Place":"RamGarh"}
        |{"Name":"Lakshman","Place":"LakshManPur","DepartMent":"Operations"}
        |{"Name":"Sita","Place":"SitaPur","Experience":14.0}
      """.stripMargin
    val df = spark.read.json(data.split(System.lineSeparator()).toSeq.toDS())
    df.show(false)
    df.printSchema()
    /**
      * +----------+----------+--------+-----------+
      * |DepartMent|Experience|Name    |Place      |
      * +----------+----------+--------+-----------+
      * |null      |null      |Ram     |RamGarh    |
      * |Operations|null      |Lakshman|LakshManPur|
      * |null      |14.0      |Sita    |SitaPur    |
      * +----------+----------+--------+-----------+
      *
      * root
      * |-- DepartMent: string (nullable = true)
      * |-- Experience: double (nullable = true)
      * |-- Name: string (nullable = true)
      * |-- Place: string (nullable = true)
      */

    val ds: Dataset[Array[dfCol]] = df.map(row => {
      row.getValuesMap[String](row.schema.map(_.name))
        .filter(_._2 != null)
        .map{f => dfCol(f._1, String.valueOf(f._2))}
        .toArray
    })
    ds.show(false)
    ds.printSchema()

    // +------------------------------------------------------------------+
    //|value                                                             |
    //+------------------------------------------------------------------+
    //|[[Name, Ram], [Place, RamGarh]]                                   |
    //|[[DepartMent, Operations], [Name, Lakshman], [Place, LakshManPur]]|
    //|[[Experience, 14.0], [Name, Sita], [Place, SitaPur]]              |
    //+------------------------------------------------------------------+
    //
    //root
    // |-- value: array (nullable = true)
    // |    |-- element: struct (containsNull = true)
    // |    |    |-- col: string (nullable = true)
    // |    |    |-- valu: string (nullable = true)
  }

  // ############################################################################################################

  @Test
  def test62511793(): Unit = {
    val data =
      """
        |Col1, Col2
        |1.19185711131486, 0.26615071205963
        |-1.3598071336738, -0.0727811733098497
        |-0.966271711572087, -0.185226008082898
        |-0.966271711572087, -0.185226008082898
        |-1.15823309349523, 0.877736754848451
        |-0.425965884412454, 0.960523044882985
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\,").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df.printSchema()
    df.show(false)
    /**
      * root
      * |-- Col1: double (nullable = true)
      * |-- Col2: double (nullable = true)
      *
      * +------------------+-------------------+
      * |Col1              |Col2               |
      * +------------------+-------------------+
      * |1.19185711131486  |0.26615071205963   |
      * |-1.3598071336738  |-0.0727811733098497|
      * |-0.966271711572087|-0.185226008082898 |
      * |-0.966271711572087|-0.185226008082898 |
      * |-1.15823309349523 |0.877736754848451  |
      * |-0.425965884412454|0.960523044882985  |
      * +------------------+-------------------+
      */

    spark.range(100).withColumn("ts",
      date_format(to_timestamp((row_number().over(Window.orderBy("id")) - 1).cast("string"),
        "mm")
        , "HH:mm:ss"))
      .show(200, false)

    /**
      * +------------------+-------------------+--------+
      * |Col1              |Col2               |ts      |
      * +------------------+-------------------+--------+
      * |-1.3598071336738  |-0.0727811733098497|00:00:00|
      * |-1.15823309349523 |0.877736754848451  |00:01:00|
      * |-0.966271711572087|-0.185226008082898 |00:02:00|
      * |-0.966271711572087|-0.185226008082898 |00:03:00|
      * |-0.425965884412454|0.960523044882985  |00:04:00|
      * |1.19185711131486  |0.26615071205963   |00:05:00|
      * +------------------+-------------------+--------+
      */
  }

  // ############################################################################################################

  @Test
  def test62518392(): Unit = {
    val df = Seq(("emp1", 100), ("emp2", 200)).toDF("ename", "sal")
    df.createOrReplaceTempView("emp")

    spark.sql(
      """
        | select ename from emp where sal=(select max(sal) from emp)
      """.stripMargin)
      .show(false)

    /**
      * +-----+
      * |ename|
      * +-----+
      * |emp2 |
      * +-----+
      */
  }

  // ############################################################################################################

  @Test
  def test62519315(): Unit = {
    val df = Seq("25-MAI-2020 13:30:30").toDF("Test")
    df.show(false)
    df.printSchema()

    /**
      * +--------------------+
      * |Test                |
      * +--------------------+
      * |25-MAI-2020 13:30:30|
      * +--------------------+
      *
      * root
      * |-- Test: string (nullable = true)
      *
      */

    def parseGermanTimeStamp(toParse: String) = {
      import java.lang.{Long => JLong}
      // improve this list for all months
      val monthTexts = java.util.Collections.singletonMap(JLong.valueOf(5), "MAI")
      val formatter = new DateTimeFormatterBuilder()
        .appendPattern("dd-")
        .appendText(ChronoField.MONTH_OF_YEAR, monthTexts)
        .appendPattern("-yyyy HH:mm:ss")
        .toFormatter(Locale.GERMANY)
      val ldt = java.time.LocalDateTime.parse(toParse, formatter)
      ldt.toString
    }

    val getTime = udf((str: String) => parseGermanTimeStamp(str))
    df.withColumn("timestamp", to_timestamp(getTime($"Test"), "yyyy-MM-dd'T'HH:mm:ss"))
      .show(false)

    /**
      * +--------------------+-------------------+
      * |Test                |timestamp          |
      * +--------------------+-------------------+
      * |25-MAI-2020 13:30:30|2020-05-25 13:30:30|
      * +--------------------+-------------------+
      */
  }
  // ############################################################################################################

  @Test
  def test62528184(): Unit = {
    val ExampleDataFrame = spark.sql("select key, contractVersion, metaData from values " +
      "('k1', 'v1', named_struct('Test', 'test1', 'DateUtc', cast(unix_timestamp() as timestamp))) " +
      "T(key, contractVersion, metaData)")
    ExampleDataFrame.show(false)
    ExampleDataFrame.printSchema()
    ExampleDataFrame.dtypes.foreach(println)
    /**
      * +---+---------------+----------------------------+
      * |key|contractVersion|metaData                    |
      * +---+---------------+----------------------------+
      * |k1 |v1             |[test1, 2020-06-23 14:39:55]|
      * +---+---------------+----------------------------+
      *
      * root
      * |-- key: string (nullable = false)
      * |-- contractVersion: string (nullable = false)
      * |-- metaData: struct (nullable = false)
      * |    |-- Test: string (nullable = false)
      * |    |-- DateUtc: timestamp (nullable = true)
      *
      * (key,StringType)
      * (contractVersion,StringType)
      * (metaData,StructType(StructField(Test,StringType,false), StructField(DateUtc,TimestampType,true)))
      */

    val isoDateFormatter = "yyyy-MM-dd'T'HH:mm:ss'Z'"
    val processedDF = ExampleDataFrame.withColumn("metaData", struct($"metaData.Test",
      date_format($"metaData.DateUtc", isoDateFormatter)))
      processedDF.show(false)

    /**
      * +---+---------------+-----------------------------+
      * |key|contractVersion|metaData                     |
      * +---+---------------+-----------------------------+
      * |k1 |v1             |[test1, 2020-06-23T14:51:17Z]|
      * +---+---------------+-----------------------------+
      */
    //    ExampleDataFrame
//      //brings back data fields with types
//      .schema
//      //Currently returning empty but works for StringType
//      .filter(_.dataType.isInstanceOf[StructType])
//      //fetch timestamp fields from struct
//      .map(_.asInstanceOf[StructType].filter(_.dataType == TimestampType))
//      //Tranforms all timestamp longs to yyyy-MM-dd'T'HH:mm:ss'Z' format
//      .foldLeft(ExampleDataFrame)((df, colName) => df.withColumn(colName, date_format(col(colName), isoDateFormatter)))
//      .show(false)
    println(Map("x" -> 10).getOrElse("y", "y not found"))
    println(Map("x" -> 10).getOrElse("y", () => throw new RuntimeException("y not found")))
  }

}
case class Foo(foo: String)
case class Bar(bar: String)

case class ExpenseEntry(
                         name: String,
                         category: String,
                         amount: BigDecimal
                       )
case class dfCol(col:String, valu:String)
