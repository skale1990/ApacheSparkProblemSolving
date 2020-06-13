package com.som.spark.learning

import java.lang.reflect.Method

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkFiles
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType
import org.testng.annotations.{BeforeClass, BeforeMethod, Test}
import org.apache.spark.ml.linalg.{Matrices, Matrix, SparseVector, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalog.{Database, Table}
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.catalyst.expressions.aggregate.ApproximatePercentile
import org.apache.spark.sql.catalyst.util.{ArrayBasedMapData, GenericArrayData, MapData}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
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

}
case class Foo(foo: String)
case class Bar(bar: String)

case class ExpenseEntry(
                         name: String,
                         category: String,
                         amount: BigDecimal
                       )
