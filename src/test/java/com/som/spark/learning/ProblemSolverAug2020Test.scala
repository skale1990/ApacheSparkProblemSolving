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
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, Transformer}
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
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.catalog.{Database, Table}
import org.apache.spark.sql.catalyst.{InternalRow, ScalaReflection}
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder, RowEncoder}
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Cast, GenericInternalRow, GenericRow, GenericRowWithSchema, MonthsBetween, Rand, Shuffle}
import org.apache.spark.sql.catalyst.expressions.aggregate.{ApproximatePercentile, Percentile}
import org.apache.spark.sql.catalyst.plans.logical.LocalRelation
import org.apache.spark.sql.catalyst.util.DateTimeUtils
import org.apache.spark.sql.expressions.{Aggregator, Window}
import org.apache.spark.sql.functions.{first, _}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geosparksql.utils.GeoSparkSQLRegistrator
import org.joda.time.{DateTimeConstants, Months}
import org.json4s.JsonAST

import scala.collection.mutable.ListBuffer
import scala.collection.{JavaConverters, mutable}
import scala.reflect.ClassTag
import scala.util.Random

class ProblemSolverAug2020Test extends Serializable {

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
  def test63213322(): Unit = {
    val data =
      """
        |col1     |  col2   |   col3 |     col4
        |null    | null    |bar      |  null
        |  null  | bar     |  null   |  null
        | null   |  null   |  null   |     kid
        | orange | null    | null    |  null
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
      * +------+----+----+----+
      * |col1  |col2|col3|col4|
      * +------+----+----+----+
      * |null  |null|bar |null|
      * |null  |bar |null|null|
      * |null  |null|null|kid |
      * |orange|null|null|null|
      * +------+----+----+----+
      *
      * root
      * |-- col1: string (nullable = true)
      * |-- col2: string (nullable = true)
      * |-- col3: string (nullable = true)
      * |-- col4: string (nullable = true)
      */

    df1.select(df1.columns.map(c => max(c).as(c)): _*)
      .show(false)
    /**
      * +------+----+----+----+
      * |col1  |col2|col3|col4|
      * +------+----+----+----+
      * |orange|bar |bar |kid |
      * +------+----+----+----+
      */

    df1.select(df1.columns.map(c => first(c, ignoreNulls = true).as(c)): _*)
      .show(false)
    /**
      * +------+----+----+----+
      * |col1  |col2|col3|col4|
      * +------+----+----+----+
      * |orange|bar |bar |kid |
      * +------+----+----+----+
      */
  }
  // ############################################################################################################
  @Test
  def test63214568(): Unit = {
    val data1 =
      """
        |salesperson1          |  salesperson2
        |Customer_17         |Customer_202
        |Customer_24         |Customer_130
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
      * +------------+------------+
      * |salesperson1|salesperson2|
      * +------------+------------+
      * |Customer_17 |Customer_202|
      * |Customer_24 |Customer_130|
      * +------------+------------+
      *
      * root
      * |-- salesperson1: string (nullable = true)
      * |-- salesperson2: string (nullable = true)
      */

    val data2 =
      """
        |Place  |Customer
        |shop  |Customer_17
        |Home  |Customer_17
        |shop  |Customer_17
        |Home  |Customer_130
        |Shop  |Customer_202
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
      * +-----+------------+
      * |Place|Customer    |
      * +-----+------------+
      * |shop |Customer_17 |
      * |Home |Customer_17 |
      * |shop |Customer_17 |
      * |Home |Customer_130|
      * |Shop |Customer_202|
      * +-----+------------+
      *
      * root
      * |-- Place: string (nullable = true)
      * |-- Customer: string (nullable = true)
      */

    val stringCol = df1.columns.map(c => s"'$c', cast(`$c` as string)").mkString(", ")
    val processedDF = df1.selectExpr(s"stack(${df1.columns.length}, $stringCol) as (Salesperson, Customer)")
    processedDF.show(false)
    /**
      * +------------+------------+
      * |Salesperson |Customer    |
      * +------------+------------+
      * |salesperson1|Customer_17 |
      * |salesperson2|Customer_202|
      * |salesperson1|Customer_24 |
      * |salesperson2|Customer_130|
      * +------------+------------+
      */

    processedDF.join(df2, Seq("Customer"), "left")
      .groupBy("Customer")
      .agg(count("Place").as("Occurance"), first("Salesperson").as("Salesperson"))
      .show(false)

    /**
      * +------------+---------+------------+
      * |Customer    |Occurance|Salesperson |
      * +------------+---------+------------+
      * |Customer_130|1        |salesperson2|
      * |Customer_17 |3        |salesperson1|
      * |Customer_202|1        |salesperson2|
      * |Customer_24 |0        |salesperson1|
      * +------------+---------+------------+
      */
  }
  // ############################################################################################################
  @Test
  def test63222853(): Unit = {
    val df = spark.sql(
      """
        |select col1, col2
        |from values
        | (array(1, 2), 'a'),
        | (array(1, 2, 3), 'b')
        | T(col1, col2)
      """.stripMargin)
    df.show(false)
    df.printSchema()

    /**
      * +---------+----+
      * |col1     |col2|
      * +---------+----+
      * |[1, 2]   |a   |
      * |[1, 2, 3]|b   |
      * +---------+----+
      *
      * root
      * |-- col1: array (nullable = false)
      * |    |-- element: integer (containsNull = false)
      * |-- col2: string (nullable = false)
      */

    // alternative-1
    df.withColumn("col3", expr("array_repeat(col2, size(col1))"))
      .show(false)

    /**
      * +---------+----+---------+
      * |col1     |col2|col3     |
      * +---------+----+---------+
      * |[1, 2]   |a   |[a, a]   |
      * |[1, 2, 3]|b   |[b, b, b]|
      * +---------+----+---------+
      */

    // alternative-2
    df.withColumn("col3", expr(s"TRANSFORM(col1, x -> col2)"))
      .show(false)

    /**
      * +---------+----+---------+
      * |col1     |col2|col3     |
      * +---------+----+---------+
      * |[1, 2]   |a   |[a, a]   |
      * |[1, 2, 3]|b   |[b, b, b]|
      * +---------+----+---------+
      */
  }
  // ############################################################################################################
  @Test
  def test63222716(): Unit = {
    def getType[T: scala.reflect.runtime.universe.TypeTag](obj: T) = scala.reflect.runtime.universe.typeOf[T]
    val path = getClass.getResource("/csv/employee.txt").getPath
    val ds = spark.read
      .schema(ScalaReflection.schemaFor[Employee].dataType.asInstanceOf[StructType])
      .option("header", true)
      .option("sep", ",")
      .csv(path)
      .as[Employee]
    println(getType(ds))
    /**
      * org.apache.spark.sql.Dataset[com.som.spark.learning.Employee]
      */
    ds.show(false)
    ds.printSchema()
    /**
      * +------+---+
      * |name  |age|
      * +------+---+
      * |John  |28 |
      * |Andrew|36 |
      * |Clarke|22 |
      * |Kevin |42 |
      * +------+---+
      *
      * root
      * |-- name: string (nullable = true)
      * |-- age: long (nullable = true)
      */

  }
  // ############################################################################################################
  case class MySum(colName: String) extends Aggregator[Row, Double, Double] {

    def zero: Double = 0d
    def reduce(acc: Double, row: Row): Double = acc * 0.78 + 0.21 * row.getAs[Double](colName)

    def merge(acc1: Double, acc2: Double): Double = acc1 + acc2
    def finish(acc: Double): Double = acc

    def bufferEncoder: Encoder[Double] = Encoders.scalaDouble
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }

  @Test
  def test63219615(): Unit = {
    val df = Seq(
      (1L, 2.569d),
      (2L, 5.89d),
      (3L, 4.28d),
      (4L, 2.15d),
      (5L, 6.43d),
      (6L, 8.92d),
      (7L, 5.86d),
      (8L, 1.65d),
      (9L, 2.28d)
    ).toDF("order", "price")

    val win = Window.orderBy("order").rowsBetween(0, 1)
//    df.withColumn("new_column", MySum("price").toColumn.over(win))
//      .show(false)
    df.as[(Long, Double)].withColumn("col3", sum("price").over(win))
      .show(false)
  }
  // ############################################################################################################
  @Test
  def test63227934(): Unit = {
    val data =
      """
        |type                 |column_ref                                            |table_object
        |foreignKeyColumn    |   FRED.FRED.BACHELOR_DEGREE_OR_HIGHER.REGION_CODE   |    FRED.FRED.US_REGIONS
        |primaryKeyColumn    |   FRED.FRED.US_REGIONS.REGION_CODE                  |    FRED.FRED.US_REGIONS
        |foreignKeyColumn    |   FRED.FRED.MEAN_REAL_WAGES_COLA.REGION_CODE        |    FRED.FRED.US_REGIONS
        |primaryKeyColumn    |   FRED.FRED.US_REGIONS.REGION_CODE                  |    FRED.FRED.US_REGIONS
        |foreignKeyColumn    |   FRED.FRED.PER_CAPITA_PERSONAL_INCOME.REGION_CODE  |    FRED.FRED.US_REGIONS
        |primaryKeyColumn    |   FRED.FRED.US_REGIONS.REGION_CODE                  |    FRED.FRED.US_REGIONS
        |foreignKeyColumn    |   FRED.FRED.HOMEOWNERSHIP_RATE.REGION_CODE          |    FRED.FRED.US_REGIONS
        |primaryKeyColumn    |   FRED.FRED.US_REGIONS.REGION_CODE                  |    FRED.FRED.US_REGIONS
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
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
      * +----------------+------------------------------------------------+--------------------+
      * |type            |column_ref                                      |table_object        |
      * +----------------+------------------------------------------------+--------------------+
      * |foreignKeyColumn|FRED.FRED.BACHELOR_DEGREE_OR_HIGHER.REGION_CODE |FRED.FRED.US_REGIONS|
      * |primaryKeyColumn|FRED.FRED.US_REGIONS.REGION_CODE                |FRED.FRED.US_REGIONS|
      * |foreignKeyColumn|FRED.FRED.MEAN_REAL_WAGES_COLA.REGION_CODE      |FRED.FRED.US_REGIONS|
      * |primaryKeyColumn|FRED.FRED.US_REGIONS.REGION_CODE                |FRED.FRED.US_REGIONS|
      * |foreignKeyColumn|FRED.FRED.PER_CAPITA_PERSONAL_INCOME.REGION_CODE|FRED.FRED.US_REGIONS|
      * |primaryKeyColumn|FRED.FRED.US_REGIONS.REGION_CODE                |FRED.FRED.US_REGIONS|
      * |foreignKeyColumn|FRED.FRED.HOMEOWNERSHIP_RATE.REGION_CODE        |FRED.FRED.US_REGIONS|
      * |primaryKeyColumn|FRED.FRED.US_REGIONS.REGION_CODE                |FRED.FRED.US_REGIONS|
      * +----------------+------------------------------------------------+--------------------+
      *
      * root
      * |-- type: string (nullable = true)
      * |-- column_ref: string (nullable = true)
      * |-- table_object: string (nullable = true)
      */
    val p = df2
      .groupBy("table_object")
      .pivot("type")
      .agg(collect_list("column_ref"))

    p
      .withColumn("x", arrays_zip($"foreignKeyColumn", $"primaryKeyColumn"))
      .selectExpr("table_object", "inline_outer(x)" )
      .show(false)

    /**
      * +--------------------+------------------------------------------------+--------------------------------+
      * |table_object        |foreignKeyColumn                                |primaryKeyColumn                |
      * +--------------------+------------------------------------------------+--------------------------------+
      * |FRED.FRED.US_REGIONS|FRED.FRED.BACHELOR_DEGREE_OR_HIGHER.REGION_CODE |FRED.FRED.US_REGIONS.REGION_CODE|
      * |FRED.FRED.US_REGIONS|FRED.FRED.MEAN_REAL_WAGES_COLA.REGION_CODE      |FRED.FRED.US_REGIONS.REGION_CODE|
      * |FRED.FRED.US_REGIONS|FRED.FRED.PER_CAPITA_PERSONAL_INCOME.REGION_CODE|FRED.FRED.US_REGIONS.REGION_CODE|
      * |FRED.FRED.US_REGIONS|FRED.FRED.HOMEOWNERSHIP_RATE.REGION_CODE        |FRED.FRED.US_REGIONS.REGION_CODE|
      * +--------------------+------------------------------------------------+--------------------------------+
      */
  }
  // ############################################################################################################
  @Test
  def test63228005(): Unit = {
    val data =
      """
        |           time
        |10:59:46.000 AM
        | 6:26:36.000 PM
        |11:13:38.000 PM
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
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

    df2.withColumn("new_time", to_timestamp(lpad(col("time"), 15, "0"),"hh:mm:ss.SSS a"))
      .show(false)

    /**
      *  For formatting, the number of pattern letters is the minimum number of digits, and shorter numbers are zero-padded to this amount.
      *  For parsing, the number of pattern letters is ignored unless it's needed to separate two adjacent fields.
      */
    df2.withColumn("new_time", to_timestamp(col("time"),"h:mm:ss.SSS a"))
      .show(false)
  }

  // ############################################################################################################
  @Test
  def test63233044(): Unit = {
    val df = Seq("20,00:0  Leagu+es Under the Sea").toDF("Title")
    df.withColumn("Title_Words", split(col("Title"), "\\s+|[,:]"))
      .show(false)
  }

  // ############################################################################################################
  @Test
  def test63238732(): Unit = {

    //data
    val dfA = spark.createDataFrame(Seq(
      (0, Vectors.sparse(6, Seq((0, 1.0), (1, 1.0), (2, 1.0)))),
      (1, Vectors.sparse(6, Seq((2, 1.0), (3, 1.0), (4, 1.0)))),
      (2, Vectors.sparse(6, Seq((0, 1.0), (2, 1.0), (4, 1.0))))
    )).toDF("id", "features")
    dfA.show(false)

    /**
      * +---+-------------------------+
      * |id |features                 |
      * +---+-------------------------+
      * |0  |(6,[0,1,2],[1.0,1.0,1.0])|
      * |1  |(6,[2,3,4],[1.0,1.0,1.0])|
      * |2  |(6,[0,2,4],[1.0,1.0,1.0])|
      * +---+-------------------------+
      */

    // Alternative-1
    //udf
    val feature_idx_to_wipe = Array(1, 2)
    import org.apache.spark.ml.linalg.{SparseVector => NewSparseVector}
    def wipe(v: NewSparseVector, idx2clean:Seq[Int]) : NewSparseVector = {
      val lb:ListBuffer[(Int, Double)]=ListBuffer()
      v.foreachActive {
        case (i, v) =>
          if(!idx2clean.contains(i)){
            lb += ((i, v))
          }
      }

      Vectors.sparse(v.size, lb.toSeq).toSparse
    }
    val udf_wipe = udf((x: NewSparseVector, idx2clean:Seq[Int]) => wipe(x, idx2clean))

    //apply udf
    val newDF = dfA.withColumn("features_wiped", udf_wipe(col("features"), lit(feature_idx_to_wipe)))

    //target (a new column of vector added, with features at index 1,2 are removed)
    newDF.select("id","features_wiped").show(false)
    /**
      * +---+-------------------+
      * |id |features_wiped     |
      * +---+-------------------+
      * |0  |(6,[0],[1.0])      |
      * |1  |(6,[3,4],[1.0,1.0])|
      * |2  |(6,[0,4],[1.0,1.0])|
      * +---+-------------------+
      */

//    Alternative2
    //data
    val feature_idx_to_wipe1 = Set(1, 2)
    val broabcastSet = spark.sparkContext.broadcast(feature_idx_to_wipe1)

    //udf
    import org.apache.spark.ml.linalg.{SparseVector => NewSparseVector}
    def wipe1(v: NewSparseVector) : NewSparseVector = {
      val idx2clean = broabcastSet.value
      val lb:ListBuffer[(Int, Double)]=ListBuffer()
      v.foreachActive {
        case (i, v) =>
          if(!idx2clean.contains(i)){
            lb += ((i, v))
          }
      }

      Vectors.sparse(v.size, lb.toSeq).toSparse
    }
    val udf_wipe1 = udf((x: NewSparseVector) => wipe1(x))

    //apply udf
    val newDF1 = dfA.withColumn("features_wiped", udf_wipe1(col("features")))

    //target (a new column of vector added, with features at index 1,2 are removed)
    newDF1.select("id","features_wiped").show(false)

    /**
      * +---+-------------------+
      * |id |features_wiped     |
      * +---+-------------------+
      * |0  |(6,[0],[1.0])      |
      * |1  |(6,[3,4],[1.0,1.0])|
      * |2  |(6,[0,4],[1.0,1.0])|
      * +---+-------------------+
      */
  }

  // ############################################################################################################
  @Test
  def test63240993(): Unit = {
    val data1 =
      """
        |Salesperson_21: Customer_575,Customer_2703,Customer_2682,Customer_2615
        |Salesperson_11: Customer_454,Customer_158,Customer_1859,Customer_2605
        |Salesperson_10: Customer_1760,Customer_613,Customer_3008,Customer_1265
        |Salesperson_4: Customer_1545,Customer_1312,Customer_861,Customer_2178
      """.stripMargin
//    val stringDS1 = data1.split(System.lineSeparator())
//      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
//      .toSeq.toDS()
    val df1 = spark.read.schema("Salesperson STRING, Customer STRING")
      .option("sep", ":")
      .csv(data1.split(System.lineSeparator()).toSeq.toDS())
    df1.show(false)
    df1.printSchema()
    /**
      * +--------------+-------------------------------------------------------+
      * |Salesperson   |Customer                                               |
      * +--------------+-------------------------------------------------------+
      * |Salesperson_21| Customer_575,Customer_2703,Customer_2682,Customer_2615|
      * |Salesperson_11| Customer_454,Customer_158,Customer_1859,Customer_2605 |
      * |Salesperson_10| Customer_1760,Customer_613,Customer_3008,Customer_1265|
      * |Salesperson_4 | Customer_1545,Customer_1312,Customer_861,Customer_2178|
      * +--------------+-------------------------------------------------------+
      *
      * root
      * |-- Salesperson: string (nullable = true)
      * |-- Customer: string (nullable = true)
      */


    val data2 =
      """
        |Type  |Customer
        |shop  |Customer_17
        |Home  |Customer_2703
        |shop  |Customer_2703
        |Home  |Customer_575
        |Shop  |Customer_202
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
      * +----+-------------+
      * |Type|Customer     |
      * +----+-------------+
      * |shop|Customer_17  |
      * |Home|Customer_2703|
      * |shop|Customer_2703|
      * |Home|Customer_575 |
      * |Shop|Customer_202 |
      * +----+-------------+
      *
      * root
      * |-- Type: string (nullable = true)
      * |-- Customer: string (nullable = true)
      */

    val processedDF = df1.withColumn("Customer", explode(split(trim(col("Customer")), ",")))
     processedDF.show(false)
    /**
      * +--------------+-------------+
      * |Salesperson   |Customer     |
      * +--------------+-------------+
      * |Salesperson_21|Customer_575 |
      * |Salesperson_21|Customer_2703|
      * |Salesperson_21|Customer_2682|
      * |Salesperson_21|Customer_2615|
      * |Salesperson_11|Customer_454 |
      * |Salesperson_11|Customer_158 |
      * |Salesperson_11|Customer_1859|
      * |Salesperson_11|Customer_2605|
      * |Salesperson_10|Customer_1760|
      * |Salesperson_10|Customer_613 |
      * |Salesperson_10|Customer_3008|
      * |Salesperson_10|Customer_1265|
      * |Salesperson_4 |Customer_1545|
      * |Salesperson_4 |Customer_1312|
      * |Salesperson_4 |Customer_861 |
      * |Salesperson_4 |Customer_2178|
      * +--------------+-------------+
      */

    processedDF.join(df2, Seq("Customer"), "left")
      .groupBy("Customer")
      .agg(count("Type").as("Occurance"), first("Salesperson").as("Salesperson"))
      .show(false)

    /**
      * +-------------+---------+--------------+
      * |Customer     |Occurance|Salesperson   |
      * +-------------+---------+--------------+
      * |Customer_1312|0        |Salesperson_4 |
      * |Customer_1545|0        |Salesperson_4 |
      * |Customer_1760|0        |Salesperson_10|
      * |Customer_2682|0        |Salesperson_21|
      * |Customer_2703|2        |Salesperson_21|
      * |Customer_3008|0        |Salesperson_10|
      * |Customer_454 |0        |Salesperson_11|
      * |Customer_613 |0        |Salesperson_10|
      * |Customer_1265|0        |Salesperson_10|
      * |Customer_158 |0        |Salesperson_11|
      * |Customer_1859|0        |Salesperson_11|
      * |Customer_2178|0        |Salesperson_4 |
      * |Customer_2605|0        |Salesperson_11|
      * |Customer_2615|0        |Salesperson_21|
      * |Customer_575 |1        |Salesperson_21|
      * |Customer_861 |0        |Salesperson_4 |
      * +-------------+---------+--------------+
      */
  }
  // ############################################################################################################
  @Test
  def test63241674(): Unit = {
    val df = spark.sql("select '\"\"\"56\"7\"' as test")
    df.show(false)
    /**
      * +--------+
      * |test    |
      * +--------+
      * |"""56"7"|
      * +--------+
      */
    df.createOrReplaceTempView("table")
    spark.sql("select test, regexp_replace(test, '^\"+|\"+$', '') as test_new from table")
      .show(false)

    /**
      * +--------+--------+
      * |test    |test_new|
      * +--------+--------+
      * |"""56"7"|56"7    |
      * +--------+--------+
      */
  }
  // ############################################################################################################
  @Test
  def test63247985(): Unit = {
    val data = Seq(
      """{"Data": [{ "name": "FName", "value": "Alex" }, { "name": "LName",   "value": "Strong"  }]}""",
      """{"Data": [{ "name": "FName", "value": "Robert " }, { "name": "MName",   "value": "Nesta "  }, {
        |"name": "LName",   "value": "Marley"  }]}""".stripMargin
    )
    val df = spark.read
      .json(data.toDS())
    df.show(false)
    df.printSchema()

//
//    +----------------------------------------------------+
//    |Data                                                |
//    +----------------------------------------------------+
//    |[[FName, Alex], [LName, Strong]]                    |
//    |[[FName, Robert ], [MName, Nesta ], [LName, Marley]]|
//    +----------------------------------------------------+
//
//    root
//    |-- Data: array (nullable = true)
//    |    |-- element: struct (containsNull = true)
//    |    |    |-- name: string (nullable = true)
//    |    |    |-- value: string (nullable = true)
//

    df.selectExpr("inline_outer(Data)")
      .groupBy()
      .pivot("name")
      .agg(collect_list("value"))
      .withColumn("x", arrays_zip($"FName", $"LName"))
      .selectExpr("inline_outer(x)")
      .show(false)

    /**
      * +-------+------+
      * |FName  |LName |
      * +-------+------+
      * |Alex   |Strong|
      * |Robert |Marley|
      * +-------+------+
      */
  }
  // ############################################################################################################
  @Test
  def test63252910(): Unit = {
    val data =
      """
        |very_hot|  hot| cold|little_snow|medium_snow|very_cold|deep_snow|freezing|windy
        |    True|False|False|      False|      False|    False|    False|   False| True
        |   False|False| True|       True|      False|    False|    False|   False|False
        |   False|False| True|      False|       True|    False|    False|   False|False
        |   False|False|False|      False|      False|     True|     True|   False|False
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", ",")
//      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS2)
    df2.show(false)
    df2.printSchema()
    /**
      * +--------+-----+-----+-----------+-----------+---------+---------+--------+-----+
      * |very_hot|hot  |cold |little_snow|medium_snow|very_cold|deep_snow|freezing|windy|
      * +--------+-----+-----+-----------+-----------+---------+---------+--------+-----+
      * |True    |False|False|False      |False      |False    |False    |False   |True |
      * |False   |False|True |True       |False      |False    |False    |False   |False|
      * |False   |False|True |False      |True       |False    |False    |False   |False|
      * |False   |False|False|False      |False      |True     |True     |False   |False|
      * +--------+-----+-----+-----------+-----------+---------+---------+--------+-----+
      *
      * root
      * |-- very_hot: string (nullable = true)
      * |-- hot: string (nullable = true)
      * |-- cold: string (nullable = true)
      * |-- little_snow: string (nullable = true)
      * |-- medium_snow: string (nullable = true)
      * |-- very_cold: string (nullable = true)
      * |-- deep_snow: string (nullable = true)
      * |-- freezing: string (nullable = true)
      * |-- windy: string (nullable = true)
      */

    val columns = df2.columns.map(c => s"named_struct('name', '$c', 'value', `$c`)").mkString(", ")
    df2.selectExpr(s"TRANSFORM(FILTER(array($columns), x -> x.value='True'), x -> x.name) as features")
      .show(false)
    /**
      * +----------------------+
      * |features              |
      * +----------------------+
      * |[very_hot, windy]     |
      * |[cold, little_snow]   |
      * |[cold, medium_snow]   |
      * |[very_cold, deep_snow]|
      * +----------------------+
      */
  }
  // ############################################################################################################
  @Test
  def test63264973(): Unit = {
    val df1=Seq(
      ("1_2_3","5_10"),
      ("4_5_6","15_20")
    )toDF("c1","c2")
    df1.show(false)

    df1.withColumn("res",
      expr("concat_ws('_', zip_with(split(c1, '_'), split(c2, '_'), (x, y) -> cast(x+y as int)))"))
      .show(false)

    /**
      * +-----+-----+-----+
      * |c1   |c2   |res  |
      * +-----+-----+-----+
      * |1_2_3|5_10 |6_12 |
      * |4_5_6|15_20|19_25|
      * +-----+-----+-----+
      */

    val end = 51 // 50 cols
    val df = spark.sql("select '1_2_3' as c1")
    val new_df = Range(2, end).foldLeft(df){(df, i) => df.withColumn(s"c$i", $"c1")}
    new_df.show(false)
    /**
      * +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      * |c1   |c2   |c3   |c4   |c5   |c6   |c7   |c8   |c9   |c10  |c11  |c12  |c13  |c14  |c15  |c16  |c17  |c18  |c19  |c20  |c21  |c22  |c23  |c24  |c25  |c26  |c27  |c28  |c29  |c30  |c31  |c32  |c33  |c34  |c35  |c36  |c37  |c38  |c39  |c40  |c41  |c42  |c43  |c44  |c45  |c46  |c47  |c48  |c49  |c50  |
      * +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      * |1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|
      * +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
      */
    val res = new_df.withColumn("res", $"c1")
    Range(2, end).foldLeft(res){(df4, i) =>
      df4.withColumn("res",
        expr(s"concat_ws('_', zip_with(split(res, '_'), split(${s"c$i"}, '_'), (x, y) -> cast(x+y as int)))"))
    }
      .show(false)
    /**
      * +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+----------+
      * |c1   |c2   |c3   |c4   |c5   |c6   |c7   |c8   |c9   |c10  |c11  |c12  |c13  |c14  |c15  |c16  |c17  |c18  |c19  |c20  |c21  |c22  |c23  |c24  |c25  |c26  |c27  |c28  |c29  |c30  |c31  |c32  |c33  |c34  |c35  |c36  |c37  |c38  |c39  |c40  |c41  |c42  |c43  |c44  |c45  |c46  |c47  |c48  |c49  |c50  |res       |
      * +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+----------+
      * |1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|1_2_3|50_100_150|
      * +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+----------+
      */
  }

  // ############################################################################################################
  @Test
  def test63264183(): Unit = {
    // 63264183
    val df_input = Seq( ("p1", """{"a": 1, "b": 2}"""), ("p2", """{"c": 3}""") )
      .toDF("p_id", "p_meta")
    df_input.show(false)
    /**
      * +----+----------------+
      * |p_id|p_meta          |
      * +----+----------------+
      * |p1  |{"a": 1, "b": 2}|
      * |p2  |{"c": 3}        |
      * +----+----------------+
      */

    df_input.withColumn("p_meta", from_json($"p_meta", "map<string, string>", Map.empty[String, String]))
      .selectExpr("p_id", "explode(p_meta) as (p_meta_key, p_meta_value)")
      .show(false)
    /**
      * +----+----------+------------+
      * |p_id|p_meta_key|p_meta_value|
      * +----+----------+------------+
      * |p1  |a         |1           |
      * |p1  |b         |2           |
      * |p2  |c         |3           |
      * +----+----------+------------+
      */
  }

  // ############################################################################################################
  @Test
  def test63277906(): Unit = {
    val data =
      """
        |col1 | col2
        |1  | 2020-02-27 15:00:00
        |1  | 2020-02-27 15:04:00
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
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
      * +----+-------------------+
      * |col1|col2               |
      * +----+-------------------+
      * |1   |2020-02-27 15:00:00|
      * |1   |2020-02-27 15:04:00|
      * +----+-------------------+
      *
      * root
      * |-- col1: integer (nullable = true)
      * |-- col2: timestamp (nullable = true)
      */

    val w = Window.partitionBy("col1")
    df2.withColumn("col3",
      when(max("col2").over(w).cast("long") - col("col2").cast("long")=== 0, "Y")
    )
      .show(false)

    /**
      * +----+-------------------+----+
      * |col1|col2               |col3|
      * +----+-------------------+----+
      * |1   |2020-02-27 15:00:00|null|
      * |1   |2020-02-27 15:04:00|Y   |
      * +----+-------------------+----+
      */

    df2.createOrReplaceTempView("table")
    spark.sql(
      """
        | select col1, col2,
        |   case when (cast(max(col2) over (partition by col1) as long) - cast(col2 as long) = 0) then 'Y' end as col3
        | from table
      """.stripMargin)
      .show(false)

    /**
      * +----+-------------------+----+
      * |col1|col2               |col3|
      * +----+-------------------+----+
      * |1   |2020-02-27 15:00:00|null|
      * |1   |2020-02-27 15:04:00|Y   |
      * +----+-------------------+----+
      */
  }

  // ############################################################################################################
  def getBinaryDF: DataFrame = {
    val sparkSession = SparkSession.builder
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName)
      .master("local[*]")
      .appName("myGeoSparkSQLdemo")
      .getOrCreate

    // register all functions from geospark-sql_2.3 to sparkSession
    GeoSparkSQLRegistrator.registerAll(sparkSession)
    val implicits = sparkSession.implicits
    import implicits._
    implicit def int2Byte(i: Int) = i.toByte
    val bytes: Array[Byte] = Array(0x00, 0x00, 0x00, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x11,
      0x00, 0x00, 0x00, 0x04, 0x00, 0xF0, 0x00, 0xDC, 0xCC, 0x1A, 0xC0, 0x87, 0x14, 0x01, 0x81, 0x1E, 0x1B, 0x41,
      0x40, 0xFC, 0xFF, 0xEF, 0x00, 0x68, 0xAA, 0x1A, 0xC0, 0xBF, 0xEE, 0x57, 0x20, 0x85, 0x19, 0x41, 0x40, 0x04,
      0x00, 0xF0, 0x00, 0x8C, 0x86, 0x1A, 0xC0, 0xCC, 0xDC, 0x8B, 0xDC, 0xAE, 0x1A, 0x41, 0x40, 0xFF, 0xFF, 0xEF,
      0x00, 0x44, 0x74, 0x1A, 0xC0, 0xCA, 0x9D, 0x5D, 0x61, 0x10, 0x1C, 0x41, 0x40, 0xFF, 0xFF, 0xEF, 0x00, 0x64, 0x63,
      0x1A, 0xC0, 0xBF, 0x1F, 0x98, 0x0B, 0x3A, 0x1D, 0x41, 0x40, 0xFF, 0xFF, 0xEF, 0x00, 0x44, 0x47, 0x1A, 0xC0, 0xE4,
      0x6B, 0xA0, 0xDD, 0xCE, 0x1D, 0x41, 0x40, 0xFC, 0xFF, 0xEF, 0x00, 0xD8, 0x2B, 0x1A, 0xC0, 0x54, 0xE4, 0x71, 0x67,
      0x6D, 0x1C, 0x41, 0x40, 0xFF, 0xFF, 0xEF, 0x00, 0x44, 0x1A, 0x1A, 0xC0, 0xBF, 0x1F, 0x98, 0x0B, 0x3A, 0x1D, 0x41,
      0x40, 0x02, 0x00, 0xF0, 0x00, 0x80, 0x0B, 0x1A, 0xC0, 0x0D, 0x80, 0x00, 0x13, 0x2F, 0x23, 0x41, 0x40, 0x02, 0x00,
      0xF0, 0x00, 0xB0, 0x35, 0x1A, 0xC0, 0xCC, 0xF6, 0x23, 0xF8, 0xBD, 0x26, 0x41, 0x40, 0x04, 0x00, 0xF0, 0x00, 0x0C,
      0x43, 0x1A, 0xC0, 0x73, 0x1A, 0x44, 0xAF, 0x16, 0x26, 0x41, 0x40, 0x02, 0x00, 0xF0, 0x00, 0x40, 0x5A, 0x1A, 0xC0,
      0xFF, 0x54, 0x9C, 0x7C, 0x2D, 0x27, 0x41, 0x40, 0x02, 0x00, 0xF0, 0x00, 0x50, 0x68, 0x1A, 0xC0, 0x87, 0x6E, 0xB9,
      0x42, 0x44, 0x28, 0x41, 0x40, 0x02, 0x00, 0xF0, 0x00, 0x00, 0x7C, 0x1A, 0xC0, 0x78, 0x2B, 0x85, 0xBA, 0xF5, 0x26,
      0x41, 0x40, 0xFC, 0xFF, 0xEF, 0x00, 0x18, 0x91, 0x1A, 0xC0, 0x49, 0x96, 0x6F, 0x58, 0xC6, 0x28, 0x41, 0x40, 0x02,
      0x00, 0xF0, 0x00, 0xB0, 0xBC, 0x1A, 0xC0, 0x91, 0xFA, 0x4B, 0x0E, 0x7F, 0x20, 0x41, 0x40, 0x04, 0x00, 0xF0, 0x00,
      0xDC, 0xCC, 0x1A, 0xC0, 0x87, 0x14, 0x01, 0x81, 0x1E, 0x1B, 0x41, 0x40)

    println(bytes.map(_.toString).mkString(" === "))
    println(UTF8String.fromBytes(bytes))
    Seq(bytes).toDF("BinaryGeometry")
  }

  // ############################################################################################################
  @Test
  def test63287196(): Unit = {
    val data =
      """
        |Country/Region| 3/7/20| 3/8/20| 3/9/20|3/10/20|3/11/20|3/12/20|3/13/20
        |       Senegal|      0|      4|     10|     18|     27|     31|     35
        |       Tunisia|      1|      8|     15|     21|     37|     42|     59
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
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
      * +--------------+------+------+------+-------+-------+-------+-------+
      * |Country/Region|3/7/20|3/8/20|3/9/20|3/10/20|3/11/20|3/12/20|3/13/20|
      * +--------------+------+------+------+-------+-------+-------+-------+
      * |Senegal       |0     |4     |10    |18     |27     |31     |35     |
      * |Tunisia       |1     |8     |15    |21     |37     |42     |59     |
      * +--------------+------+------+------+-------+-------+-------+-------+
      *
      * root
      * |-- Country/Region: string (nullable = true)
      * |-- 3/7/20: integer (nullable = true)
      * |-- 3/8/20: integer (nullable = true)
      * |-- 3/9/20: integer (nullable = true)
      * |-- 3/10/20: integer (nullable = true)
      * |-- 3/11/20: integer (nullable = true)
      * |-- 3/12/20: integer (nullable = true)
      * |-- 3/13/20: integer (nullable = true)
      */

    val new_df = df2.withColumn("01/01/70", lit(0))
    val tuples = new_df.schema.filter(_.dataType.isInstanceOf[NumericType])
      .map(_.name)
      .map(c => {
      val sdf = new SimpleDateFormat("MM/dd/yy")
      (sdf.parse(c), c)
    }).sortBy(_._1)
      .map(_._2)
      .sliding(2, 1)
      .map(seq => (col(seq.last) - col(seq.head)).as(seq.last))

    new_df.select(col("Country/Region") +: tuples.toSeq: _* )
      .show(false)

    /**
      * +--------------+------+------+------+-------+-------+-------+-------+
      * |Country/Region|3/7/20|3/8/20|3/9/20|3/10/20|3/11/20|3/12/20|3/13/20|
      * +--------------+------+------+------+-------+-------+-------+-------+
      * |Senegal       |0     |4     |6     |8      |9      |4      |4      |
      * |Tunisia       |1     |7     |7     |6      |16     |5      |17     |
      * +--------------+------+------+------+-------+-------+-------+-------+
      */


  }
  // ############################################################################################################
  @Test
  def test63295804(): Unit = {
    val df = Seq((1, 2, "hi", "hello")).toDF()

    df.selectExpr("max(bit_length(concat_ws('', *)))/8 as bytes")
      .show(false)
    /**
      * +-----+
      * |bytes|
      * +-----+
      * |9.0  |
      * +-----+
      */


    df.selectExpr("concat_ws('', *)")
      .write
      .mode(SaveMode.Overwrite)
      .text("/Users/sokale/models/test63295804")
  }
  // ############################################################################################################
  @Test
  def test63310735(): Unit = {
    // I think in pyspark you can easily do that with help of pipeline.
    // 1. convert each of the pipe function into the transformer. There are some predefined transformers that spark
    // provides, we can make use of that also
    // 2. Create pipeline using the transformers
    // 3. Run the pipeline

    // Example
    val df = Seq(("a", 1), ("b", 2), ("c", 3)).toDF("col1", "col2")
    df.show(false)
    df.printSchema()
    /**
      * +----+----+
      * |col1|col2|
      * +----+----+
      * |a   |1   |
      * |b   |2   |
      * |c   |3   |
      * +----+----+
      *
      * root
      * |-- col1: string (nullable = true)
      * |-- col2: integer (nullable = false)
      */

    // for .pipe(lambda df: df[df.col1 != 'a']), we can easily use spark SQLTransformer as below-
    val transform1 = new SQLTransformer()
      .setStatement("select * from __THIS__ where col1 != 'a'")
    val transform2 = new SQLTransformer()
      .setStatement("select col1, col2, SQRT(col2) as col3 from __THIS__")

    val pipeline = new Pipeline()
      .setStages(Array(transform1, transform2))

    // Use current pipeline to transform the given dataframe
    pipeline.fit(df).transform(df)
      .show(false)

    /**
      * +----+----+------------------+
      * |col1|col2|col3              |
      * +----+----+------------------+
      * |b   |2   |1.4142135623730951|
      * |c   |3   |1.7320508075688772|
      * +----+----+------------------+
      */

  }

  // ############################################################################################################
  @Test
  def test63308854(): Unit = {
    val data =
      """
        |Name | Id | Month
        |Mark | 01 | 2020-01-02
        |Aana | 12 | 2020-01-02
      """.stripMargin
    val stringDS2 = data.split(System.lineSeparator())
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
      * +----+---+-------------------+
      * |Name|Id |Month              |
      * +----+---+-------------------+
      * |Mark|1  |2020-01-02 00:00:00|
      * |Aana|12 |2020-01-02 00:00:00|
      * +----+---+-------------------+
      *
      * root
      * |-- Name: string (nullable = true)
      * |-- Id: integer (nullable = true)
      * |-- Month: timestamp (nullable = true)
      */

    df2.withColumn("months_to_add", explode(sequence(lit(0), lit(2))))
      .withColumn("Month", expr("add_months(Month, months_to_add)"))
      .show(false)

    /**
      * +----+---+----------+-------------+
      * |Name|Id |Month     |months_to_add|
      * +----+---+----------+-------------+
      * |Mark|1  |2020-01-02|0            |
      * |Mark|1  |2020-02-02|1            |
      * |Mark|1  |2020-03-02|2            |
      * |Aana|12 |2020-01-02|0            |
      * |Aana|12 |2020-02-02|1            |
      * |Aana|12 |2020-03-02|2            |
      * +----+---+----------+-------------+
      */
  }
  // ############################################################################################################
  @Test
  def test63317105(): Unit = {
    val data =
      """
        | Country| 3/7/20| 3/8/20| 3/9/20|3/10/20|3/11/20|3/12/20|3/13/20
        |   Japan|      0|      4|     10|     18|     27|     31|     35
      """.stripMargin

    val stringDS2 = data.split(System.lineSeparator())
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
      * +-------+------+------+------+-------+-------+-------+-------+
      * |Country|3/7/20|3/8/20|3/9/20|3/10/20|3/11/20|3/12/20|3/13/20|
      * +-------+------+------+------+-------+-------+-------+-------+
      * |Japan  |0     |4     |10    |18     |27     |31     |35     |
      * +-------+------+------+------+-------+-------+-------+-------+
      *
      * root
      * |-- Country: string (nullable = true)
      * |-- 3/7/20: integer (nullable = true)
      * |-- 3/8/20: integer (nullable = true)
      * |-- 3/9/20: integer (nullable = true)
      * |-- 3/10/20: integer (nullable = true)
      * |-- 3/11/20: integer (nullable = true)
      * |-- 3/12/20: integer (nullable = true)
      * |-- 3/13/20: integer (nullable = true)
      */
    val stringCol = df2.columns.map(c => s"'$c', cast(`$c` as string)").mkString(", ")
    val processedDF = df2.selectExpr(s"stack(${df2.columns.length}, $stringCol) as (col_1, col_2)")
    processedDF.show(false)
    /**
      * +-------+-----+
      * |col_1  |col_2|
      * +-------+-----+
      * |Country|Japan|
      * |3/7/20 |0    |
      * |3/8/20 |4    |
      * |3/9/20 |10   |
      * |3/10/20|18   |
      * |3/11/20|27   |
      * |3/12/20|31   |
      * |3/13/20|35   |
      * +-------+-----+
      */
  }
  // ############################################################################################################
  @Test
  def test63321887(): Unit = {
    val ddlSchema =
      """
        |x struct<
        |	_id: struct<
        |		oid: string
        |	>,
        |	id: string,
        |	sparse_rep: struct<
        |		`1`: double,
        |		`10`: double,
        |		`11`: double,
        |		`12`: double,
        |		`13`: double,
        |		`14`: double,
        |		`15`: double,
        |		`17`: double,
        |		`18`: double,
        |		`2`: double,
        |		`20`: double,
        |		`21`: double,
        |		`22`: double,
        |		`23`: double,
        |		`24`: double,
        |		`25`: double,
        |		`26`: double,
        |		`27`: double,
        |		`3`: double,
        |		`4`: double,
        |		`7`: double,
        |		`9`: double
        |	>,
        |	title: string
        |>
      """.stripMargin
    val schema = DataType.fromDDL(ddlSchema).asInstanceOf[StructType]("x").dataType.asInstanceOf[StructType]
    println(schema.prettyJson)
    val df = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)
    df.printSchema()
    /**
      * root
      * |-- _id: struct (nullable = true)
      * |    |-- oid: string (nullable = true)
      * |-- id: string (nullable = true)
      * |-- sparse_rep: struct (nullable = true)
      * |    |-- 1: double (nullable = true)
      * |    |-- 10: double (nullable = true)
      * |    |-- 11: double (nullable = true)
      * |    |-- 12: double (nullable = true)
      * |    |-- 13: double (nullable = true)
      * |    |-- 14: double (nullable = true)
      * |    |-- 15: double (nullable = true)
      * |    |-- 17: double (nullable = true)
      * |    |-- 18: double (nullable = true)
      * |    |-- 2: double (nullable = true)
      * |    |-- 20: double (nullable = true)
      * |    |-- 21: double (nullable = true)
      * |    |-- 22: double (nullable = true)
      * |    |-- 23: double (nullable = true)
      * |    |-- 24: double (nullable = true)
      * |    |-- 25: double (nullable = true)
      * |    |-- 26: double (nullable = true)
      * |    |-- 27: double (nullable = true)
      * |    |-- 3: double (nullable = true)
      * |    |-- 4: double (nullable = true)
      * |    |-- 7: double (nullable = true)
      * |    |-- 9: double (nullable = true)
      * |-- title: string (nullable = true)
      */

    val ds =
      df.withColumn("sparse_rep", expr("from_json(to_json(sparse_rep), 'map<int, double>')"))
        .withColumn("_id",$"_id.oid")
        .withColumn("id",$"id".cast("int"))
      .as[BlogRow]
    ds.printSchema()

    /**
      * root
      * |-- _id: string (nullable = true)
      * |-- id: integer (nullable = true)
      * |-- sparse_rep: map (nullable = true)
      * |    |-- key: integer
      * |    |-- value: double (valueContainsNull = true)
      * |-- title: string (nullable = true)
      */
  }
  // ############################################################################################################
  @Test
  def test63321569(): Unit = {

  }



}
case class Employee(name: String, age: Long)
case class BlogRow(_id:String, id:Int, sparse_rep:Map[Int,Double],title:String)
