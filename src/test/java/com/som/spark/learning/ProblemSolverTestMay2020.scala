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
import org.apache.spark.sql.catalog.{Database, Table}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{struct, _}
import org.apache.spark.sql.types._

import scala.collection.mutable


class ProblemSolverTestMay2020 extends Serializable {

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

  case class Sales(Name: String, Platform: String, Year: Int, Genre: String, Publisher: String,
                   NA_Sales: Double, EU_Sales: Double, JP_Sales: Double, Other_Sales: Double)

  @Test
  def test61929924(): Unit = {
    import org.apache.spark.sql.catalyst.ScalaReflection

    val data =
      """
        |Gran Turismo 3: A-Spec;PS2;2001;Racing;Sony Computer Entertainment;6.85;5.09;1.87;1.16
        |Call of Duty: Modern Warfare 3;X360;2011;Shooter;Activision;9.03;4.28;0.13;1.32
        |Pokemon Yellow: Special Pikachu Edition;GB;1998;Role-Playing;Nintendo;5.89;5.04;3.12;0.59
        |Call of Duty: Black Ops;X360;2010;Shooter;Activision;9.67;3.73;0.11;1.13
        |Pokemon HeartGold/Pokemon SoulSilver;DS;2009;Action;Nintendo;4.4;2.77;3.96;0.77
        |High Heat Major League Baseball 2003;PS2;2002;Sports;3DO;0.18;0.14;0;0.05
        |Panzer Dragoon;SAT;1995;Shooter;Sega;0;0;0.37;0
        |Corvette;GBA;2003;Racing;TDK Mediactive;0.2;0.07;0;0.01
      """.stripMargin

    val ds = spark.read
      .schema(ScalaReflection.schemaFor[Sales].dataType.asInstanceOf[StructType])
      .option("sep", ";")
      .csv(data.split("\n").toSeq.toDS())

    ds.show(false)
    ds.printSchema()

    // global sales
    val processedDF = ds.withColumn("global_sale", col("NA_Sales") + col("EU_Sales") + col("JP_Sales"))
      .groupBy("Genre")
      .agg(sum("global_sale").as("global_sale_by_genre"))

    println("Lowest selling :: " + processedDF.orderBy(col("global_sale_by_genre").asc).head()
      .getValuesMap(Seq("Genre", "global_sale_by_genre")).mkString(", "))
    println("Highest selling :: " + processedDF.orderBy(col("global_sale_by_genre").desc).head()
      .getValuesMap(Seq("Genre", "global_sale_by_genre")).mkString(", "))
  }

  // ############################################################################################################
  @Test
  def test61995414(): Unit = {
    val data =
      """Gran Turismo 3: A-Spec;PS2;2001;Racing;Sony Computer Entertainment;6.85;5.09;1.87;1.16
        |Call of Duty: Modern Warfare 3;X360;2011;Shooter;Activision;9.03;4.28;0.13;1.32
        |Pokemon Yellow: Special Pikachu Edition;GB;1998;Role-Playing;Nintendo;5.89;5.04;3.12;0.59
        |Call of Duty: Black Ops;X360;2010;Shooter;Activision;9.67;3.73;0.11;1.13
        |Pokemon HeartGold/Pokemon SoulSilver;DS;2009;Action;Nintendo;4.4;2.77;3.96;0.77
        |High Heat Major League Baseball 2003;PS2;2002;Sports;3DO;0.18;0.14;0;0.05
        |Panzer Dragoon;SAT;1995;Shooter;Sega;0;0;0.37;0
        |Corvette;GBA;2003;Racing;TDK Mediactive;0.2;0.07;0;0.01""".stripMargin

    val vgdataLines = spark.sparkContext.makeRDD(data.split("\n").toSeq)
    val vgdata = vgdataLines.map(_.split(";"))

    val GlobalSales = vgdata.map(r => (r(3), r(5).toDouble + r(6).toDouble + r(7).toDouble)).reduceByKey(_ + _)

    GlobalSales.foreach(println)
    //   (Shooter,27.32)
    //   (Role-Playing,14.05)
    //   (Sports,0.32)
    //   (Action,11.129999999999999)
    //   (Racing,14.079999999999998)

    println("### min-max ###")
    val minSale = GlobalSales.min()(Ordering.by(_._2))
    val maxSale = GlobalSales.max()(Ordering.by(_._2))
    println(s"Highest selling Genre: '${maxSale._1}' Global Sale (in millions): '${maxSale._2}'.")
    println(s"Lowest selling Genre: '${minSale._1}' Global Sale (in millions): '${minSale._2}'.")

    //   ### min-max ###
    //     Highest selling Genre: 'Shooter' Global Sale (in millions): '27.32'.
    //   Lowest selling Genre: 'Sports' Global Sale (in millions): '0.32'.

  }


  // ############################################################################################################

  case class Data(matrix: Matrix)

  @Test
  def test61994423(): Unit = {
    import org.apache.hadoop.fs.Path
    import org.apache.spark.ml.linalg.{Matrices, Matrix}

    def save(matrix: Matrix, path: String): Unit = {
      val data = Data(matrix)
      val df = spark.createDataFrame(Seq(data))
      val dataPath = new Path(path, "data").toString
      df.repartition(1).write.mode("overwrite").parquet(dataPath)
    }

    def load(path: String): Matrix = {
      val dataPath = new Path(path, "data").toString
      val df = spark.read.parquet(dataPath)
      val Row(matrix: Matrix) = df.select("matrix").head()
      matrix
    }

    println("### input matrix ###")
    val matrixToSave = Matrices.eye(3)
    println(matrixToSave)
    save(matrixToSave, "/Users/sokale/models/matrix")
    val matrixLoaded = load("/Users/sokale/models/matrix")
    println("### Loaded matrix ###")
    println(matrixLoaded)

    //  ### input matrix ###
    //    1.0  0.0  0.0
    //  0.0  1.0  0.0
    //  0.0  0.0  1.0
    //  ### Loaded matrix ###
    //    1.0  0.0  0.0
    //  0.0  1.0  0.0
    //  0.0  0.0  1.0
  }

  // ############################################################################################################

  @Test
  def test61993247(): Unit = {
    val data1 =
      """
        |InvoiceNo|StockCode|         Description|Quantity|   InvoiceDate|UnitPrice|CustomerID|       Country
        |   536365|   85123A|WHITE HANGING HEA...|       6|12/1/2010 8:26|     2.55|     17850|United Kingdom
        |   536365|    71053| WHITE METAL LANTERN|       6|12/1/2010 8:26|     3.39|     17850|United Kingdom
        |   536365|   84406B|CREAM CUPID HEART...|       8|12/1/2010 8:26|     2.75|     17850|United Kingdom
        |   536365|   84029G|KNITTED UNION FLA...|       6|12/1/2010 8:26|     3.39|     17850|United Kingdom
        |   536365|   84029E|RED WOOLLY HOTTIE...|       6|12/1/2010 8:26|     3.39|     17850|United Kingdom
        |   536365|    22752|SET 7 BABUSHKA NE...|       -2|12/1/2010 8:26|     7.65|    17850|United Kingdom
      """.stripMargin

    val stringDS = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    df.filter("Quantity>=0").show(false)
  }

  // ############################################################################################################

  def mlOnIrisData(): (PipelineModel, Pipeline, CrossValidatorModel, DataFrame) = {

    // Flag to switch between StringIndexer and CustomStringIndexer
    val USE_CUSTOM_STRING_INDEXER = false

    val irisDataURL = getClass.getResource("/data/classification/irisData.csv")

    val irisDatasetDF = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(irisDataURL.getPath)

    irisDatasetDF.show(5)

    val featurePreprocessingPipeline = new Pipeline()

    val cat_columns = Array("sepal_length", "sepal_width", "petal_length", "petal_width")
    val cat_columns_idx = cat_columns.map(col => col + "_Idx")

    val out_columns = cat_columns.map(col => col + "_Ohe")
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setDropLast(false)
      .setInputCols(cat_columns_idx)
      .setOutputCols(out_columns)

    val vectorAssembler = new VectorAssembler()
      .setInputCols(out_columns)
      .setOutputCol("features")

    val inputColsOutputCols = cat_columns.zip(cat_columns_idx)
    val index_transformers = inputColsOutputCols.map(inputColOutputCol => {
      new StringIndexer()
        .setInputCol(inputColOutputCol._1)
        .setOutputCol(inputColOutputCol._2)
    })
    featurePreprocessingPipeline.setStages(index_transformers ++ Array(oneHotEncoder) ++ Array(vectorAssembler))

    val transformModel = featurePreprocessingPipeline.fit(irisDatasetDF)
    var ds_enc = transformModel.transform(irisDatasetDF)
    ds_enc = ds_enc.drop(cat_columns_idx: _*).drop(out_columns: _*).drop(cat_columns: _*)

    ds_enc.printSchema()

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(ds_enc)

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(ds_enc)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = ds_enc.randomSplit(Array(0.7, 0.3), seed = 123)

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(10))
      .build()

    // Convert indexed labels back to original labels. Testing IndexToString
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("indexedLabel"))
      .setNumFolds(3)

    // Train model. This also runs the indexers.
    val model = cv.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + accuracy)

    // val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    // println("Learned classification forest model:\n" + rfModel.toDebugString)

    (transformModel, pipeline, model, predictions)
  }

  // ############################################################################################################

  @Test
  def test61981478(): Unit = {
    //    // Chain indexers and forest in a Pipeline.
    //    val pipeline = new Pipeline()
    //      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    val (featurePipelineModel, pipeline, crossValidatorModel, predictions) = mlOnIrisData()

    val labelIndexer = pipeline.getStages(0).asInstanceOf[StringIndexerModel]
    // in my case, StringIndexerModel is referenced as labelIndexer
    val labelToIndex = labelIndexer.labels.zipWithIndex.map(_.swap).toMap
    println(labelToIndex)

    import org.apache.spark.ml.linalg.Vector
    val mapToLabel = udf((vector: Vector) => vector.toArray.zipWithIndex.toMap.map{
      case(prob, index) => labelToIndex(index) -> prob
    })
    predictions.select(
      col("features"),
      col("probability"),
      to_json(mapToLabel(col("probability"))).as("probability_json"),
      col("prediction"),
      col("predictedLabel"))
      .show(5,false)

    //    +-------------------------------------+------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+----------+--------------+
    //    |features                             |probability                                                 |probability_json                                                                                             |prediction|predictedLabel|
    //    +-------------------------------------+------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+----------+--------------+
    //    |(123,[0,37,82,101],[1.0,1.0,1.0,1.0])|[0.7094347002635046,0.174338768115942,0.11622653162055337]  |{"Iris-setosa":0.7094347002635046,"Iris-versicolor":0.174338768115942,"Iris-virginica":0.11622653162055337}  |0.0       |Iris-setosa   |
    //      |(123,[0,39,58,101],[1.0,1.0,1.0,1.0])|[0.7867074275362319,0.12433876811594202,0.0889538043478261] |{"Iris-setosa":0.7867074275362319,"Iris-versicolor":0.12433876811594202,"Iris-virginica":0.0889538043478261} |0.0       |Iris-setosa   |
    //      |(123,[0,39,62,107],[1.0,1.0,1.0,1.0])|[0.5159492704509036,0.2794443583750028,0.2046063711740936]  |{"Iris-setosa":0.5159492704509036,"Iris-versicolor":0.2794443583750028,"Iris-virginica":0.2046063711740936}  |0.0       |Iris-setosa   |
    //      |(123,[2,39,58,101],[1.0,1.0,1.0,1.0])|[0.7822379507920459,0.12164981462756994,0.09611223458038423]|{"Iris-setosa":0.7822379507920459,"Iris-versicolor":0.12164981462756994,"Iris-virginica":0.09611223458038423}|0.0       |Iris-setosa   |
    //      |(123,[2,43,62,101],[1.0,1.0,1.0,1.0])|[0.7049652235193186,0.17164981462756992,0.1233849618531115] |{"Iris-setosa":0.7049652235193186,"Iris-versicolor":0.17164981462756992,"Iris-virginica":0.1233849618531115} |0.0       |Iris-setosa   |
    //      +-------------------------------------+------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+----------+--------------+
    //    only showing top 5 rows
  }

  // ############################################################################################################

  @Test
  def test61896971(): Unit = {
    //    spark-test-data.json
    //    --------------------
    //    {"id":1,"name":"abc1"}
    //    {"id":2,"name":"abc2"}
    //    {"id":3,"name":"abc3"}
    import scala.io.Source
    val path = "spark-test-data.json"
    val fileContent = Source.fromFile(getClass.getResource("/" + path).getPath).getLines()
    val df = spark.read
      .json(fileContent.toSeq.toDS())

    df.show(false)
    df.printSchema()

    spark.sparkContext.addFile(getClass.getResource("/" + path).getPath)
    val absolutePathOfFile = SparkFiles.get(path)

    println(s"Absolute path added via sparkcontext.addfile(filePath): $absolutePathOfFile")
    val df1 = spark.read
      .json(absolutePathOfFile)
    df1.show(false)

    //    Absolute path added via sparkcontext.addfile(filePath): /private/var/folders/mg/lcv5jvyd6dx2vtr7zpmg48s80000gn/T
    //    /spark-77290ab6-21ef-4960-8a11-91812675d759/userFiles-681fa1fd-9f09-4d9a-a083-6f3c24f9ab8e/spark-test-data.json
    //      +---+----+
    //      |id |name|
    //      +---+----+
    //      |1  |abc1|
    //      |2  |abc2|
    //      |3  |abc3|
    //      +---+----+
  }

  // ############################################################################################################

  @Test
  def test61955248(): Unit = {
    val data =
      """
        |2018-04-07 07:07:17
        |2018-04-07 07:32:27
        |2018-04-07 08:36:44
        |2018-04-07 08:38:00
        |2018-04-07 08:39:29
        |2018-04-08 01:43:08
        |2018-04-08 01:43:55
        |2018-04-09 07:52:31
        |2018-04-09 07:52:42
        |2019-01-24 11:52:31
        |2019-01-24 12:52:42
        |2019-01-25 12:52:42
      """.stripMargin
    val df = spark.read
      .schema(StructType(Array(StructField("date_time", DataTypes.TimestampType))))
      .csv(data.split(System.lineSeparator()).toSeq.toDS())
    df.show(false)
    df.printSchema()

    // Bucketize the data and find the count for each hour
    val hour = 60 * 60
    // convert the time into unix epoch
    val processedDF = df.withColumn("unix_epoch", unix_timestamp(col("date_time")))
      .withColumn("hour_bucket", floor(col("unix_epoch")/hour))
      .groupBy("hour_bucket")
      .count()

    processedDF.show(false)

    // find hourly average count
    processedDF.agg(avg("count")).show(false)
  }

  // ############################################################################################################

  @Test
  def test61970371(): Unit = {
    val schema = StructType(
      "ID|LOAN|COUNT|A1 |A2 |A3 |A4 |B1 |B2 |B3 |B4"
        .split("\\|")
        .map(f => StructField(f.trim, DataTypes.IntegerType))
    )
    val data =
      """
        | 1| 100|    1| 35|   |   |   |444|   |   |
        | 2| 200|    3| 30| 15| 18|   |111|222|333|
        | 3| 300|    2| 18| 20|   |   |555|666|   |
        | 4| 400|    4| 28| 60| 80| 90|777|888|123|456
        | 5| 500|    1| 45|   |   |   |245|   |   |
      """.stripMargin
    val df = spark.read
      .schema(schema)
      .option("sep", "|")
      .csv(data.split(System.lineSeparator()).map(_.replaceAll("\\s*", "")).toSeq.toDS())
    df.show(false)
    df.printSchema()

    //    +---+----+-----+---+----+----+----+---+----+----+----+
    //    |ID |LOAN|COUNT|A1 |A2  |A3  |A4  |B1 |B2  |B3  |B4  |
    //    +---+----+-----+---+----+----+----+---+----+----+----+
    //    |1  |100 |1    |35 |null|null|null|444|null|null|null|
    //      |2  |200 |3    |30 |15  |18  |null|111|222 |333 |null|
    //      |3  |300 |2    |18 |20  |null|null|555|666 |null|null|
    //      |4  |400 |4    |28 |60  |80  |90  |777|888 |123 |456 |
    //      |5  |500 |1    |45 |null|null|null|245|null|null|null|
    //      +---+----+-----+---+----+----+----+---+----+----+----+

    //    unpivot the table and remove null entry
    df.selectExpr(
      "ID",
      "LOAN",
      "stack(4, A1, B1, A2, B2, A3, B3, A4, B4) as (A, B)"
    ).where("A is not null and B is not null").show(false)


    //    +---+----+---+---+
    //    |ID |LOAN|A  |B  |
    //    +---+----+---+---+
    //    |1  |100 |35 |444|
    //      |2  |200 |30 |111|
    //      |2  |200 |15 |222|
    //      |2  |200 |18 |333|
    //      |3  |300 |18 |555|
    //      |3  |300 |20 |666|
    //      |4  |400 |28 |777|
    //      |4  |400 |60 |888|
    //      |4  |400 |80 |123|
    //      |4  |400 |90 |456|
    //      |5  |500 |45 |245|
    //      +---+----+---+---+

  }

  // ############################################################################################################

  @Test
  def test62016466(): Unit = {
    val table_0 =  spark.range(1, 5)
      .withColumn("Array_0",
        array(struct(lit("a").cast(StringType).as("f1"), lit(2).as("f2"))))
      .withColumn("Array_1", array(lit(null))) // .cast("array<struct<f1:string, f2:int>>")
    table_0.show(false)
    table_0.printSchema()

    //    +---+-------+-------+
    //    |id |Array_0|Array_1|
    //    +---+-------+-------+
    //    |1  |[1, 2] |[]     |
    //      |2  |[1, 2] |[]     |
    //      |3  |[1, 2] |[]     |
    //      |4  |[1, 2] |[]     |
    //      +---+-------+-------+
    //
    //    root
    //    |-- id: long (nullable = false)
    //    |-- Array_0: array (nullable = false)
    //    |    |-- element: integer (containsNull = false)
    //    |-- Array_1: array (nullable = false)
    //    |    |-- element: integer (containsNull = true)
    table_0.createOrReplaceTempView("table_0")

    spark.sql(
      """
        |SELECT exploded_b_values.*, table_0.id
        |FROM table_0
        |    LATERAL VIEW explode(table_0.Array_1) exploded_b_values AS B
      """.stripMargin).printSchema()

    //    root
    //    |-- B: null (nullable = true)
    //    |-- id: long (nullable = false)

    //    +----+---+
    //    |B   |id |
    //    +----+---+
    //    |null|1  |
    //      |null|2  |
    //      |null|3  |
    //      |null|4  |
    //      +----+---+

    spark.sql(
      """
        |SELECT exploded_a_values.*, table_0.id
        |FROM table_0
        |    LATERAL VIEW explode(table_0.Array_0) exploded_a_values AS A
      """.stripMargin).printSchema()

    //    root
    //    |-- A: struct (nullable = false)
    //    |    |-- f1: string (nullable = false)
    //    |    |-- f2: integer (nullable = false)
    //    |-- id: long (nullable = false)

    val processed = spark.sql(
      """
        |SELECT exploded_a_values.*, table_0.id
        |FROM table_0
        |    LATERAL VIEW explode(table_0.Array_0) exploded_a_values AS A
        |UNION
        |SELECT exploded_b_values.*, table_0.id
        |FROM table_0
        |    LATERAL VIEW explode(table_0.Array_1) exploded_b_values AS B
      """.stripMargin)
    processed.show(false)
    processed.printSchema()

    //    +------+---+
    //    |A     |id |
    //    +------+---+
    //    |[a, 2]|1  |
    //      |[a, 2]|2  |
    //      |[a, 2]|4  |
    //      |null  |2  |
    //      |null  |4  |
    //      |[a, 2]|3  |
    //      |null  |1  |
    //      |null  |3  |
    //      +------+---+
    //
    //    root
    //    |-- A: struct (nullable = true)
    //    |    |-- f1: string (nullable = false)
    //    |    |-- f2: integer (nullable = false)
    //    |-- id: long (nullable = false)

  }

  // ############################################################################################################

  @Test
  def test61961840(): Unit = {
    val data =
      """
        |{"person":[{"name":"david", "email": "david@gmail.com"}, {"name":"steve", "email": "steve@gmail.com"}]}
      """.stripMargin
    val df = spark.read
      .json(data.split(System.lineSeparator()).toSeq.toDS())
    df.show(false)
    df.printSchema()

    //
    //    +----------------------------------------------------+
    //    |person                                              |
    //    +----------------------------------------------------+
    //    |[[david@gmail.com, david], [steve@gmail.com, steve]]|
    //    +----------------------------------------------------+
    //
    //    root
    //    |-- person: array (nullable = true)
    //    |    |-- element: struct (containsNull = true)
    //    |    |    |-- email: string (nullable = true)
    //    |    |    |-- name: string (nullable = true)
    //

    // Spark > 2.3
    val answer1 = df.withColumn("person_processed",
      expr("transform(person, x -> named_struct( 'email', reverse(x.email), 'name', x.name))"))
    answer1.show(false)
    answer1.printSchema()

    //
    //    +----------------------------------------------------+----------------------------------------------------+
    //    |person                                              |person_processed                                    |
    //    +----------------------------------------------------+----------------------------------------------------+
    //    |[[david@gmail.com, david], [steve@gmail.com, steve]]|[[moc.liamg@divad, david], [moc.liamg@evets, steve]]|
    //    +----------------------------------------------------+----------------------------------------------------+
    //
    //    root
    //    |-- person: array (nullable = true)
    //    |    |-- element: struct (containsNull = true)
    //    |    |    |-- email: string (nullable = true)
    //    |    |    |-- name: string (nullable = true)
    //    |-- person_processed: array (nullable = true)
    //    |    |-- element: struct (containsNull = false)
    //    |    |    |-- email: string (nullable = true)
    //    |    |    |-- name: string (nullable = true)
    //

    // spark < 2.3
    case class InfoData(name: String, email: String)
    val infoDataSchema =
      ArrayType(StructType(Array(StructField("name", StringType), StructField("email", StringType))))

    val reverseEmailUDF = udf((arr1: mutable.WrappedArray[String], arr2: mutable.WrappedArray[String]) => {
      if (arr1.length != arr2.length) null
      else arr1.zipWithIndex.map(t => InfoData(t._1, arr2(t._2).reverse))
    }, infoDataSchema)

    val spark2_3Processed = df
      .withColumn("person_processed",
        reverseEmailUDF(
          col("person.name").cast("array<string>"),
          col("person.email").cast("array<string>")
        )
      )

    spark2_3Processed.show(false)
    spark2_3Processed.printSchema()

    //    +----------------------------------------------------+----------------------------------------------------+
    //    |person                                              |person_processed                                    |
    //    +----------------------------------------------------+----------------------------------------------------+
    //    |[[david@gmail.com, david], [steve@gmail.com, steve]]|[[david, moc.liamg@divad], [steve, moc.liamg@evets]]|
    //    +----------------------------------------------------+----------------------------------------------------+
    //
    //    root
    //    |-- person: array (nullable = true)
    //    |    |-- element: struct (containsNull = true)
    //    |    |    |-- email: string (nullable = true)
    //    |    |    |-- name: string (nullable = true)
    //    |-- person_processed: array (nullable = true)
    //    |    |-- element: struct (containsNull = true)
    //    |    |    |-- name: string (nullable = true)
    //    |    |    |-- email: string (nullable = true)

    // spark < 2.3
    // if you can't create case class
    println(df.schema("person").dataType)
    val subSchema = df.schema("person").dataType

    val reverseEmailUDF_withoutCaseClass = //udf((nameArrayRow: Row, emailArrayRow: Row) => {
      udf((nameArray: mutable.WrappedArray[String], emailArray: mutable.WrappedArray[String]) => {
        if (nameArray.length != emailArray.length) null
        else nameArray.zipWithIndex.map(t => (t._1, emailArray(t._2).reverse))
      }, subSchema)

    val withoutCaseClasDF = df
      .withColumn("person_processed",
        reverseEmailUDF_withoutCaseClass(
          col("person.name").cast("array<string>"),
          col("person.email").cast("array<string>")
        )
      )

    withoutCaseClasDF.show(false)
    withoutCaseClasDF.printSchema()
    withoutCaseClasDF.select("person_processed.email").show(false)

    //
    //    ArrayType(StructType(StructField(email,StringType,true), StructField(name,StringType,true)),true)
    //    +----------------------------------------------------+----------------------------------------------------+
    //    |person                                              |person_processed                                    |
    //    +----------------------------------------------------+----------------------------------------------------+
    //    |[[david@gmail.com, david], [steve@gmail.com, steve]]|[[david, moc.liamg@divad], [steve, moc.liamg@evets]]|
    //    +----------------------------------------------------+----------------------------------------------------+
    //
    //    root
    //    |-- person: array (nullable = true)
    //    |    |-- element: struct (containsNull = true)
    //    |    |    |-- email: string (nullable = true)
    //    |    |    |-- name: string (nullable = true)
    //    |-- person_processed: array (nullable = true)
    //    |    |-- element: struct (containsNull = true)
    //    |    |    |-- email: string (nullable = true)
    //    |    |    |-- name: string (nullable = true)
    //
    //    +--------------+
    //    |email         |
    //    +--------------+
    //    |[david, steve]|
    //    +--------------+
    //

  }


  // ############################################################################################################

  class FlatMapTransformer(override val uid: String)
    extends Transformer {
    /**
      * Param for input column name.
      *
      * @group param
      */
    final val inputCol = new Param[String](this, "inputCol", "The input column")

    final def getInputCol: String = $(inputCol)

    /**
      * Param for output column name.
      *
      * @group param
      */
    final val outputCol = new Param[String](this, "outputCol", "The output column")

    final def getOutputCol: String = $(outputCol)

    def setInputCol(value: String): this.type = set(inputCol, value)

    def setOutputCol(value: String): this.type = set(outputCol, value)

    def this() = this(Identifiable.randomUID("FlatMapTransformer"))

    private val flatMap: String => Seq[String] = { input: String =>
      input.split(",")
    }

    def copy(extra: ParamMap): FlatMapTransformer = defaultCopy(extra)

    override def transform(dataset: Dataset[_]): DataFrame = {
      val flatMapUdf = udf(flatMap, ArrayType(StringType))
      dataset.withColumn($(outputCol), flatMapUdf(col($(inputCol))))
    }

    override def transformSchema(schema: StructType): StructType = {
      val dataType = schema($(inputCol)).dataType
      require(
        dataType.isInstanceOf[StringType],
        s"Input column must be of type StringType but got ${dataType}")
      val inputFields = schema.fields
      require(
        !inputFields.exists(_.name == $(outputCol)),
        s"Output column ${$(outputCol)} already exists.")

      schema.add($(outputCol), ArrayType(StringType))
    }
  }

  @Test
  def test62018875(): Unit = {

    val train = spark.range(10)
      .withColumn("cat", rand())
      .withColumn("cat_ohe", lit("a,b"))


    val num_features = Array("id")
    val cat_ohe_features = Array("cat_ohe")
    val cat_features_string_index = Array("cat")
    val catIndexer = cat_features_string_index.map {
      feature =>
        new StringIndexer()
          .setInputCol(feature)
          .setOutputCol(feature + "_index")
          .setHandleInvalid("keep")
    }

    val flatMapper = cat_ohe_features.map {
      feature =>
        new FlatMapTransformer()
          .setInputCol(feature)
          .setOutputCol(feature + "_transformed")
    }

    val countVectorizer = cat_ohe_features.map {
      feature =>

        new CountVectorizer()
          .setInputCol(feature + "_transformed")
          .setOutputCol(feature + "_vectorized")
          .setVocabSize(10)
    }


    // val countVectorizer = cat_ohe_features.map {
    //   feature =>
    //
    //     val flatMapper = new FlatMapTransformer()
    //       .setInputCol(feature)
    //       .setOutputCol(feature + "_transformed")
    //
    //     new CountVectorizer()
    //       .setInputCol(flatMapper.getOutputCol)
    //       .setOutputCol(feature + "_vectorized")
    //       .setVocabSize(10)
    // }

    val cat_features_index = cat_features_string_index.map {
      (feature: String) => feature + "_index"
    }

    val count_vectorized_index = cat_ohe_features.map {
      (feature: String) => feature + "_vectorized"
    }

    val catFeatureAssembler = new VectorAssembler()
      .setInputCols(cat_features_index)
      .setOutputCol("cat_features")

    val oheFeatureAssembler = new VectorAssembler()
      .setInputCols(count_vectorized_index)
      .setOutputCol("cat_ohe_features")

    val numFeatureAssembler = new VectorAssembler()
      .setInputCols(num_features)
      .setOutputCol("num_features")

    val featureAssembler = new VectorAssembler()
      .setInputCols(Array("cat_features", "num_features", "cat_ohe_features"))
      .setOutputCol("features")

    val pipelineStages = catIndexer ++ flatMapper ++ countVectorizer ++
      Array(
        catFeatureAssembler,
        oheFeatureAssembler,
        numFeatureAssembler,
        featureAssembler)

    val pipeline = new Pipeline().setStages(pipelineStages)
    val model = pipeline.fit(dataset = train)
    model.transform(train).show(false)

    /**
      * +---+-------------------+-------+---------+-------------------+-------------------+------------+----------------+------------+-----------------+
      * |id |cat                |cat_ohe|cat_index|cat_ohe_transformed|cat_ohe_vectorized |cat_features|cat_ohe_features|num_features|features         |
      * +---+-------------------+-------+---------+-------------------+-------------------+------------+----------------+------------+-----------------+
      * |0  |0.5090906225798505 |a,b    |2.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[2.0]       |[1.0,1.0]       |[0.0]       |[2.0,0.0,1.0,1.0]|
      * |1  |0.8019883419510832 |a,b    |7.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[7.0]       |[1.0,1.0]       |[1.0]       |[7.0,1.0,1.0,1.0]|
      * |2  |0.6189101074687529 |a,b    |5.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[5.0]       |[1.0,1.0]       |[2.0]       |[5.0,2.0,1.0,1.0]|
      * |3  |0.1855605832809084 |a,b    |6.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[6.0]       |[1.0,1.0]       |[3.0]       |[6.0,3.0,1.0,1.0]|
      * |4  |0.23381846247134597|a,b    |8.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[8.0]       |[1.0,1.0]       |[4.0]       |[8.0,4.0,1.0,1.0]|
      * |5  |0.47886431990303546|a,b    |3.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[3.0]       |[1.0,1.0]       |[5.0]       |[3.0,5.0,1.0,1.0]|
      * |6  |0.8733308393998128 |a,b    |9.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[9.0]       |[1.0,1.0]       |[6.0]       |[9.0,6.0,1.0,1.0]|
      * |7  |0.8250921802204912 |a,b    |1.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[1.0]       |[1.0,1.0]       |[7.0]       |[1.0,7.0,1.0,1.0]|
      * |8  |0.8698673151005127 |a,b    |4.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[4.0]       |[1.0,1.0]       |[8.0]       |[4.0,8.0,1.0,1.0]|
      * |9  |0.9832602298773477 |a,b    |0.0      |[a, b]             |(2,[0,1],[1.0,1.0])|[0.0]       |[1.0,1.0]       |[9.0]       |[0.0,9.0,1.0,1.0]|
      * +---+-------------------+-------+---------+-------------------+-------------------+------------+----------------+------------+-----------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62025380(): Unit = {
    val dfInput = spark.range(1).withColumn("Qty", col("id").cast(StringType))
    val processDF = dfInput.withColumn("QtyOut",dfInput.col("Qty").cast("decimal(32,9)"))
    processDF.show(false)
    processDF.printSchema()

    processDF.withColumn("NewQtyOut",format_number(processDF.col("QtyOut"),9)).show()
    processDF.withColumn("NewQtyOut",format_number(processDF.col("QtyOut"),9)).printSchema()

    processDF
      .withColumn("isTrue", when(col("QtyOut").equalTo(0), true).otherwise(false))
      .show(false)

    def bigDecimalFormatter(x: Double, y: Int): Double =
      BigDecimal(x).setScale(y, BigDecimal.RoundingMode.HALF_UP).toDouble

    val decimalFormatter = udf((decimal: Double, scale: Int) => bigDecimalFormatter(decimal, scale))
    processDF.select(decimalFormatter(col("QtyOut"), lit(9)),
      decimalFormatter(lit(1.1000000453E4), lit(5)))
      .show(false)

    /**
      * +--------------+--------------------+
      * |UDF(QtyOut, 9)|UDF(11000.000453, 5)|
      * +--------------+--------------------+
      * |0.0           |11000.00045         |
      * +--------------+--------------------+
      */

    /**
      * +---+---+------+
      * |id |Qty|QtyOut|
      * +---+---+------+
      * |0  |0  |0E-9  |
      * +---+---+------+
      *
      * root
      * |-- id: long (nullable = false)
      * |-- Qty: string (nullable = false)
      * |-- QtyOut: decimal(32,9) (nullable = true)
      *
      * +---+---+------+-----------+
      * | id|Qty|QtyOut|  NewQtyOut|
      * +---+---+------+-----------+
      * |  0|  0|  0E-9|0.000000000|
      * +---+---+------+-----------+
      *
      * root
      * |-- id: long (nullable = false)
      * |-- Qty: string (nullable = false)
      * |-- QtyOut: decimal(32,9) (nullable = true)
      * |-- NewQtyOut: string (nullable = true)
      *
      * +---+---+------+------+
      * |id |Qty|QtyOut|isTrue|
      * +---+---+------+------+
      * |0  |0  |0E-9  |true  |
      * +---+---+------+------+
      */

  }

  // ############################################################################################################

  @Test
  def test62045116(): Unit = {
    val data =
      """
        |2017-04-07 07:07:17
        |2017-04-07 07:32:27
        |2017-04-07 08:36:44
        |2017-04-07 08:38:00
        |2017-04-07 08:39:29
        |2017-04-07 07:07:17
        |2018-04-07 07:32:27
        |2018-04-07 08:36:44
        |2018-04-07 08:38:00
        |2018-04-07 08:39:29
        |2018-04-08 01:43:08
        |2018-04-08 01:43:55
        |2018-04-09 07:52:31
        |2018-04-09 07:52:42
        |2019-01-24 11:52:31
        |2019-01-24 12:52:42
        |2019-01-25 12:52:42
      """.stripMargin
    val df = spark.read
      .schema(StructType(Array(StructField("startDate", DataTypes.TimestampType))))
      .csv(data.split(System.lineSeparator()).toSeq.toDS())
    df.show(false)
    df.printSchema()

    /**
      * +-------------------+
      * |startDate          |
      * +-------------------+
      * |2018-04-07 07:07:17|
      * |2018-04-07 07:32:27|
      * |2018-04-07 08:36:44|
      * |2018-04-07 08:38:00|
      * |2018-04-07 08:39:29|
      * |2018-04-08 01:43:08|
      * |2018-04-08 01:43:55|
      * |2018-04-09 07:52:31|
      * |2018-04-09 07:52:42|
      * |2019-01-24 11:52:31|
      * |2019-01-24 12:52:42|
      * |2019-01-25 12:52:42|
      * +-------------------+
      *
      * root
      * |-- startDate: timestamp (nullable = true)
      */


    val filterCOl = (currentDate: String) =>  when(datediff(date_format(lit(currentDate), "yyyy-MM-dd")
      ,date_format(lit(currentDate), "yyyy-MM-01"))===lit(0),
      date_format(col("startDate"), "yyyy-MM") ===
        date_format(concat_ws("-",year(lit(currentDate)), month(lit(currentDate)) -1), "yyyy-MM")
    ).otherwise(to_date(col("startDate"))
      .between(date_format(lit(currentDate), "yyyy-MM-01"), lit(currentDate)))
    // if current date
    var currentDateStr = "2018-04-08"
    df.filter(filterCOl(currentDateStr)).show(false)

    /**
      * +-------------------+
      * |startDate          |
      * +-------------------+
      * |2018-04-07 07:07:17|
      * |2018-04-07 07:32:27|
      * |2018-04-07 08:36:44|
      * |2018-04-07 08:38:00|
      * |2018-04-07 08:39:29|
      * |2018-04-08 01:43:08|
      * |2018-04-08 01:43:55|
      * +-------------------+
      */

    currentDateStr = "2018-05-01"
    df.filter(filterCOl(currentDateStr)).show(false)

    /**
      * +-------------------+
      * |startDate          |
      * +-------------------+
      * |2018-04-07 07:07:17|
      * |2018-04-07 07:32:27|
      * |2018-04-07 08:36:44|
      * |2018-04-07 08:38:00|
      * |2018-04-07 08:39:29|
      * |2018-04-08 01:43:08|
      * |2018-04-08 01:43:55|
      * |2018-04-09 07:52:31|
      * |2018-04-09 07:52:42|
      * +-------------------+
      */

  }

  // ############################################################################################################

  @Test
  def test62044544(): Unit = {

    val data =
      """
        |{
        |  "id": "1",
        |  "type": "arr1",
        |  "address": [
        |    {
        |      "id": "1",
        |      "street": "abc",
        |      "city": "NY",
        |      "order": "Primary"
        |    },
        |    {
        |      "id": "2",
        |      "street": "xyz",
        |      "city": "SA",
        |      "order": "Secondary"
        |    },
        |    {
        |      "id": "1",
        |      "street": "abc",
        |      "city": "NY",
        |      "order": "Secondary"
        |    }
        |  ]
        |}
      """.stripMargin
    val df = spark.read
      .option("multiline", true)
      .json(Seq(data).toDS())
    df.show(false)
    df.printSchema()


    println(df.schema("address").dataType)
    val subSchema = df.schema("address").dataType

    val distingAddress =
      udf((addressArray: mutable.WrappedArray[Row]) => {
        val sortedArray = addressArray.sortWith((current, next) => {
          val (cStreet, cCity) =  (current.getAs[String]("street"), current.getAs[String]("city"))
          val (nStreet, nCity) =  (next.getAs[String]("street"), next.getAs[String]("city"))
          cStreet == nStreet && cCity == nCity
        } )

        //        var c = 0
        //       val array = for (i <- sortedArray.indices) yield {
        //         val (current, next) =  (sortedArray(c), sortedArray(i))
        //          val (cStreet, cCity) =  (current.getAs[String]("street"), current.getAs[String]("city"))
        //          val (nStreet, nCity) =  (next.getAs[String]("street"), next.getAs[String]("city"))
        //          if (cStreet == nStreet && cCity == nCity) {
        //            current
        //          } else {
        //            c = i
        //          }
        //        }
        ////       val set = addressArray.map(row =>
        ////          (row.getAs[String]("order"),
        ////            row.getAs[String]("id"),
        ////            row.getAs[String]("street"),
        ////            row.getAs[String]("city"))
        //////         (row.getAs[String]("street"), row.getAs[String]("city")) -> row
        ////        )
        sortedArray
      }, subSchema)

    val distingAddressDF = df
      .withColumn("address_ab",
        distingAddress(
          col("address")
        )
      )

    distingAddressDF.show(false)
    distingAddressDF.printSchema()

  }

  // ############################################################################################################

  @Test
  def test62054332(): Unit = {
    val data = """[{"A":120.0,"B":"0005236"},{"A":10.0,"B":"0005200"},{"A":12.0,"B":"00042276"},{"A":20.0,"B":"00052000"}]"""

    // case-1 if this is input data
    val df = spark.read.json(Seq(data).toDS())
    df.printSchema()
    df.show(false)

    /**
      * root
      * |-- A: double (nullable = true)
      * |-- B: string (nullable = true)
      *
      * +-----+--------+
      * |A    |B       |
      * +-----+--------+
      * |120.0|0005236 |
      * |10.0 |0005200 |
      * |12.0 |00042276|
      * |20.0 |00052000|
      * +-----+--------+
      */
    // case-2 if this is one of the column
    val df2 = Seq(data).toDF("gtins")
    df2.show(false)
    df2.printSchema()

    /**
      * +--------------------------------------------------------------------------------------------------------+
      * |gtins                                                                                                   |
      * +--------------------------------------------------------------------------------------------------------+
      * |[{"A":120.0,"B":"0005236"},{"A":10.0,"B":"0005200"},{"A":12.0,"B":"00042276"},{"A":20.0,"B":"00052000"}]|
      * +--------------------------------------------------------------------------------------------------------+
      *
      * root
      * |-- gtins: string (nullable = true)
      */

    df2.selectExpr("inline_outer(from_json(gtins, 'array<struct<A:double, B:string>>')) as (packQty, gtin)")
      .show(false)

    /**
      * +-------+--------+
      * |packQty|gtin    |
      * +-------+--------+
      * |120.0  |0005236 |
      * |10.0   |0005200 |
      * |12.0   |00042276|
      * |20.0   |00052000|
      * +-------+--------+
      */
  }

  // ############################################################################################################

  @Test
  def test62058267(): Unit = {
    val df1 = Seq(
      ("a", 2, "c"),
      ("a", 2, "c"),
      ("a", 2, "c"),
      ("b", 2, "d"),
      ("b", 2, "d")
    ).toDF("col1", "col2", "col3").groupBy("col2").agg(
      collect_list("col1").as("col1"),
      collect_list("col3").as("col3")
    )
    df1.show(false)
    df1.printSchema()

    /**
      * +----+---------------+---------------+
      * |col2|col1           |col3           |
      * +----+---------------+---------------+
      * |2   |[a, a, a, b, b]|[c, c, c, d, d]|
      * +----+---------------+---------------+
      *
      * root
      * |-- col2: integer (nullable = false)
      * |-- col1: array (nullable = true)
      * |    |-- element: string (containsNull = true)
      * |-- col3: array (nullable = true)
      * |    |-- element: string (containsNull = true)
      */
    val transform = (str: String) => expr(s"TRANSFORM($str, x -> concat('$str-', x)) as $str")
    val cols = df1.schema.map(f => if (f.dataType.isInstanceOf[ArrayType]) {
      transform(f.name)
    } else expr(f.name))

    df1.select(cols: _*).show(false)

    /**
      * +----+----------------------------------------+----------------------------------------+
      * |col2|col1                                    |col3                                    |
      * +----+----------------------------------------+----------------------------------------+
      * |2   |[col1-a, col1-a, col1-a, col1-b, col1-b]|[col3-c, col3-c, col3-c, col3-d, col3-d]|
      * +----+----------------------------------------+----------------------------------------+
      */
  }
  // ############################################################################################################

  @Test
  def test62060242(): Unit = {
    val df = Seq(
      ("a", 2, "c"),
      ("a", 2, "c"),
      ("a", 2, "c"),
      ("b", 2, "d"),
      ("b", 2, "d")
    ).toDF("col1", "col2", "col3")
    df.repartition(5).map((row)=>row.toString())
      .write.mode(SaveMode.Append)
      .text("/Users/sokale/models/x")

    /**
      * [a,2,c]
      * [b,2,d]
      */
    df.repartition(5).select(concat_ws(",", df.columns.map(col): _*))
      .write.mode(SaveMode.Append)
      .text("/Users/sokale/models/x1")

    //for controlA
    //    df.repartition(5).select(concat_ws("""\"""+"""u001""", df.columns.map(col): _*))
    //      .write.mode(SaveMode.Append)
    //      .text("/Users/sokale/models/x2")
    //
    //    df.repartition(5)
    //      .write
    //      .mode(SaveMode.Append)
    //      .option("header", true)
    //      .option("sep", """\"""+"""u001""")
    //      .csv("/Users/sokale/models/csv")
    /**
      * a,2,c
      * b,2,d
      */
  }

  // ############################################################################################################

  @Test
  def test62054357(): Unit = {
    val data = """{"Id":"31279605299","Type":"12121212","client":"Checklist _API","eventTime":"2020-03-17T15:50:30.640Z","eventType":"Event","payload":{"sourceApp":"ios","questionnaire":{"version":"1.0","question":"How to resolve ? ","fb":"Na"}}}"""

    val df = Seq(data).toDF("jsonCol")
    df.show(false)
    df.printSchema()

    /**
      * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |jsonCol                                                                                                                                                                                                                              |
      * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |{"Id":"31279605299","Type":"12121212","client":"Checklist _API","eventTime":"2020-03-17T15:50:30.640Z","eventType":"Event","payload":{"sourceApp":"ios","questionnaire":{"version":"1.0","question":"How to resolve ? ","fb":"Na"}}} |
      * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      *
      * root
      * |-- jsonCol: string (nullable = true)
      */

    df.select(json_tuple(col("jsonCol"), "Id", "Type", "client", "eventTime", "eventType", "payload"))
      .show(false)

    /**
      * +-----------+--------+--------------+------------------------+-----+----------------------------------------------------------------------------------------------+
      * |c0         |c1      |c2            |c3                      |c4   |c5                                                                                            |
      * +-----------+--------+--------------+------------------------+-----+----------------------------------------------------------------------------------------------+
      * |31279605299|12121212|Checklist _API|2020-03-17T15:50:30.640Z|Event|{"sourceApp":"ios","questionnaire":{"version":"1.0","question":"How to resolve ? ","fb":"Na"}}|
      * +-----------+--------+--------------+------------------------+-----+----------------------------------------------------------------------------------------------+
      */
    df.select(schema_of_json(data).as("schema")).show(false)

    /**
      * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |schema                                                                                                                                                                       |
      * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      * |struct<Id:string,Type:string,client:string,eventTime:string,eventType:string,payload:struct<questionnaire:struct<fb:string,question:string,version:string>,sourceApp:string>>|
      * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      */

    val processed = df.select(
      expr("from_json(jsonCol, 'struct<Id:string,Type:string,client:string,eventTime:string, eventType:string," +
        "payload:struct<questionnaire:struct<fb:string,question:string,version:string>,sourceApp:string>>')")
        .as("json_converted"))
    processed.show(false)
    processed.printSchema()

    //
    //    +-------------------------------------------------------------------------------------------------------------+
    //    |json_converted                                                                                               |
    //    +-------------------------------------------------------------------------------------------------------------+
    //    |[31279605299, 12121212, Checklist _API, 2020-03-17T15:50:30.640Z, Event, [[Na, How to resolve ? , 1.0], ios]]|
    //    +-------------------------------------------------------------------------------------------------------------+
    //
    //    root
    //    |-- json_converted: struct (nullable = true)
    //    |    |-- Id: string (nullable = true)
    //    |    |-- Type: string (nullable = true)
    //    |    |-- client: string (nullable = true)
    //    |    |-- eventTime: string (nullable = true)
    //    |    |-- eventType: string (nullable = true)
    //    |    |-- payload: struct (nullable = true)
    //    |    |    |-- questionnaire: struct (nullable = true)
    //    |    |    |    |-- fb: string (nullable = true)
    //    |    |    |    |-- question: string (nullable = true)
    //    |    |    |    |-- version: string (nullable = true)
    //    |    |    |-- sourceApp: string (nullable = true)
    //
  }
  // ############################################################################################################

  @Test
  def test62050145(): Unit = {
    val  data =
      """
        | {
        |	"student": {
        |		"name": "kaleem",
        |		"rollno": "12"
        |	}
        |}
      """.stripMargin
    val df = spark.read.json(Seq(data).toDS())
    df.show(false)
    println(df.schema("student"))

    /**
      * +------------+
      * |student     |
      * +------------+
      * |[kaleem, 12]|
      * +------------+
      *
      * StructField(student,StructType(StructField(name,StringType,true), StructField(rollno,StringType,true)),true)
      */

    val processedDf = df.withColumn("student",
      expr("named_struct('student_details', student)")
    )
    processedDf.show(false)
    println(processedDf.schema("student"))

    //
    //    +--------------+
    //    |student       |
    //    +--------------+
    //    |[[kaleem, 12]]|
    //    +--------------+
    //
    //    StructField(student,StructType(StructField(student_details,StructType(StructField(name,StringType,true), StructField(rollno,StringType,true)),true)),false)
    //
  }

  // ############################################################################################################

  @Test
  def test62094520(): Unit = {
    val mapConcat = udf((map1: Map[String, Int], map2: Map[String, Int]) => {

      val finalMap = mutable.Map.empty[String, mutable.ArrayBuffer[Int]]
      map1.foreach { case (key: String, value: Int) =>
        if (finalMap.contains(key))
          finalMap(key) :+ key
        else finalMap.put(key, mutable.ArrayBuffer(value))
      }
      map2.foreach { case (key: String, value: Int) =>
        if (finalMap.contains(key))
          finalMap(key) :+ key
        else finalMap.put(key, mutable.ArrayBuffer(value))
      }
      finalMap.mapValues(_.max)
    })
    spark.udf.register("my_map_concat", mapConcat)
    spark.range(2).selectExpr("map('a',1,'b',0)","map('a',0,'c',1)",
      "my_map_concat(map('a',1,'b',0),map('a',0,'c',1))")
      .show(false)

    /**
      * +----------------+----------------+-------------------------------------+
      * |map(a, 1, b, 0) |map(a, 0, c, 1) |UDF(map(a, 1, b, 0), map(a, 0, c, 1))|
      * +----------------+----------------+-------------------------------------+
      * |[a -> 1, b -> 0]|[a -> 0, c -> 1]|[b -> 0, a -> 1, c -> 1]             |
      * |[a -> 1, b -> 0]|[a -> 0, c -> 1]|[b -> 0, a -> 1, c -> 1]             |
      * +----------------+----------------+-------------------------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62108794(): Unit = {
    val data=
      """
        |{
        |	"goods": [{
        |		"brand_id": ["brand1", "brand2", "brand3"],
        |		"product_id": "product1"
        |	}]
        |}
      """.stripMargin
    val df = spark.read.json(Seq(data).toDS())
    df.show(false)
    df.printSchema()
    df.createOrReplaceTempView("goodsInfo")

    /**
      * +--------------------------------------+
      * |goods                                 |
      * +--------------------------------------+
      * |[[[brand1, brand2, brand3], product1]]|
      * +--------------------------------------+
      *
      * root
      * |-- goods: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- brand_id: array (nullable = true)
      * |    |    |    |-- element: string (containsNull = true)
      * |    |    |-- product_id: string (nullable = true)
      */

    // filter Dataframe by product_id
    spark.sql("select * from goodsInfo where array_contains(goods.product_id, 'product1')").show(false)

    /**
      * +--------------------------------------+
      * |goods                                 |
      * +--------------------------------------+
      * |[[[brand1, brand2, brand3], product1]]|
      * +--------------------------------------+
      */
    // filter Dataframe by brand_id which is an array within array..
    // positive case
    spark.sql("select * from goodsInfo where array_contains(flatten(goods.brand_id), 'brand3')")
      .show(false)

    /**
      * +--------------------------------------+
      * |goods                                 |
      * +--------------------------------------+
      * |[[[brand1, brand2, brand3], product1]]|
      * +--------------------------------------+
      */
    // negative case
    spark.sql("select * from goodsInfo where array_contains(flatten(goods.brand_id), 'brand4')")
      .show(false)

    /**
      * +-----+
      * |goods|
      * +-----+
      * +-----+
      */

  }

  // ############################################################################################################

  @Test
  def test62107880(): Unit = {
    val data =
      """
        |id|      tim|  price|qty|qtyChg
        | 1|31951.509|  0.370|  1|     1
        | 2|31951.515|145.380|100|   100
        | 3|31951.519|149.370|100|   100
        | 4|31951.520|144.370|100|   100
        | 5|31951.520|149.370|300|   200
        | 6|31951.520|119.370|  5|     5
        | 7|31951.521|149.370|400|   100
        | 8|31951.522|109.370| 50|    50
        | 9|31951.522|149.370|410|    10
        |10|31951.522|144.370|400|   300
        |11|31951.522|149.870| 50|    50
        |12|31951.524|149.370|610|   200
        |13|31951.526|135.130| 22|    22
        |14|31951.527|149.370|750|   140
        |15|31951.528| 89.370|100|   100
        |16|31951.528|145.870| 50|    50
        |17|31951.528|139.370|100|   100
        |18|31951.531|149.370|769|    19
        |19|31951.531|144.370|410|    10
        |20|31951.538|149.370|869|   100
      """.stripMargin

    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS)
    df.show(false)
    df.printSchema()
    /**
      * +---+---------+------+---+------+
      * |id |tim      |price |qty|qtyChg|
      * +---+---------+------+---+------+
      * |1  |31951.509|0.37  |1  |1     |
      * |2  |31951.515|145.38|100|100   |
      * |3  |31951.519|149.37|100|100   |
      * |4  |31951.52 |144.37|100|100   |
      * |5  |31951.52 |149.37|300|200   |
      * |6  |31951.52 |119.37|5  |5     |
      * |7  |31951.521|149.37|400|100   |
      * |8  |31951.522|109.37|50 |50    |
      * |9  |31951.522|149.37|410|10    |
      * |10 |31951.522|144.37|400|300   |
      * |11 |31951.522|149.87|50 |50    |
      * |12 |31951.524|149.37|610|200   |
      * |13 |31951.526|135.13|22 |22    |
      * |14 |31951.527|149.37|750|140   |
      * |15 |31951.528|89.37 |100|100   |
      * |16 |31951.528|145.87|50 |50    |
      * |17 |31951.528|139.37|100|100   |
      * |18 |31951.531|149.37|769|19    |
      * |19 |31951.531|144.37|410|10    |
      * |20 |31951.538|149.37|869|100   |
      * +---+---------+------+---+------+
      *
      * root
      * |-- id: integer (nullable = true)
      * |-- tim: double (nullable = true)
      * |-- price: double (nullable = true)
      * |-- qty: integer (nullable = true)
      * |-- qtyChg: integer (nullable = true)
      */
    // what is the highest price so far at every moment.
    val w = Window.orderBy("tim").rangeBetween(Window.unboundedPreceding, Window.currentRow)
    val w1 = Window.orderBy("tim")

    val processedDF = df.withColumn("maxPriceQty", max(struct(col("price"), col("qty"))).over(w))
      .withColumn("secondMaxPriceQty", lag(col("maxPriceQty"), 1).over(w1))
      .withColumn("top1price", col("maxPriceQty.price"))
      .withColumn("top1priceQty", col("maxPriceQty.qty"))
      .withColumn("top2price", col("secondMaxPriceQty.price"))
      .withColumn("top2priceQty", col("secondMaxPriceQty.qty"))
    processedDF.show(false)

    /**
      * +---+---------+------+---+------+-------------+-----------------+---------+------------+---------+------------+
      * |id |tim      |price |qty|qtyChg|maxPriceQty  |secondMaxPriceQty|top1price|top1priceQty|top2price|top2priceQty|
      * +---+---------+------+---+------+-------------+-----------------+---------+------------+---------+------------+
      * |1  |31951.509|0.37  |1  |1     |[0.37, 1]    |null             |0.37     |1           |null     |null        |
      * |2  |31951.515|145.38|100|100   |[145.38, 100]|[0.37, 1]        |145.38   |100         |0.37     |1           |
      * |3  |31951.519|149.37|100|100   |[149.37, 100]|[145.38, 100]    |149.37   |100         |145.38   |100         |
      * |4  |31951.52 |144.37|100|100   |[149.37, 300]|[149.37, 100]    |149.37   |300         |149.37   |100         |
      * |5  |31951.52 |149.37|300|200   |[149.37, 300]|[149.37, 300]    |149.37   |300         |149.37   |300         |
      * |6  |31951.52 |119.37|5  |5     |[149.37, 300]|[149.37, 300]    |149.37   |300         |149.37   |300         |
      * |7  |31951.521|149.37|400|100   |[149.37, 400]|[149.37, 300]    |149.37   |400         |149.37   |300         |
      * |8  |31951.522|109.37|50 |50    |[149.87, 50] |[149.37, 400]    |149.87   |50          |149.37   |400         |
      * |9  |31951.522|149.37|410|10    |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |10 |31951.522|144.37|400|300   |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |11 |31951.522|149.87|50 |50    |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |12 |31951.524|149.37|610|200   |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |13 |31951.526|135.13|22 |22    |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |14 |31951.527|149.37|750|140   |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |15 |31951.528|89.37 |100|100   |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |16 |31951.528|145.87|50 |50    |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |17 |31951.528|139.37|100|100   |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |18 |31951.531|149.37|769|19    |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |19 |31951.531|144.37|410|10    |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * |20 |31951.538|149.37|869|100   |[149.87, 50] |[149.87, 50]     |149.87   |50          |149.87   |50          |
      * +---+---------+------+---+------+-------------+-----------------+---------+------------+---------+------------+
      */
  }


  // ############################################################################################################

  @Test
  def test62104470(): Unit = {
    val df = Seq(("2020-05-21","x",1),("2020-05-21","y",2),("2020-05-22","x",3),("2020-05-22","y",4),("2020-05-23","x",
      5), ("2020-05-23","y",6)).toDF("dt","A","B")

    df.show(false)
    df.printSchema()
    /**
      * +----------+---+---+
      * |dt        |A  |B  |
      * +----------+---+---+
      * |2020-05-21|x  |1  |
      * |2020-05-21|y  |2  |
      * |2020-05-22|x  |3  |
      * |2020-05-22|y  |4  |
      * |2020-05-23|x  |5  |
      * |2020-05-23|y  |6  |
      * +----------+---+---+
      *
      * root
      * |-- dt: string (nullable = true)
      * |-- A: string (nullable = true)
      * |-- B: integer (nullable = false)
      */

    val w = Window.partitionBy("A").orderBy("dt")
    df.withColumn("previusB", lag("B", 1, 0).over(w))
      .withColumn("sum", col("B") + col("previusB"))
      .orderBy("dt")
      .show(false)

    /**
      * +----------+---+---+--------+---+
      * |dt        |A  |B  |previusB|sum|
      * +----------+---+---+--------+---+
      * |2020-05-21|x  |1  |null    |1  |
      * |2020-05-21|y  |2  |null    |2  |
      * |2020-05-22|x  |3  |1       |4  |
      * |2020-05-22|y  |4  |2       |6  |
      * |2020-05-23|x  |5  |3       |8  |
      * |2020-05-23|y  |6  |4       |10 |
      * +----------+---+---+--------+---+
      */

  }

  // ############################################################################################################

  @Test
  def test62130128(): Unit = {
    val data =
      """
        |1234_4567_DigitalDoc_XRay-01.pdf
        |2345_5678_DigitalDoc_CTC-03.png
        |1234_5684_DigitalDoc_XRay-05.pdf
        |1234_3345_DigitalDoc_XRay-02.pdf
      """.stripMargin

    val customSchema = StructType(
      StructField("catg", StringType, true)
        :: StructField("sub_catg", StringType, true)
        :: StructField("doc_name", StringType, true)
        :: StructField("revision_label", StringType, true)
        :: StructField("extension", StringType, true)
        :: Nil
    )
    val df = spark.read.schema(customSchema)
      .option("sep", "_")
      .csv(data.split(System.lineSeparator()).toSeq.toDS())
    df.show(false)
    df.printSchema()

    /**
      * +----+--------+----------+--------------+---------+
      * |catg|sub_catg|doc_name  |revision_label|extension|
      * +----+--------+----------+--------------+---------+
      * |1234|4567    |DigitalDoc|XRay-01.pdf   |null     |
      * |2345|5678    |DigitalDoc|CTC-03.png    |null     |
      * |1234|5684    |DigitalDoc|XRay-05.pdf   |null     |
      * |1234|3345    |DigitalDoc|XRay-02.pdf   |null     |
      * +----+--------+----------+--------------+---------+
      *
      * root
      * |-- catg: string (nullable = true)
      * |-- sub_catg: string (nullable = true)
      * |-- doc_name: string (nullable = true)
      * |-- revision_label: string (nullable = true)
      * |-- extension: string (nullable = true)
      */

    df.withColumn("doc_name", concat_ws("_", col("doc_name"), col("revision_label")))
      .withColumn("extension", substring_index(col("revision_label"), ".", -1))
      .withColumn("revision_label", regexp_extract(col("revision_label"),"""\d+""", 0))
      .show(false)

    /**
      * +----+--------+----------------------+--------------+---------+
      * |catg|sub_catg|doc_name              |revision_label|extension|
      * +----+--------+----------------------+--------------+---------+
      * |1234|4567    |DigitalDoc_XRay-01.pdf|01            |pdf      |
      * |2345|5678    |DigitalDoc_CTC-03.png |03            |png      |
      * |1234|5684    |DigitalDoc_XRay-05.pdf|05            |pdf      |
      * |1234|3345    |DigitalDoc_XRay-02.pdf|02            |pdf      |
      * +----+--------+----------------------+--------------+---------+
      */
  }

  // ############################################################################################################

  @Test
  def test62133517(): Unit = {
    val data =
      """
        |id | name | disconnect_dt_time
        |1  | "a"  | 2020-05-10 00:00:00
        |2  | "b"  | 2020-05-20 00:00:00
      """.stripMargin

    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    /**
      * +---+----+-------------------+
      * |id |name|disconnect_dt_time |
      * +---+----+-------------------+
      * |1  |a   |2020-05-10 00:00:00|
      * |2  |b   |2020-05-20 00:00:00|
      * +---+----+-------------------+
      *
      * root
      * |-- id: integer (nullable = true)
      * |-- name: string (nullable = true)
      * |-- disconnect_dt_time: timestamp (nullable = true)
      */
    df.createOrReplaceTempView("df1")

    val analysisStartDate = "20200515T00:00:00+0000"
    val analysisEndDate = "20200530T00:00:00+0000"
    val fmt = "yyyyMMdd'T'HH:mm:ssZ"
    val processedDF = spark.table("df1")
      .filter(col("disconnect_dt_time").cast("timestamp")
        .between(to_timestamp(lit(analysisStartDate), fmt) , to_timestamp(lit(analysisEndDate), fmt)) )
    processedDF.show(false)

    /**
      * +---+----+-------------------+
      * |id |name|disconnect_dt_time |
      * +---+----+-------------------+
      * |2  |b   |2020-05-20 00:00:00|
      * +---+----+-------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62134528(): Unit = {
    val data =
      """
        |Name1         |   Name2
        |RR Industries |
        |RR Industries |   RR Industries
      """.stripMargin

    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    /**
      * +-------------+-------------+
      * |Name1        |Name2        |
      * +-------------+-------------+
      * |RR Industries|null         |
      * |RR Industries|RR Industries|
      * +-------------+-------------+
      *
      * root
      * |-- Name1: string (nullable = true)
      * |-- Name2: string (nullable = true)
      */
    df.withColumn("Name3(Expected)", concat_ws("", df.columns.map(col).map(c => coalesce(c, lit(""))): _*))
      .show(false)

    /**
      * +-------------+-------------+--------------------------+
      * |Name1        |Name2        |Name3(Expected)           |
      * +-------------+-------------+--------------------------+
      * |RR Industries|null         |RR Industries             |
      * |RR Industries|RR Industries|RR IndustriesRR Industries|
      * +-------------+-------------+--------------------------+
      */
    df.withColumn("Name3(Expected)", concat_ws("", df.columns.map(col): _*))
      .show(false)

    /**
      * +-------------+-------------+--------------------------+
      * |Name1        |Name2        |Name3(Expected)           |
      * +-------------+-------------+--------------------------+
      * |RR Industries|null         |RR Industries             |
      * |RR Industries|RR Industries|RR IndustriesRR Industries|
      * +-------------+-------------+--------------------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62119385(): Unit = {

    val df = spark.range(1, 5)
    df.createOrReplaceTempView("df_view")
    println(spark.catalog.currentDatabase)
    val db: Database = spark.catalog.getDatabase(spark.catalog.currentDatabase)
    val tables: Dataset[Table] = spark.catalog.listTables(db.name)
    tables.show(false)

    /**
      * default
      * +-------+--------+-----------+---------+-----------+
      * |name   |database|description|tableType|isTemporary|
      * +-------+--------+-----------+---------+-----------+
      * |df_view|null    |null       |TEMPORARY|true       |
      * +-------+--------+-----------+---------+-----------+
      */
  }

  // ############################################################################################################

  @Test
  def test62147049(): Unit = {
    val df = spark.range(1,5)
      .withColumn("batch_id", lit(70) + col("id"))

    df.show(false)
    df.printSchema()

    /**
      * +---+--------+
      * |id |batch_id|
      * +---+--------+
      * |1  |71      |
      * |2  |72      |
      * |3  |73      |
      * |4  |74      |
      * +---+--------+
      *
      * root
      * |-- id: long (nullable = false)
      * |-- batch_id: long (nullable = false)
      */

    df.write.partitionBy("batch_id")
      .mode(SaveMode.Overwrite)
      .parquet("/Users/sokale/models/run_1")
    /**
      * $ cd run_1/
      * $ ls -l
      * total 0
      * ............ _SUCCESS
      * ............ batch_id=71
      * ............ batch_id=72
      * ............ batch_id=73
      * ............ batch_id=74
      */

    // read only file with batch_id=73
    //spark.sql.parquet.filterPushdown	true	Enables Parquet filter push-down optimization when set to true.
    spark.read.parquet("/Users/sokale/models/run_1").where(col("batch_id").equalTo(73))
      .show(false)

    /**
      * +---+--------+
      * |id |batch_id|
      * +---+--------+
      * |3  |73      |
      * +---+--------+
      */
    // read all partitions
    val readDF = spark.read.parquet("/Users/sokale/models/run_1")
    readDF.show(false)
    readDF.printSchema()

    /**
      * +---+--------+
      * |id |batch_id|
      * +---+--------+
      * |3  |73      |
      * |2  |72      |
      * |1  |71      |
      * |4  |74      |
      * +---+--------+
      *
      * root
      * |-- id: long (nullable = true)
      * |-- batch_id: integer (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62144985(): Unit = {
    val data =
      """
        |a    | b   |   c
        |cat  | 3-3 |   78-b
        |cat  | 3-3 |   89-0
        |cat  | 4-4 |   78-n
        |dog  | 4-4 |   89-b
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS)
    df.show(false)
    df.printSchema()
    /**
      * +---+---+----+
      * |a  |b  |c   |
      * +---+---+----+
      * |cat|3-3|78-b|
      * |cat|3-3|89-0|
      * |cat|4-4|78-n|
      * |dog|4-4|89-b|
      * +---+---+----+
      *
      * root
      * |-- a: string (nullable = true)
      * |-- b: string (nullable = true)
      * |-- c: string (nullable = true)
      */

    val processedDF = df
      .groupBy("a")
      .agg(
        collect_list(struct(col("b"), col("c"))).as("value"),
        collect_list(col("b")).as("key")
      )
      .withColumn("map", map_from_arrays(col("key"), col("value")))


    processedDF.show(false)
    processedDF.printSchema()

    /**
      * +---+---------------------------------------+---------------+------------------------------------------------------------+
      * |a  |value                                  |key            |map                                                         |
      * +---+---------------------------------------+---------------+------------------------------------------------------------+
      * |cat|[[3-3, 78-b], [3-3, 89-0], [4-4, 78-n]]|[3-3, 3-3, 4-4]|[3-3 -> [3-3, 78-b], 3-3 -> [3-3, 89-0], 4-4 -> [4-4, 78-n]]|
      * |dog|[[4-4, 89-b]]                          |[4-4]          |[4-4 -> [4-4, 89-b]]                                        |
      * +---+---------------------------------------+---------------+------------------------------------------------------------+
      *
      * root
      * |-- a: string (nullable = true)
      * |-- value: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- b: string (nullable = true)
      * |    |    |-- c: string (nullable = true)
      * |-- key: array (nullable = true)
      * |    |-- element: string (containsNull = true)
      * |-- map: map (nullable = true)
      * |    |-- key: string
      * |    |-- value: struct (valueContainsNull = true)
      * |    |    |-- b: string (nullable = true)
      * |    |    |-- c: string (nullable = true)
      */

    processedDF.select(col("a"), to_json(col("map"))).write
      .mode(SaveMode.Overwrite)
      .partitionBy("a")
      .text("/Users/sokale/models/run_2")

    /**
      * File directory and content of file
      * a=cat
      * |-  {"3-3":{"b":"3-3","c":"78-b"},"3-3":{"b":"3-3","c":"89-0"},"4-4":{"b":"4-4","c":"78-n"}}
      * a=dog
      * |-  {"4-4":{"b":"4-4","c":"89-b"}}
      */

  }
  // ############################################################################################################

  @Test
  def test62148704(): Unit = {
    val data =
      """
        | id | age  |   dob
        |  1 | 24   |
        |  2 | 25   |
        |  3 |      | 1/1/1973
        |  4 |      | 6/6/1980
        |  5 | 46   |
        |  6 |      | 1/1/1971
      """.stripMargin

    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    /**
      * +---+----+--------+
      * |id |age |dob     |
      * +---+----+--------+
      * |1  |24  |null    |
      * |2  |25  |null    |
      * |3  |null|1/1/1973|
      * |4  |null|6/6/1980|
      * |5  |46  |null    |
      * |6  |null|1/1/1971|
      * +---+----+--------+
      *
      * root
      * |-- id: integer (nullable = true)
      * |-- age: integer (nullable = true)
      * |-- dob: string (nullable = true)
      */

    df.withColumn("diff",
      coalesce(col("age"),
        round(months_between(current_date(),to_date(col("dob"), "d/M/yyyy"),true).divide(12),2)
      )
    ).show()

    /**
      * +---+----+--------+-----+
      * | id| age|     dob| diff|
      * +---+----+--------+-----+
      * |  1|  24|    null| 24.0|
      * |  2|  25|    null| 25.0|
      * |  3|null|1/1/1973|47.42|
      * |  4|null|6/6/1980|39.99|
      * |  5|  46|    null| 46.0|
      * |  6|null|1/1/1971|49.42|
      * +---+----+--------+-----+
      */

  }

  // ############################################################################################################

  @Test
  def test6214678(): Unit = {
    val startTimeStamp = "02-05-2020 01:00"
    val endTimeStamp = "03-05-2020 02:00"
    spark.range(1).selectExpr(
      s"""
         |explode(sequence(
         |      to_timestamp('$startTimeStamp', 'dd-MM-yyyy HH:mm'),
         |      to_timestamp('$endTimeStamp', 'dd-MM-yyyy HH:mm'),
         |      interval 1 hour
         |    )) as generated_timestamp
      """.stripMargin
    ).show(false)

    /**
      * +-------------------+
      * |generated_timestamp|
      * +-------------------+
      * |2020-05-02 01:00:00|
      * |2020-05-02 02:00:00|
      * |2020-05-02 03:00:00|
      * |2020-05-02 04:00:00|
      * |2020-05-02 05:00:00|
      * |2020-05-02 06:00:00|
      * |2020-05-02 07:00:00|
      * |2020-05-02 08:00:00|
      * |2020-05-02 09:00:00|
      * |2020-05-02 10:00:00|
      * |2020-05-02 11:00:00|
      * |2020-05-02 12:00:00|
      * |2020-05-02 13:00:00|
      * |2020-05-02 14:00:00|
      * |2020-05-02 15:00:00|
      * |2020-05-02 16:00:00|
      * |2020-05-02 17:00:00|
      * |2020-05-02 18:00:00|
      * |2020-05-02 19:00:00|
      * |2020-05-02 20:00:00|
      * +-------------------+
      * only showing top 20 rows
      */
  }

  // ############################################################################################################

  @Test
  def test62166453(): Unit = {
    val data =
      """
        |id | user_id | date     | expense
        |1  | 1       | 20200521 | 200
        |2  | 2       | 20200601 | 100
        |3  | 1       | 20200603 | 90
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    df.createOrReplaceTempView("user_1")
    val data1 =
      """
        |id | user_id | date     | expense
        |1  | 3       | 20200521 | 200
        |2  | 4       | 20200601 | 100
        |3  | 5       | 20200603 | 90
      """.stripMargin
    val stringDS1 = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()
    df.createOrReplaceTempView("user_1")
    df1.createOrReplaceTempView("user_2")

    //    spark.catalog.createTable()
    spark.sql(
      """
        |CREATE VIEW users
        | AS SELECT * from user_1
        | UNION ALL SELECT * from user_2
      """.stripMargin).show(false)
    spark.sql(
      """
        |select user_id, sum(expense) from users
      """.stripMargin).show()
  }

  // ############################################################################################################

  @Test
  def test62177090(): Unit = {
    val data =
      """
        |      Date|Mode
        |2020-05-10|   A
        |2020-05-10|   B
        |2020-05-10|   A
        |2020-05-11|   C
        |2020-05-11|   C
        |2020-05-12|   B
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df = spark.read
      .option("sep", ",")
      //      .option("inferSchema", "true")
      .option("header", "true")
      .csv(stringDS)
    df.show(false)
    df.printSchema()

    val w = Window.partitionBy(to_date(col("Date")))
    df.withColumn("set(Mode)",
      collect_set("mode").over(w)
    ).show(false)

    /**
      * +----------+----+
      * |Date      |Mode|
      * +----------+----+
      * |2020-05-10|A   |
      * |2020-05-10|B   |
      * |2020-05-10|A   |
      * |2020-05-11|C   |
      * |2020-05-11|C   |
      * |2020-05-12|B   |
      * +----------+----+
      *
      * root
      * |-- Date: string (nullable = true)
      * |-- Mode: string (nullable = true)
      *
      * +----------+----+---------+
      * |Date      |Mode|set(Mode)|
      * +----------+----+---------+
      * |2020-05-10|A   |[B, A]   |
      * |2020-05-10|B   |[B, A]   |
      * |2020-05-10|A   |[B, A]   |
      * |2020-05-11|C   |[C]      |
      * |2020-05-11|C   |[C]      |
      * |2020-05-12|B   |[B]      |
      * +----------+----+---------+
      */
  }
  // ############################################################################################################

  @Test
  def test62174098(): Unit = {
    /**
      * file content
      * spark-test-data.json
      * --------------------
      * {"id":1,"name":"abc1"}
      * {"id":2,"name":"abc2"}
      * {"id":3,"name":"abc3"}
      */
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
    // Collect only statistics that do not require scanning the whole table (that is, size in bytes).
    spark.sql("ANALYZE TABLE df COMPUTE STATISTICS NOSCAN")
    spark.sql("DESCRIBE EXTENDED df ").filter(col("col_name") === "Statistics").show(false)

    /**
      * +----------+---------+-------+
      * |col_name  |data_type|comment|
      * +----------+---------+-------+
      * |Statistics|68 bytes |       |
      * +----------+---------+-------+
      */
    spark.sql("ANALYZE TABLE df COMPUTE STATISTICS")
    spark.sql("DESCRIBE EXTENDED df ").filter(col("col_name") === "Statistics").show(false)

    /**
      * +----------+----------------+-------+
      * |col_name  |data_type       |comment|
      * +----------+----------------+-------+
      * |Statistics|68 bytes, 3 rows|       |
      * +----------+----------------+-------+
      */
  }

  // ############################################################################################################

  @Test
  def test62187307(): Unit = {
    val data =
      """
        |emp_id|emp_site             |emp_name
        |1     |Washigton            | Will
        |2     |null                 | null
        |3     |New York             | Norman
        |4     |Iowa                 | Ian
      """.stripMargin
    val stringDS = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS)
    df1.show(false)
    df1.printSchema()
    /**
      * +------+---------+--------+
      * |emp_id|emp_site |emp_name|
      * +------+---------+--------+
      * |1     |Washigton|Will    |
      * |2     |null     |null    |
      * |3     |New York |Norman  |
      * |4     |Iowa     |Ian     |
      * +------+---------+--------+
      *
      * root
      * |-- emp_id: integer (nullable = true)
      * |-- emp_site: string (nullable = true)
      * |-- emp_name: string (nullable = true)
      */


    val data1 =
      """
        |emp_id|emp_site             |emp_name
        |1     |Washigton            | Watson
        |2     |Wisconsin            | Sam
        |3     |New York             | null
        |4     |Illinois             | Ican
        |5     |Pennsylvania         | Patrick
      """.stripMargin
    val stringDS1 = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df2 = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df2.show(false)
    df2.printSchema()
    /**
      * +------+------------+--------+
      * |emp_id|emp_site    |emp_name|
      * +------+------------+--------+
      * |1     |Washigton   |Watson  |
      * |2     |Wisconsin   |Sam     |
      * |3     |New York    |null    |
      * |4     |Illinois    |Ican    |
      * |5     |Pennsylvania|Patrick |
      * +------+------------+--------+
      *
      * root
      * |-- emp_id: integer (nullable = true)
      * |-- emp_site: string (nullable = true)
      * |-- emp_name: string (nullable = true)
      */

    val joiningKey = "emp_id"
    val cols =
      df1.columns.filterNot(_.equals(joiningKey)).zip(df2.columns.filterNot(_.equals(joiningKey)))
        .map { c =>
          val (df1Col, df2Col) = df1.col(c._1) -> df2.col(c._2)
          when(df1Col.isNull && df2Col.isNotNull,
            array(map(lit("to"), df2Col), map(lit("change"), lit("insert"))))
            .when(df1Col.isNotNull && df2Col.isNull,
              array(map(lit("from"), df1Col), map(lit("change"), lit("delete"))))
            .when(df1Col.isNotNull && df2Col.isNotNull && df1Col === df2Col,
              lit(null))
            .when(df1Col.isNull && df2Col.isNull,
              lit(null))
            .otherwise(array(map(lit("from"), df1Col), map(lit("to"), df2Col), map(lit("change"), lit("update"))))
            .as(c._1)

        }

    df1.join(df2, Seq(joiningKey), "outer")
      .select(cols ++ Seq(col(colName = joiningKey)): _*)
      .orderBy(joiningKey)
      .show(false)

    //
    //   +------------------------------------------------------+----------------------------------------------------+------+
    //   |emp_site                                              |emp_name                                            |emp_id|
    //   +------------------------------------------------------+----------------------------------------------------+------+
    //   |null                                                  |[[from -> Will], [to -> Watson], [change -> update]]|1     |
    //   |[[to -> Wisconsin], [change -> insert]]               |[[to -> Sam], [change -> insert]]                   |2     |
    //   |null                                                  |[[from -> Norman], [change -> delete]]              |3     |
    //   |[[from -> Iowa], [to -> Illinois], [change -> update]]|[[from -> Ian], [to -> Ican], [change -> update]]   |4     |
    //   |[[to -> Pennsylvania], [change -> insert]]            |[[to -> Patrick], [change -> insert]]               |5     |
    //   +------------------------------------------------------+----------------------------------------------------+------+
    //

    // in case column is not of type string
    val getExpr = (fromExpr: String, toExpr: String, changeExpr: String) =>
      s"named_struct('from', $fromExpr, 'to', $toExpr, 'change', '$changeExpr')"
    val cols1 =
      df1.columns.filterNot(_.equals(joiningKey)).zip(df2.columns.filterNot(_.equals(joiningKey)))
        .map { c =>
          val (c1, c2) = s"df1.${c._1}" -> s"df2.${c._2}"
          when(expr(s"$c1 is null and $c2 is not null"), expr(getExpr("null", c2, "insert")))
            .when(expr(s"$c1 is not null and $c2 is null"), expr(getExpr(c1, "null", "delete")))
            .when(expr(s"$c1 is not null and $c2 is not null and $c1=$c2"), expr(getExpr("null", "null", "null")))
            .when(expr(s"$c1 is null and $c2 is null"), expr(getExpr("null", "null", "null")))
            .otherwise(expr(getExpr(c1, c2, "update")))
            .as(c._1)
        }

    val processedDF = df1.as("df1").join(df2.as("df2"), Seq(joiningKey), "outer")
      .select(cols1 ++ Seq(col(colName = joiningKey)): _*)
      .orderBy(joiningKey)
    processedDF.show(false)
    processedDF.printSchema()

    /**
      * +------------------------+----------------------+------+
      * |emp_site                |emp_name              |emp_id|
      * +------------------------+----------------------+------+
      * |[,, null]               |[Will, Watson, update]|1     |
      * |[, Wisconsin, insert]   |[, Sam, insert]       |2     |
      * |[,, null]               |[Norman,, delete]     |3     |
      * |[Iowa, Illinois, update]|[Ian, Ican, update]   |4     |
      * |[, Pennsylvania, insert]|[, Patrick, insert]   |5     |
      * +------------------------+----------------------+------+
      *
      * root
      * |-- emp_site: struct (nullable = false)
      * |    |-- from: string (nullable = true)
      * |    |-- to: string (nullable = true)
      * |    |-- change: string (nullable = false)
      * |-- emp_name: struct (nullable = false)
      * |    |-- from: string (nullable = true)
      * |    |-- to: string (nullable = true)
      * |    |-- change: string (nullable = false)
      * |-- emp_id: integer (nullable = true)
      */
  }

  // ############################################################################################################

  @Test
  def test62187989(): Unit = {
    val df = spark.range(2).withColumn("webhooks",
      array(
        struct(lit("index1").as("index"), lit("failed_at1").as("failed_at"),
          lit("status1").as("status"), lit("updated_at1").as("updated_at")),
        struct(lit("index2").as("index"), lit("failed_at2").as("failed_at"),
          lit("status2").as("status"), lit("updated_at2").as("updated_at"))
      )
    )
    df.show(false)
    df.printSchema()
    //
    //    +---+----------------------------------------------------------------------------------------+
    //    |id |webhooks                                                                                |
    //    +---+----------------------------------------------------------------------------------------+
    //    |0  |[[index1, failed_at1, status1, updated_at1], [index2, failed_at2, status2, updated_at2]]|
    //    |1  |[[index1, failed_at1, status1, updated_at1], [index2, failed_at2, status2, updated_at2]]|
    //    +---+----------------------------------------------------------------------------------------+
    //

    val filterList: List[String]= List("index1","status1")
    val (index, status) = filterList.head -> filterList.last
    df.selectExpr( "webhooks",
      s"filter(webhooks, x -> array(x.index, x.status)=array('$index', '$status')) as processed")
      .show(false)

    //
    //   +----------------------------------------------------------------------------------------+--------------------------------------------+
    //   |webhooks                                                                                |processed                                   |
    //   +----------------------------------------------------------------------------------------+--------------------------------------------+
    //   |[[index1, failed_at1, status1, updated_at1], [index2, failed_at2, status2, updated_at2]]|[[index1, failed_at1, status1, updated_at1]]|
    //   |[[index1, failed_at1, status1, updated_at1], [index2, failed_at2, status2, updated_at2]]|[[index1, failed_at1, status1, updated_at1]]|
    //   +----------------------------------------------------------------------------------------+--------------------------------------------+
    //
  }

  // ############################################################################################################

  @Test
  def test62188447(): Unit = {

    val data1 =
      """
        |Class
        |A
        |AA
        |BB
        |AAAA
        |ABA
        |AAAAA
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
      * +-----+
      * |Class|
      * +-----+
      * |A    |
      * |AA   |
      * |BB   |
      * |AAAA |
      * |ABA  |
      * |AAAAA|
      * +-----+
      *
      * root
      * |-- Class: string (nullable = true)
      */

    df1.filter(col("Class").rlike("""^A+$"""))
      .show(false)

    /**
      * +-----+
      * |Class|
      * +-----+
      * |A    |
      * |AA   |
      * |AAAA |
      * |AAAAA|
      * +-----+
      */
  }

  // ############################################################################################################

  @Test
  def test62188667(): Unit = {
    val data =
      """
        |[{
        |    "ItemType": "CONSTRUCTION",
        |    "ItemId": "9169-bd62eac18e73",
        |    "Content": {
        |        "MetadataSetList": [
        |            {
        |                "SetId": "privacy-metadata-set",
        |                "MetadataList": [
        |                    {
        |                        "MetadataValue": "true",
        |                        "MetadataId": "Public"
        |                    }
        |                ]
        |            },
        |            {
        |                "SetId": "asset-metadata-set",
        |                "MetadataList": [
        |                    {
        |                        "MetadataValue": "new upload & edit test",
        |                        "MetadataId": "title"
        |                    },
        |                    {
        |                        "MetadataValue": "someone",
        |                        "MetadataId": "uploader"
        |                    },
        |                    {
        |                        "MetadataValue": "One,Five",
        |                        "MetadataId": "Families"
        |                    },
        |                    {
        |                        "MetadataValue": "@xyzzzzz",
        |                        "MetadataId": "creator"
        |                    }
        |                ]
        |            }
        |        ],
        |        "MetadataType": "UNDER CONSTRUCTION",
        |        "Tenant": "8ef4-0e976f342606"
        |    },
        |    "Version":"1.0",
        |    "IsActive":false,
        |    "Status":"DEPRECATED"
        |}]
      """.stripMargin

    val df = spark.read
      .option("multiline", true)
      .json(Seq(data).toDS())
    df.show(false)
    df.printSchema()

    /**
      * +------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+-----------------+------------+----------+-------+
      * |Content                                                                                                                                                                                                     |IsActive|ItemId           |ItemType    |Status    |Version|
      * +------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+-----------------+------------+----------+-------+
      * |[[[[[Public, true]], privacy-metadata-set], [[[title, new upload & edit test], [uploader, someone], [Families, One,Five], [creator, @xyzzzzz]], asset-metadata-set]], UNDER CONSTRUCTION, 8ef4-0e976f342606]|false   |9169-bd62eac18e73|CONSTRUCTION|DEPRECATED|1.0    |
      * +------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+-----------------+------------+----------+-------+
      *
      * root
      * |-- Content: struct (nullable = true)
      * |    |-- MetadataSetList: array (nullable = true)
      * |    |    |-- element: struct (containsNull = true)
      * |    |    |    |-- MetadataList: array (nullable = true)
      * |    |    |    |    |-- element: struct (containsNull = true)
      * |    |    |    |    |    |-- MetadataId: string (nullable = true)
      * |    |    |    |    |    |-- MetadataValue: string (nullable = true)
      * |    |    |    |-- SetId: string (nullable = true)
      * |    |-- MetadataType: string (nullable = true)
      * |    |-- Tenant: string (nullable = true)
      * |-- IsActive: boolean (nullable = true)
      * |-- ItemId: string (nullable = true)
      * |-- ItemType: string (nullable = true)
      * |-- Status: string (nullable = true)
      * |-- Version: string (nullable = true)
      */

    val mergeMap = udf((arr: mutable.WrappedArray[Map[String, String]]) => {
      val res = mutable.HashMap.empty[String, String]
      arr.foldLeft(res){case (map, next) => next.++:(map)(collection.breakOut)}
    })

    val processedDF = df.select(col("IsActive").as("is_active"),
      col("ItemId").as("item_id"),
      col("ItemType").as("item_type"),
      col("Status").as("status"),
      col("Version").as("version"),
      col("Content.MetadataType").as("metadata_type"),
      col("Content.Tenant").as("tenant"),
      col("Content.MetadataSetList").getItem(0).getField("MetadataList").as("content1"),
      col("Content.MetadataSetList").getItem(1).getField("MetadataList").as("content2")
    ).withColumn("content",
      array_union(
        col("content1"),
        col("content2")
      )
    )
      .withColumn("content", expr("TRANSFORM(content, x -> map(x.MetadataId, x.MetadataValue))"))
      .withColumn("content", mergeMap(col("content")))
      .drop("content1", "content2")

    processedDF.show(false)
    processedDF.printSchema()

    /**
      * +---------+-----------------+------------+----------+-------+------------------+-----------------+-----------------------------------------------------------------------------------------------------------------+
      * |is_active|item_id          |item_type   |status    |version|metadata_type     |tenant           |content                                                                                                          |
      * +---------+-----------------+------------+----------+-------+------------------+-----------------+-----------------------------------------------------------------------------------------------------------------+
      * |false    |9169-bd62eac18e73|CONSTRUCTION|DEPRECATED|1.0    |UNDER CONSTRUCTION|8ef4-0e976f342606|[Families -> One,Five, Public -> true, creator -> @xyzzzzz, title -> new upload & edit test, uploader -> someone]|
      * +---------+-----------------+------------+----------+-------+------------------+-----------------+-----------------------------------------------------------------------------------------------------------------+
      *
      * root
      * |-- is_active: boolean (nullable = true)
      * |-- item_id: string (nullable = true)
      * |-- item_type: string (nullable = true)
      * |-- status: string (nullable = true)
      * |-- version: string (nullable = true)
      * |-- metadata_type: string (nullable = true)
      * |-- tenant: string (nullable = true)
      * |-- content: map (nullable = true)
      * |    |-- key: string
      * |    |-- value: string (valueContainsNull = true)
      */
    processedDF.toJSON
      .show(false)

    //    {
    //      "is_active": false,
    //      "item_id": "9169-bd62eac18e73",
    //      "item_type": "CONSTRUCTION",
    //      "status": "DEPRECATED",
    //      "version": "1.0",
    //      "metadata_type": "UNDER CONSTRUCTION",
    //      "tenant": "8ef4-0e976f342606",
    //      "content": {
    //        "Public": "true",
    //        "Families": "One,Five",
    //        "creator": "@xyzzzzz",
    //        "uploader": "someone",
    //        "title": "new upload & edit test"
    //      }
    //    }
  }

  // ############################################################################################################

  @Test
  def test62204815(): Unit = {
    val data = """109.169.248.247 - - [12/Dec/2015:18:25:11 +0100] GET /administrator/ HTTP/1.1 200 4263 - Mozilla/5.0 (Windows NT 6.0; rv:34.0) Gecko/20100101 Firefox/34.0 -"""
    val df = Seq(data).toDF("header")
    df.show(false)
    df.printSchema()

    val timestamp_pattern= """\[\d{2}\/\w{3}\/\d{4}\:\d{2}\:\d{2}\:\d{2}\s\S+\d{4}]"""
    df.select(regexp_extract(col("header"),timestamp_pattern,0).alias("timestamp"))
      .show(false)
  }

  // ############################################################################################################

  @Test
  def test62210282(): Unit = {
    val data1 =
      """
        |    Timestamp     | RowType |   Value
        | 2020. 6. 5. 8:12 | X       | Null
        | 2020. 6. 5. 8:13 | Y       | Null
        | 2020. 6. 5. 8:14 | Y       | Null
        | 2020. 6. 5. 8:15 | A       | SomeValue
        | 2020. 6. 5. 8:16 | Y       | Null
        | 2020. 6. 5. 8:17 | Y       | Null
        | 2020. 6. 5. 8:18 | X       | Null
        | 2020. 6. 5. 8:19 | Y       | Null
        | 2020. 6. 5. 8:20 | Y       | Null
        | 2020. 6. 6. 8:21 | A       | SomeValue2
        | 2020. 6. 7. 8:22 | Y       | Null
        | 2020. 6. 8. 8:23 | Y       | Null
        | 2020. 6. 9. 8:24 | X       | Null
      """.stripMargin
    val stringDS1 = data1.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "Null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()

    df1.filter(col("RowType").isInCollection(Seq("X", "A")))
      .withColumn("Value", lead("Value", 1).over(Window.orderBy('Timestamp)))
      .show(false)

    /**
      * +----------------+-------+----------+
      * |Timestamp       |RowType|Value     |
      * +----------------+-------+----------+
      * |2020. 6. 5. 8:12|X      |null      |
      * |2020. 6. 5. 8:13|Y      |null      |
      * |2020. 6. 5. 8:14|Y      |null      |
      * |2020. 6. 5. 8:15|A      |SomeValue |
      * |2020. 6. 5. 8:16|Y      |null      |
      * |2020. 6. 5. 8:17|Y      |null      |
      * |2020. 6. 5. 8:18|X      |null      |
      * |2020. 6. 5. 8:19|Y      |null      |
      * |2020. 6. 5. 8:20|Y      |null      |
      * |2020. 6. 6. 8:21|A      |SomeValue2|
      * |2020. 6. 7. 8:22|Y      |null      |
      * |2020. 6. 8. 8:23|Y      |null      |
      * |2020. 6. 9. 8:24|X      |null      |
      * +----------------+-------+----------+
      *
      * root
      * |-- Timestamp: string (nullable = true)
      * |-- RowType: string (nullable = true)
      * |-- Value: string (nullable = true)
      *
      * +----------------+-------+----------+----------+
      * |Timestamp       |RowType|Value     |Value1    |
      * +----------------+-------+----------+----------+
      * |2020. 6. 5. 8:12|X      |null      |SomeValue |
      * |2020. 6. 5. 8:15|A      |SomeValue |null      |
      * |2020. 6. 5. 8:18|X      |null      |SomeValue2|
      * |2020. 6. 6. 8:21|A      |SomeValue2|null      |
      * |2020. 6. 9. 8:24|X      |null      |null      |
      * +----------------+-------+----------+----------+
      */
  }

  // ############################################################################################################

  @Test
  def test62211108(): Unit = {
    val df1=Seq((1,10.0),(1,20.0),(1,40.6),(1,15.6),(1,17.6),(1,25.6),(1,39.6),(2,20.5),
      (2,70.3),(2,69.4),(2,74.4),(2,45.4),(3,60.6),(3,80.6),(4,30.6),(4,90.6))toDF("ID","Count")

    val idBucketMapping=Seq((1,4),(2,3),(3,2),(4,2))toDF("ID","Bucket")

    def doBucketing(bucket_size : Int) =
      (1 until bucket_size).scanLeft(0d)((a, _) => a + (1 / bucket_size.toDouble))

    var res = df1.withColumn("percentile",
      expr(s"percentile_approx(count, array(${doBucketing(3).mkString(", ")})) over(partition by ID)"))

    println(doBucketing(2))

    res.show(false)

    /**
      * +---+-----+------------------+
      * |ID |Count|percentile        |
      * +---+-----+------------------+
      * |2  |20.5 |[20.5, 45.4, 70.3]|
      * |2  |70.3 |[20.5, 45.4, 70.3]|
      * |2  |69.4 |[20.5, 45.4, 70.3]|
      * |2  |74.4 |[20.5, 45.4, 70.3]|
      * |2  |45.4 |[20.5, 45.4, 70.3]|
      * |4  |30.6 |[30.6, 30.6, 90.6]|
      * |4  |90.6 |[30.6, 30.6, 90.6]|
      * |1  |10.0 |[10.0, 17.6, 25.6]|
      * |1  |20.0 |[10.0, 17.6, 25.6]|
      * |1  |40.6 |[10.0, 17.6, 25.6]|
      * |1  |15.6 |[10.0, 17.6, 25.6]|
      * |1  |17.6 |[10.0, 17.6, 25.6]|
      * |1  |25.6 |[10.0, 17.6, 25.6]|
      * |1  |39.6 |[10.0, 17.6, 25.6]|
      * |3  |60.6 |[60.6, 60.6, 80.6]|
      * |3  |80.6 |[60.6, 60.6, 80.6]|
      * +---+-----+------------------+
      */

    import org.apache.spark.sql.catalyst.expressions.aggregate.ApproximatePercentile

    val getPercentage = udf((bucket_size: Int) => doBucketing(bucket_size))
    //    spark.udf.register("get_percentage_array", getPercentage)
    val processedDF = df1.join(idBucketMapping, "ID")
      .withColumn("percentage", getPercentage(col("Bucket")))
      //        .withColumn("percentile", expr(s"TRANSFORM(percentage," +
      //          s" x -> percentile_approx(count, 0.5, ${ApproximatePercentile.DEFAULT_PERCENTILE_ACCURACY}))"))
      .withColumn("percentile",
      expr(s"percentile_approx(count, percentage, ${ApproximatePercentile.DEFAULT_PERCENTILE_ACCURACY})" +
        s" over(partition by ID)"))
    processedDF.show(false)
    processedDF.printSchema()

  }

  // ############################################################################################################

  @Test
  def test62212504(): Unit = {
    val data =
      """
        |{
        |    "group": "1",
        |    "name": "badboi",
        |    "rank": "3",
        |    "fellows": [
        |        {
        |            "name": "David",
        |            "age": "25",
        |            "hobby": "code"
        |        },
        |        {
        |            "name": "John",
        |            "age": "27",
        |            "hobby": "tennis"
        |        },
        |        {
        |            "name": "Anata",
        |            "age": "23",
        |            "hobby": "dance"
        |        }
        |    ]
        |}
      """.stripMargin

    val df = spark.read.option("multiLine", "true").json(Seq(data).toDS())
    df.show(false)
    df.printSchema()

    /**
      * +-----------------------------------------------------------+-----+------+----+
      * |fellows                                                    |group|name  |rank|
      * +-----------------------------------------------------------+-----+------+----+
      * |[[25, code, David], [27, tennis, John], [23, dance, Anata]]|1    |badboi|3   |
      * +-----------------------------------------------------------+-----+------+----+
      *
      * root
      * |-- fellows: array (nullable = true)
      * |    |-- element: struct (containsNull = true)
      * |    |    |-- age: string (nullable = true)
      * |    |    |-- hobby: string (nullable = true)
      * |    |    |-- name: string (nullable = true)
      * |-- group: string (nullable = true)
      * |-- name: string (nullable = true)
      * |-- rank: string (nullable = true)
      */

    val processedDF = df.withColumn("fellows",
      expr("TRANSFORM(fellows, x -> named_struct('ID', md5(x.name), 'NAME', x.name, 'AGE', x.age, 'HOBBY', x.hobby))"))
    processedDF.show(false)
    processedDF.printSchema()

    /**
      * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-----+------+----+
      * |fellows                                                                                                                                                          |group|name  |rank|
      * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-----+------+----+
      * |[[464e07afc9e46359fb480839150595c5, David, 25, code], [61409aa1fd47d4a5332de23cbf59a36f, John, 27, tennis], [540356fa1779480b07d0743763c78159, Anata, 23, dance]]|1    |badboi|3   |
      * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-----+------+----+
      *
      * root
      * |-- fellows: array (nullable = true)
      * |    |-- element: struct (containsNull = false)
      * |    |    |-- ID: string (nullable = true)
      * |    |    |-- NAME: string (nullable = true)
      * |    |    |-- AGE: string (nullable = true)
      * |    |    |-- HOBBY: string (nullable = true)
      * |-- group: string (nullable = true)
      * |-- name: string (nullable = true)
      * |-- rank: string (nullable = true)
      */
    processedDF.toJSON.show(false)

    //    {
    //      "fellows": [{
    //      "ID": "464e07afc9e46359fb480839150595c5",
    //      "NAME": "David",
    //      "AGE": "25",
    //      "HOBBY": "code"
    //    }, {
    //      "ID": "61409aa1fd47d4a5332de23cbf59a36f",
    //      "NAME": "John",
    //      "AGE": "27",
    //      "HOBBY": "tennis"
    //    }, {
    //      "ID": "540356fa1779480b07d0743763c78159",
    //      "NAME": "Anata",
    //      "AGE": "23",
    //      "HOBBY": "dance"
    //    }],
    //      "group": "1",
    //      "name": "badboi",
    //      "rank": "3"
    //    }
  }


  // ############################################################################################################

  @Test
  def test62222257(): Unit = {
    spark.range(3).createOrReplaceTempView("df1")
    println(spark.catalog.listTables().map(_.name).collect()
      .map(table => table -> spark.table(table).count()).mkString(", "))

    /**
      * (df1,3)
      */
    println(spark.catalog.listTables(spark.catalog.currentDatabase).map(_.name).collect()
      .map(table => table -> spark.table(table).count()).mkString(", "))

    /**
      * (df1,3)
      */
  }

  // ############################################################################################################

  @Test
  def test62222717(): Unit = {
    val data =
      """
        |dept_id|user_id|entry_date
        |      3|      1|2020-06-03
        |      3|      2|2020-06-03
        |      3|      3|2020-06-03
        |      3|      4|2020-06-03
        |      3|      1|2020-06-04
        |      3|      1|2020-06-05
      """.stripMargin

    val stringDS1 = data.split(System.lineSeparator())
      .map(_.split("\\|").map(_.replaceAll("""^[ \t]+|[ \t]+$""", "")).mkString(","))
      .toSeq.toDS()
    val df1 = spark.read
      .option("sep", ",")
      //      .option("inferSchema", "true")
      .option("header", "true")
      .option("nullValue", "null")
      .csv(stringDS1)
    df1.show(false)
    df1.printSchema()

    /**
      * +-------+-------+----------+
      * |dept_id|user_id|entry_date|
      * +-------+-------+----------+
      * |3      |1      |2020-06-03|
      * |3      |2      |2020-06-03|
      * |3      |3      |2020-06-03|
      * |3      |4      |2020-06-03|
      * |3      |1      |2020-06-04|
      * |3      |1      |2020-06-05|
      * +-------+-------+----------+
      *
      * root
      * |-- dept_id: string (nullable = true)
      * |-- user_id: string (nullable = true)
      * |-- entry_date: string (nullable = true)
      */

    val w = Window.partitionBy("dept_id", "user_id")
    val latestRec = when(datediff(max(to_date($"entry_date")).over(w), to_date($"entry_date")) =!= lit(0), 0)
      .otherwise(1)
    df1.withColumn("latest_rec", latestRec)
      .orderBy("dept_id", "user_id", "entry_date")
      .show(false)

    /**
      * +-------+-------+----------+----------+
      * |dept_id|user_id|entry_date|latest_rec|
      * +-------+-------+----------+----------+
      * |3      |1      |2020-06-03|0         |
      * |3      |1      |2020-06-04|0         |
      * |3      |1      |2020-06-05|1         |
      * |3      |2      |2020-06-03|1         |
      * |3      |3      |2020-06-03|1         |
      * |3      |4      |2020-06-03|1         |
      * +-------+-------+----------+----------+
      */
  }
  // ############################################################################################################

  @Test
  def test62224198(): Unit = {
    val df = spark.range(1).withColumn("memberurn", lit("urn:li:member:10000012"))
    df.withColumn("member_id",
      expr("""CAST(regexp_extract(memberurn, 'urn:li:member:(\\d+)', 1) AS BIGINT)"""))
      .show(false)

    /**
      * +---+----------------------+---------+
      * |id |memberurn             |member_id|
      * +---+----------------------+---------+
      * |0  |urn:li:member:10000012|10000012 |
      * +---+----------------------+---------+
      */

    df.withColumn("member_id",
      substring_index($"memberurn", ":", -1).cast("bigint"))
      .show(false)

    /**
      * +---+----------------------+---------+
      * |id |memberurn             |member_id|
      * +---+----------------------+---------+
      * |0  |urn:li:member:10000012|10000012 |
      * +---+----------------------+---------+
      */
  }

  // ############################################################################################################

  @Test
  def test62228733(): Unit = {
    /**
      * test/File1.json
      * -----
      * {
      * "Value": 123
      * }
      */
    /**
      * test/File2.json
      * ---------
      * {
      * "Value": {
      * "Value": "On",
      * "ValueType": "State",
      * "IsSystemValue": true
      * }
      * }
      */
    val path = getClass.getResource("/test" ).getPath
    val df = spark.read
      .option("multiLine", true)
      .json(path)

    df.show(false)
    df.printSchema()

    /**
      * +-------------------------------------------------------+
      * |Value                                                  |
      * +-------------------------------------------------------+
      * |{"Value":"On","ValueType":"State","IsSystemValue":true}|
      * |123                                                    |
      * +-------------------------------------------------------+
      *
      * root
      * |-- Value: string (nullable = true)
      */
    df.withColumn("File", substring_index(input_file_name(),"/", -1))
      .withColumn("ValueType", get_json_object(col("Value"), "$.ValueType"))
      .withColumn("IsSystemValue", get_json_object(col("Value"), "$.IsSystemValue"))
      .withColumn("Value", coalesce(get_json_object(col("Value"), "$.Value"), col("Value")))
      .show(false)

    /**
      * +-----+----------+---------+-------------+
      * |Value|File      |ValueType|IsSystemValue|
      * +-----+----------+---------+-------------+
      * |On   |File2.json|State    |true         |
      * |123  |File1.json|null     |null         |
      * +-----+----------+---------+-------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62229725(): Unit = {
    val df = spark.range(1)
      .withColumn("Description", lit("{0} is the 4th biggest"))
      .withColumn("States", lit("Andhra Pradesh"))

    df.show(false)
    df.printSchema()
    /**
      * +---+----------------------+--------------+
      * |id |Description           |States        |
      * +---+----------------------+--------------+
      * |0  |{0} is the 4th biggest|Andhra Pradesh|
      * +---+----------------------+--------------+
      *
      * root
      * |-- id: long (nullable = false)
      * |-- Description: string (nullable = false)
      * |-- States: string (nullable = false)
      */
    val replace1 = udf((s: String, replace: String) => java.text.MessageFormat.format(s, replace))
    df.withColumn("Description", replace1($"Description", $"States"))
      .show(false)

    /**
      * +---+---------------------------------+--------------+
      * |id |Description                      |States        |
      * +---+---------------------------------+--------------+
      * |0  |Andhra Pradesh is the 4th biggest|Andhra Pradesh|
      * +---+---------------------------------+--------------+
      */

    df.withColumn("Description", expr("case when States is null then Description else replace(Description, '{0}', States) end"))
      .show(false)

    /**
      * +---+---------------------------------+--------------+
      * |id |Description                      |States        |
      * +---+---------------------------------+--------------+
      * |0  |Andhra Pradesh is the 4th biggest|Andhra Pradesh|
      * +---+---------------------------------+--------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62239923(): Unit = {
    val data =
      """
        |id | id1 | seq_nbr   |  id2 |orig_id1 | orig_id2
        |1  | 740 |     2     |  217 |    740  |   217
        |1  | 740 |     3     |  216 |    740  |   216
        |1  | 740 |     4     |  216 |    740  |   216
        |1  | 740 |     5     |  217 |    740  |   217
        |1  | 367 |     1     |  217 |    740  |   217
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
      * +---+---+-------+---+--------+--------+
      * |id |id1|seq_nbr|id2|orig_id1|orig_id2|
      * +---+---+-------+---+--------+--------+
      * |1  |740|2      |217|740     |217     |
      * |1  |740|3      |216|740     |216     |
      * |1  |740|4      |216|740     |216     |
      * |1  |740|5      |217|740     |217     |
      * |1  |367|1      |217|740     |217     |
      * +---+---+-------+---+--------+--------+
      *
      * root
      * |-- id: integer (nullable = true)
      * |-- id1: integer (nullable = true)
      * |-- seq_nbr: integer (nullable = true)
      * |-- id2: integer (nullable = true)
      * |-- orig_id1: integer (nullable = true)
      * |-- orig_id2: integer (nullable = true)
      */

    val win = Window.partitionBy("orig_id1", "orig_id2")
    df1.withColumn("orig_seq_nbr",
      min(when($"orig_id1" === $"id1" && $"orig_id2" === $"id2", $"seq_nbr").otherwise(null))
        .over(win)
    ).show()

    /**
      * +---+---+-------+---+--------+--------+------------+
      * | id|id1|seq_nbr|id2|orig_id1|orig_id2|orig_seq_nbr|
      * +---+---+-------+---+--------+--------+------------+
      * |  1|740|      3|216|     740|     216|           3|
      * |  1|740|      4|216|     740|     216|           3|
      * |  1|740|      2|217|     740|     217|           2|
      * |  1|740|      5|217|     740|     217|           2|
      * |  1|367|      1|217|     740|     217|           2|
      * +---+---+-------+---+--------+--------+------------+
      */

    df1.withColumn("orig_seq_nbr",
      expr("min(case when orig_id1=id1 and orig_id2=id2 then seq_nbr else NULL end) " +
        "OVER (PARTITION BY orig_id1, orig_id2) ")
    ).show()

    /**
      * +---+---+-------+---+--------+--------+------------+
      * | id|id1|seq_nbr|id2|orig_id1|orig_id2|orig_seq_nbr|
      * +---+---+-------+---+--------+--------+------------+
      * |  1|740|      3|216|     740|     216|           3|
      * |  1|740|      4|216|     740|     216|           3|
      * |  1|740|      2|217|     740|     217|           2|
      * |  1|740|      5|217|     740|     217|           2|
      * |  1|367|      1|217|     740|     217|           2|
      * +---+---+-------+---+--------+--------+------------+
      */
  }

  // ############################################################################################################

  @Test
  def test62240275(): Unit = {
    val data =
      """
        |date|value
        |   1|19.75
        |   2|15.51
        |   3|20.66
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
      * +----+-----+
      * |date|value|
      * +----+-----+
      * |1   |19.75|
      * |2   |15.51|
      * |3   |20.66|
      * +----+-----+
      *
      * root
      * |-- date: integer (nullable = true)
      * |-- value: double (nullable = true)
      */

    df1.selectExpr(
      s"element_at(array('None','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'), date) as date",
      "value"
    ).show(false)

    /**
      * +----+-----+
      * |date|value|
      * +----+-----+
      * |None|19.75|
      * |Jan |15.51|
      * |Feb |20.66|
      * +----+-----+
      */
  }

  // ############################################################################################################

  trait R {
    def run
  }
  object A extends R {
    def run() = println("Class A")
  }
  object B extends R {
    def run() = println("Class B")
  }
  object C extends R {
    def run() = println("class C")
  }
  @Test
  def test62246488(): Unit = {
    def runUtil(r: Seq[R] = Seq(A, B, C) ) = r.foreach(_.run)
    println("Run-1")
    runUtil(Seq(A))
    println("Run-2")
    runUtil()

    /**
      * Run-1
      * Class A
      * Run-2
      * Class A
      * Class B
      * class C
      */
  }

  // ############################################################################################################
  @Test
  def test62264106(): Unit = {
    val table = spark.sql("select name, age from values ('bob', 1), ('sam', 2), ('bob', 1) T(name,age)")
    table.show(false)
    table.printSchema()
    /**
      * +----+---+
      * |name|age|
      * +----+---+
      * |bob |1  |
      * |sam |2  |
      * |bob |1  |
      * +----+---+
      *
      * root
      * |-- name: string (nullable = false)
      * |-- age: integer (nullable = false)
      */
    val rowArray = table.select($"name", $"age").collect()
    val nameList = rowArray.map(_(0)).toList.distinct
    val ageList = rowArray.map(_(1)).toList.distinct
    println(nameList.mkString(", "))
    println(ageList.mkString(", "))

    /**
      * bob, sam
      * 1, 2
      */

    val row = table.select(
      collect_set($"name").as("name"),
      collect_set($"age").as("age")
    ).head

    val nameSet = row.getAs[Seq[String]]("name")
    val ageSet = row.getAs[Seq[Int]]("age")
    println(nameSet.mkString(", "))
    println(ageSet.mkString(", "))

    /**
      * bob, sam
      * 1, 2
      */
  }

  // ############################################################################################################
  @Test
  def test62262873(): Unit = {

    val workers: Dataset[Worker] = Seq(
      Worker("Bob", id = 1, skills = Array("communication", "teamwork")),
      Worker("Sam", id = 2, skills = Array("self-motivation"))
    ).toDS

    import scala.reflect.runtime.universe._
    def doWork[T : TypeTag](persons: Dataset[T]): Unit = {
      typeOf[T] match {
        case t if t =:= typeOf[Worker] => println("I'm worker")
          persons.as[Worker].filter(_.id == 2).show(false)
        case t if t =:= typeOf[Customer] => println("I'm Customer")
          persons.as[Customer].filter(_.name.contains("B")).show(false)

      }
    }
    doWork(workers)

    /**
      * I'm worker
      * +----+---+-----------------+
      * |name|id |skills           |
      * +----+---+-----------------+
      * |Sam |2  |[self-motivation]|
      * +----+---+-----------------+
      */
  }

  // ############################################################################################################
  @Test
  def test62272099(): Unit = {
    val df = spark.range(1).withColumn("date",
      explode(sequence(to_date(lit("2020-06-09")), to_date(lit("2020-06-20")), expr("interval 1 day")))
    ).withColumn("year", year($"date"))
      .withColumn("month", month($"date"))
      .withColumn("day", dayofmonth($"date"))
    df.show(false)
    df.printSchema()

    /**
      * +---+----------+----+-----+---+
      * |id |date      |year|month|day|
      * +---+----------+----+-----+---+
      * |0  |2020-06-09|2020|6    |9  |
      * |0  |2020-06-10|2020|6    |10 |
      * |0  |2020-06-11|2020|6    |11 |
      * |0  |2020-06-12|2020|6    |12 |
      * |0  |2020-06-13|2020|6    |13 |
      * |0  |2020-06-14|2020|6    |14 |
      * |0  |2020-06-15|2020|6    |15 |
      * |0  |2020-06-16|2020|6    |16 |
      * |0  |2020-06-17|2020|6    |17 |
      * |0  |2020-06-18|2020|6    |18 |
      * |0  |2020-06-19|2020|6    |19 |
      * |0  |2020-06-20|2020|6    |20 |
      * +---+----------+----+-----+---+
      *
      * root
      * |-- id: long (nullable = false)
      * |-- date: date (nullable = false)
      * |-- year: integer (nullable = false)
      * |-- month: integer (nullable = false)
      * |-- day: integer (nullable = false)
      */
    df.repartition(2).write.partitionBy("year", "month", "day")
      .option("header", true)
      .mode(SaveMode.Overwrite)
      .csv("/Users/sokale/models/hive_table")
    /**
      * File structure
      * ---------------
      * year=2020
      * year=2020/month=6
      * year=2020/month=6/day=10
      * |- part...csv files (same part files for all the below directories)
      * year=2020/month=6/day=11
      * year=2020/month=6/day=12
      * year=2020/month=6/day=13
      * year=2020/month=6/day=14
      * year=2020/month=6/day=15
      * year=2020/month=6/day=16
      * year=2020/month=6/day=17
      * year=2020/month=6/day=18
      * year=2020/month=6/day=19
      * year=2020/month=6/day=20
      * year=2020/month=6/day=9
      */

    val csvDF = spark.read.option("header", true)
      .csv("/Users/sokale/models/hive_table")
    //    val readDF = spark.catalog.createTable("my_table", "csv", csvDF.schema, Map.empty[String, String])
    //    spark.catalog.recoverPartitions("my_table")
    //    val readDF = spark.sql(
    //      """
    //        |CREATE EXTERNAL TABLE my_table (date1 String)
    //        |    PARTITIONED BY (year INT, month INT, day INT)
    //        |    ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
    //        |    STORED AS TEXTFILE LOCATION '/Users/sokale/models/hive_table'
    //        |    TBLPROPERTIES ('skip.header.line.count' = '1')
    //      """.stripMargin)

    csvDF.show(false)
    csvDF.printSchema()

    /**
      * +---+----------+----+-----+---+
      * |id |date      |year|month|day|
      * +---+----------+----+-----+---+
      * |0  |2020-06-20|2020|6    |20 |
      * |0  |2020-06-19|2020|6    |19 |
      * |0  |2020-06-09|2020|6    |9  |
      * |0  |2020-06-12|2020|6    |12 |
      * |0  |2020-06-10|2020|6    |10 |
      * |0  |2020-06-15|2020|6    |15 |
      * |0  |2020-06-16|2020|6    |16 |
      * |0  |2020-06-17|2020|6    |17 |
      * |0  |2020-06-13|2020|6    |13 |
      * |0  |2020-06-18|2020|6    |18 |
      * |0  |2020-06-14|2020|6    |14 |
      * |0  |2020-06-11|2020|6    |11 |
      * +---+----------+----+-----+---+
      *
      * root
      * |-- id: string (nullable = true)
      * |-- date: string (nullable = true)
      * |-- year: integer (nullable = true)
      * |-- month: integer (nullable = true)
      * |-- day: integer (nullable = true)
      */
  }

}

sealed trait Person {
  def name: String
}

final case class Customer(override val name: String, email: String)                extends Person
final case class Worker(override val name: String, id: Int, skills: Array[String]) extends Person
