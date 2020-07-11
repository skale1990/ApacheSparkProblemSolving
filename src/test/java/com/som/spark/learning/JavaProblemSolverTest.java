package com.som.spark.learning;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ImmutableList;
import com.twitter.chill.KryoSerializer;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkFiles;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.catalyst.ScalaReflection;
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.catalyst.plans.RightOuter;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.geosparksql.expressions.ST_GeomFromText;
import org.apache.spark.sql.geosparksql.expressions.ST_GeomFromText$;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.util.LongAccumulator;
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator;
import org.datasyslab.geosparksql.UDF.Catalog;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;
import org.apache.spark.ml.linalg.*;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.sql.expressions.Window;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.*;
import scala.Function1;
import scala.Serializable;
import scala.Tuple2;
import scala.Tuple4;
import scala.Tuple5;
import scala.collection.Iterator;
import scala.collection.JavaConversions;
import scala.collection.JavaConverters;
import scala.collection.Seq;
import scala.collection.mutable.Buffer;
import scala.reflect.ClassTag;

import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.datasyslab.geosparksql.utils.GeoSparkSQLRegistrator;

import static scala.collection.JavaConversions.*;
import static scala.collection.JavaConverters.*;

public class JavaProblemSolverTest implements Serializable {

    private static SparkSession spark = SparkSession.builder().master("local[2]")
            .appName("TestSuite")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate();

    @BeforeClass
    public void setupBeforeAllTests() {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
    }

    @BeforeMethod
    public void nameBefore(Method method) {
        System.out.println("\n==========================================================================");
        System.out.println("Test name: " + method.getName());
        System.out.println("Stack Overflow Link: https://stackoverflow.com/questions/" +
                method.getName()
                        .replaceFirst("test", ""));
        System.out.println("===========================================================================\n");
    }

    <T> Buffer<T> toScalaSeq(List<T> list) {
        return JavaConversions.asScalaBuffer(list);
    }

    // ############################################################################################################

    @Test
    public void test62015370() {

        String data = "id  Col_1 Col_2 Col_3 Col_4 Col_5\n" +
                "1    A     B      C     D     E\n" +
                "2    X     Y      Z     P     Q";

        List<String> list = Arrays.stream(data.split(System.lineSeparator()))
                .map(s -> s.replaceAll("\\s+", ","))
                .collect(Collectors.toList());
        Dataset<Row> df1 = spark.read()
                .option("header", true)
                .option("sep", ",")
                .csv(spark.createDataset(list, Encoders.STRING()));
        df1.show();
        df1.printSchema();
        /**
         * +---+-----+-----+-----+-----+-----+
         * | id|Col_1|Col_2|Col_3|Col_4|Col_5|
         * +---+-----+-----+-----+-----+-----+
         * |  1|    A|    B|    C|    D|    E|
         * |  2|    X|    Y|    Z|    P|    Q|
         * +---+-----+-----+-----+-----+-----+
         *
         * root
         *  |-- id: string (nullable = true)
         *  |-- Col_1: string (nullable = true)
         *  |-- Col_2: string (nullable = true)
         *  |-- Col_3: string (nullable = true)
         *  |-- Col_4: string (nullable = true)
         *  |-- Col_5: string (nullable = true)
         */

        String data1 = "id  Col_1 Col_2 Col_3 Col_4 Col_5\n" +
                "1    A     B6     C     D     E\n" +
                "2    X     Y      Z8    P     Q3";

        List<String> list1 = Arrays.stream(data1.split(System.lineSeparator()))
                .map(s -> s.replaceAll("\\s+", ","))
                .collect(Collectors.toList());
        Dataset<Row> df2 = spark.read()
                .option("sep", ",")
                .option("header", true)
                .csv(spark.createDataset(list1, Encoders.STRING()));

        df2.show();
        df2.printSchema();

        /**
         * +---+-----+-----+-----+-----+-----+
         * | id|Col_1|Col_2|Col_3|Col_4|Col_5|
         * +---+-----+-----+-----+-----+-----+
         * |  1|    A|   B6|    C|    D|    E|
         * |  2|    X|    Y|   Z8|    P|   Q3|
         * +---+-----+-----+-----+-----+-----+
         *
         * root
         *  |-- id: string (nullable = true)
         *  |-- Col_1: string (nullable = true)
         *  |-- Col_2: string (nullable = true)
         *  |-- Col_3: string (nullable = true)
         *  |-- Col_4: string (nullable = true)
         *  |-- Col_5: string (nullable = true)
         *
         * */

        List<Column> cols = Arrays.stream(df1.columns())
                .map(c -> {
                    if (c.equalsIgnoreCase("id"))
                        return col("a.id");
                    else
                        return array(toScalaSeq(Arrays.asList(col("a."+c), col("b."+c))).toBuffer()).as(c);
                }).collect(Collectors.toList());
        Dataset<Row> processedDf = df1.as("a").join(df2.as("b"), df1.col("id").equalTo(df2.col("id")))
                .select(toScalaSeq(cols).toBuffer());


        List<Column> cols1 =  Arrays.stream(df1.columns())
                .map(f -> {
                    if (f.equalsIgnoreCase("id"))
                        return expr(f);
                    else
                        return expr("if(size(array_distinct(" + f + "))==1, NULL, " + f + " ) as " + f);
                }).collect(Collectors.toList());

        processedDf.select(toScalaSeq(cols1).toBuffer())
                .show(false);
        /**
         * +---+-----+-------+-------+-----+-------+
         * |id |Col_1|Col_2  |Col_3  |Col_4|Col_5  |
         * +---+-----+-------+-------+-----+-------+
         * |1  |null |[B, B6]|null   |null |null   |
         * |2  |null |null   |[Z, Z8]|null |[Q, Q3]|
         * +---+-----+-------+-------+-----+-------+
         */

    }

    // ############################################################################################################

    @Test
    public void test62066377() {
        String data = "id  Col_1 Col_2 Col_3 Col_4 Col_5\n" +
                "1    A     B      C     D     E\n" +
                "2    X     Y      Z     P     \"\"";

        List<String> list = Arrays.stream(data.split(System.lineSeparator()))
                .map(s -> s.replaceAll("\\s+", ","))
                .collect(Collectors.toList());
        List<StructField> fields = Arrays.stream("id  Col_1 Col_2 Col_3 Col_4 Col_5".split("\\s+"))
                .map(s -> new StructField(s, DataTypes.StringType, true, Metadata.empty()))
                .collect(Collectors.toList());
        Dataset<Row> df1 = spark.read()
                .schema(new StructType(fields.toArray(new StructField[fields.size()])))
                .option("header", true)
                .option("sep", ",")
                .csv(spark.createDataset(list, Encoders.STRING()));
        df1.show();
        df1.printSchema();

        String data1 = "id  Col_1 Col_2 Col_3 Col_4 Col_5\n" +
                "1    A     B      C     D     E\n" +
                "2    X     Y      Z     P     F";

        List<String> list1 = Arrays.stream(data1.split(System.lineSeparator()))
                .map(s -> s.replaceAll("\\s+", ","))
                .collect(Collectors.toList());
        List<StructField> fields1 = Arrays.stream("id  Col_1 Col_2 Col_3 Col_4 Col_5".split("\\s+"))
                .map(s -> new StructField(s, DataTypes.StringType, true, Metadata.empty()))
                .collect(Collectors.toList());
        Dataset<Row> df2 = spark.read()
                .schema(new StructType(fields1.toArray(new StructField[fields.size()])))
                .option("header", true)
                .option("sep", ",")
                .csv(spark.createDataset(list1, Encoders.STRING()));
        df2.show();
        df2.printSchema();
        /**
         * Why is your output telling your fields are nullable = true if they are not ?
         * Why printSchema() doesn't writes "... : string (nullable = false)" if the field are not nullable.
         */
    }

    // ############################################################################################################

    @Test
    public void test62091589() {

        B instanceB = new B();
        instanceB.setA(1);
        instanceB.setB(2);
        C instanceC = new C();
        instanceC.setA(instanceB);
        Encoder<? extends C> encoder = Encoders.bean(instanceC.getClass());
        Function<C, String> toJson = c -> {
            try {
                return new ObjectMapper().writeValueAsString(c);
            } catch (JsonProcessingException e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        };

//        Dataset<Row> json = spark.read().json(spark.createDataset(Collections.singletonList(toJson.apply(instanceC)), Encoders.STRING()));
//
//        json.as()
//        Dataset<C> ds = spark.createDataFrame(Collections.singletonList(RowFactory.create(instanceC)), RowEncoder.apply(encoder.schema()));
//        ds.printSchema();
//        ds.show(false);

//       StructType schema = new StructType().add(
//               new StructField("c", ScalaReflection.schemaFor(instanceB).dataType(), true, Metadata.empty())
//       );
//        Dataset<Row> ds2 = spark.createDataset(Collections.singletonList(instanceC), ExpressionEncoder.apply());
    }

    // ############################################################################################################

    class MyUDF implements UDF1<Long, String> {
        private Map<Long, String> broadCastMap;
        public MyUDF(Broadcast<Map<Long, String>> broadCastMap) {
            this.broadCastMap = broadCastMap.value();
        }
        public String call(Long id) {
            return id +" -> " + broadCastMap.getOrDefault(id, "No mapping");
        }
    }
    @Test
    public void test62121715() {

        Dataset<Row> inputDf = spark.range(1, 5).withColumn("col1", lit("a"));
        inputDf.show(false);
        inputDf.printSchema();
        /**
         * +---+----+
         * |id |col1|
         * +---+----+
         * |1  |a   |
         * |2  |a   |
         * |3  |a   |
         * |4  |a   |
         * +---+----+
         *
         * root
         *  |-- id: long (nullable = false)
         *  |-- col1: string (nullable = false)
         */

        // Create broadcast
        Map<Long, String> map = new HashMap<>();
        map.put(1L, "b");
        map.put(2L, "c");
        Broadcast<Map<Long, String>> broadCastMap = new JavaSparkContext(spark.sparkContext()).broadcast(map);

        UserDefinedFunction myUdf = udf(new MyUDF(broadCastMap), DataTypes.StringType);

        spark.sqlContext().udf().register("myUdf", myUdf);

        inputDf.withColumn("new_col", callUDF("myUdf",
                JavaConverters.asScalaBufferConverter(Collections.singletonList(col("id"))).asScala()))
                .show();
        /**
         * +---+----+---------------+
         * | id|col1|        new_col|
         * +---+----+---------------+
         * |  1|   a|         1 -> b|
         * |  2|   a|         2 -> c|
         * |  3|   a|3 -> No mapping|
         * |  4|   a|4 -> No mapping|
         * +---+----+---------------+
         */

        inputDf.withColumn("new_col", myUdf.apply(col("id")))
                .show();
        /**
         * +---+----+---------------+
         * | id|col1|        new_col|
         * +---+----+---------------+
         * |  1|   a|         1 -> b|
         * |  2|   a|         2 -> c|
         * |  3|   a|3 -> No mapping|
         * |  4|   a|4 -> No mapping|
         * +---+----+---------------+
         */

    }

    // ############################################################################################################

    @Test
    public void test62166849() {
        List<String> new_lst = new ArrayList<>();
        new_lst.add("value_1");
        new_lst.add("value_2");

        Dataset<Row> df = spark.range(1).withColumn("col_1", lit("A"))
                .withColumn("col_2", lit("value_2"))
                .withColumn("col_3", lit("C"));
        Dataset<Row> df_new = df.withColumn("new_column",functions.when(functions.col("col_1").equalTo("A")
                .and(functions.col("col_2").isInCollection(new_lst)), functions.col("col_3"))
                .otherwise(functions.col("col_1"))
        );
        df_new.show(false);

        /**
         * +---+-----+-------+-----+----------+
         * |id |col_1|col_2  |col_3|new_column|
         * +---+-----+-------+-----+----------+
         * |0  |A    |value_2|C    |C         |
         * +---+-----+-------+-----+----------+
         */
        df.withColumn("new_column",functions.when(functions.col("col_1").equalTo("A")
                .and(functions.col("col_2").isin((Object[]) new_lst.toArray())),functions.col("col_3"))
                .otherwise(functions.col("col_1"))).show(false);
        /**
         * +---+-----+-------+-----+----------+
         * |id |col_1|col_2  |col_3|new_column|
         * +---+-----+-------+-----+----------+
         * |0  |A    |value_2|C    |C         |
         * +---+-----+-------+-----+----------+
         */
    }

    // ############################################################################################################

    @Test
    public void test62206832() {
        String data = " firstname|  lastname|  age\n" +
                " John      | Doe      | 21\n" +
                " John.     | Doe.     | 21\n" +
                " Mary.     | William. | 30";

        List<String> list1 = Arrays.stream(data.split(System.lineSeparator()))
                .map(s -> Arrays.stream(s.split("\\|"))
                        .map(s1 -> s1.replaceAll("^[ \t]+|[ \t]+$", ""))
                        .collect(Collectors.joining(","))
                )
                .collect(Collectors.toList());

        Dataset<Row> df2 = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .option("sep", ",")
                .csv(spark.createDataset(list1, Encoders.STRING()));

        df2.show(false);
        df2.printSchema();
        /**
         * +---------+--------+---+
         * |firstname|lastname|age|
         * +---------+--------+---+
         * |John     |Doe     |21 |
         * |John.    |Doe.    |21 |
         * |Mary.    |William.|30 |
         * +---------+--------+---+
         *
         * root
         *  |-- firstname: string (nullable = true)
         *  |-- lastname: string (nullable = true)
         *  |-- age: integer (nullable = true)
         */

        List<Column> allCols = Arrays.stream(df2.columns()).map(functions::col).collect(Collectors.toList());
        // using sha2
        //. The Wikipedia page gives an estimate of the likelihood of a collision. If you run the numbers,
        // you'll see that all harddisks ever produced on Earth can't hold enough 1MB files to get a likelihood of
        // a collision of even 0.01% for SHA-256.
        //
        //Basically, you can simply ignore the possibility.
        df2.withColumn("stringId", sha2(concat_ws(":", toScalaSeq(allCols)), 256))
                .show(false);
        /**
         * run-1
         * +---------+--------+---+----------------------------------------------------------------+
         * |firstname|lastname|age|stringId                                                        |
         * +---------+--------+---+----------------------------------------------------------------+
         * |John     |Doe     |21 |95903bdd538bc48810c367d0cbe59364e10068fd2511c1a0377015b02157ad30|
         * |John.    |Doe.    |21 |52092b925014246e67cc80ce460db8791981775f7e2f7a9fc02eed620f7e84f9|
         * |Mary.    |William.|30 |a782aa33b3a94148fe450b3e251d0a526ecbe83a4e6fbf49781a2f62dbaadc88|
         * +---------+--------+---+----------------------------------------------------------------+
         * run-2
         * +---------+--------+---+----------------------------------------------------------------+
         * |firstname|lastname|age|stringId                                                        |
         * +---------+--------+---+----------------------------------------------------------------+
         * |John     |Doe     |21 |95903bdd538bc48810c367d0cbe59364e10068fd2511c1a0377015b02157ad30|
         * |John.    |Doe.    |21 |52092b925014246e67cc80ce460db8791981775f7e2f7a9fc02eed620f7e84f9|
         * |Mary.    |William.|30 |a782aa33b3a94148fe450b3e251d0a526ecbe83a4e6fbf49781a2f62dbaadc88|
         * +---------+--------+---+----------------------------------------------------------------+
         * run-3
         * +---------+--------+---+----------------------------------------------------------------+
         * |firstname|lastname|age|stringId                                                        |
         * +---------+--------+---+----------------------------------------------------------------+
         * |John     |Doe     |21 |95903bdd538bc48810c367d0cbe59364e10068fd2511c1a0377015b02157ad30|
         * |John.    |Doe.    |21 |52092b925014246e67cc80ce460db8791981775f7e2f7a9fc02eed620f7e84f9|
         * |Mary.    |William.|30 |a782aa33b3a94148fe450b3e251d0a526ecbe83a4e6fbf49781a2f62dbaadc88|
         * +---------+--------+---+----------------------------------------------------------------+
         */

        // using row_number order by all cols
        // performance will degrade since there is no partitionBy clause while using big dataset
        df2.withColumn("number", row_number().over(Window.orderBy(toScalaSeq(allCols))))
                .show(false);
        /**
         * run-1
         * +---------+--------+---+------+
         * |firstname|lastname|age|number|
         * +---------+--------+---+------+
         * |John     |Doe     |21 |1     |
         * |John.    |Doe.    |21 |2     |
         * |Mary.    |William.|30 |3     |
         * +---------+--------+---+------+
         * run-2
         * +---------+--------+---+------+
         * |firstname|lastname|age|number|
         * +---------+--------+---+------+
         * |John     |Doe     |21 |1     |
         * |John.    |Doe.    |21 |2     |
         * |Mary.    |William.|30 |3     |
         * +---------+--------+---+------+
         * run-3
         * +---------+--------+---+------+
         * |firstname|lastname|age|number|
         * +---------+--------+---+------+
         * |John     |Doe     |21 |1     |
         * |John.    |Doe.    |21 |2     |
         * |Mary.    |William.|30 |3     |
         * +---------+--------+---+------+
         */


        // using UUID.nameUUIDFromBytes
        UserDefinedFunction id_udf = udf( (String s) ->
                UUID.nameUUIDFromBytes(
                        s.getBytes(StandardCharsets.UTF_8)
                ).toString(), DataTypes.StringType);

        df2.withColumn("stringId", id_udf.apply(concat_ws(":", toScalaSeq(allCols))))
                .show(false);
        /**
         * run-1
         * +---------+--------+---+------------------------------------+
         * |firstname|lastname|age|stringId                            |
         * +---------+--------+---+------------------------------------+
         * |John     |Doe     |21 |3d319fa5-7a48-3c21-bdb8-f4546a18dffb|
         * |John.    |Doe.    |21 |49ab483f-692d-3e14-aa53-2e35e0cf2a17|
         * |Mary.    |William.|30 |9b758f70-3723-3623-b262-6d200d6111cf|
         * +---------+--------+---+------------------------------------+
         * run-2
         * +---------+--------+---+------------------------------------+
         * |firstname|lastname|age|stringId                            |
         * +---------+--------+---+------------------------------------+
         * |John     |Doe     |21 |3d319fa5-7a48-3c21-bdb8-f4546a18dffb|
         * |John.    |Doe.    |21 |49ab483f-692d-3e14-aa53-2e35e0cf2a17|
         * |Mary.    |William.|30 |9b758f70-3723-3623-b262-6d200d6111cf|
         * +---------+--------+---+------------------------------------+
         * run-3
         * +---------+--------+---+------------------------------------+
         * |firstname|lastname|age|stringId                            |
         * +---------+--------+---+------------------------------------+
         * |John     |Doe     |21 |3d319fa5-7a48-3c21-bdb8-f4546a18dffb|
         * |John.    |Doe.    |21 |49ab483f-692d-3e14-aa53-2e35e0cf2a17|
         * |Mary.    |William.|30 |9b758f70-3723-3623-b262-6d200d6111cf|
         * +---------+--------+---+------------------------------------+
         */

    }

    // ############################################################################################################

    @Test
    public void test62221075() {
        String data = "customerid|        customername|       contactname|             address|       city|postalcode|country\n" +
                "         1| Alfreds Futterkiste|      Maria Anders|       Obere Str. 57|     Berlin|     12209|Germany\n" +
                "         2|Ana Trujillo Empa...|      Ana Trujillo|Avda. de la Const...|M�xico D.F.|      5021| Mexico\n" +
                "         3|Antonio Moreno Ta...|    Antonio Moreno|      Mataderos 2312|M�xico D.F.|      5023| Mexico\n" +
                "         4|     Around the Horn|      Thomas Hardy|     120 Hanover Sq.|     London|   WA1 1DP|     UK\n" +
                "         5|  Berglunds snabbk�p|Christina Berglund|      Berguvsv�gen 8|      Lule�|  S-958 22| Sweden";


        List<String> list1 = Arrays.stream(data.split(System.lineSeparator()))
                .map(s -> Arrays.stream(s.split("\\|"))
                        .map(s1 -> s1.replaceAll("^[ \t]+|[ \t]+$", ""))
                        .collect(Collectors.joining(","))
                )
                .collect(Collectors.toList());

        Dataset<Row> dataset = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .option("sep", ",")
                .csv(spark.createDataset(list1, Encoders.STRING()));
        dataset.show(false);
        dataset.printSchema();

        dataset.createOrReplaceTempView("customers");
        final Dataset<Row> dataset1 = spark.sql("SELECT count(customerid) as count, customerid, country FROM " +
                "customers" +
                " GROUP BY country, customerid HAVING count > 5 ORDER BY count DESC");
        dataset1.show();
    }

    // ############################################################################################################

    @Test
    public void test62308169() {
        Dataset<Row> df = spark.range(2).withColumn("value", lit(2));
        df.show(false);
        df.printSchema();

        /**
         * +---+-----+
         * |id |value|
         * +---+-----+
         * |0  |2    |
         * |1  |2    |
         * +---+-----+
         *
         * root
         *  |-- id: long (nullable = false)
         *  |-- value: integer (nullable = false)
         */
        Map<String, String> map = new HashMap<>();
        for(String column:df.columns())
            map.put(column, "sum");

        List<Column> cols = map.entrySet().stream().map(c -> expr(String.format("%s(%s) as %s", c.getValue(), c.getKey(), c.getKey())))
                .collect(Collectors.toList());


        df.agg(cols.get(0), toScalaSeq(cols.subList(1, cols.size()))).show(false);
        /**
         * +---+-----+
         * |id |value|
         * +---+-----+
         * |1  |4    |
         * +---+-----+
         */
    }

    // ############################################################################################################

    @Test
    public void test62344764() {
        List<Integer> lst = Arrays.asList(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20);
        Dataset<Integer> DF = spark.createDataset(lst, Encoders.INT());
        System.out.println(DF.javaRDD().getNumPartitions());
        MapPartitionsFunction<Integer, Integer> f =
                it -> ImmutableList.of(JavaConverters.asScalaIteratorConverter(it).asScala().length()).iterator();
        Dataset<Integer> integerDataset = DF.repartition(3).mapPartitions(f,
                Encoders.INT());
        integerDataset
                .show(false);
        /**
         * 2
         * +-----+
         * |value|
         * +-----+
         * |6    |
         * |8    |
         * |6    |
         * +-----+
         */

        JavaRDD<Integer> mappartRdd = DF.repartition(3).javaRDD().mapPartitions(it->  Arrays.asList(JavaConversions.asScalaIterator(it).length()).iterator());

//        def createDataFrame(rowRDD: JavaRDD[Row], schema: StructType): DataFrame = {
        StructType schema = new StructType()
                .add(new StructField("value", DataTypes.IntegerType, true, Metadata.empty()));
        Dataset<Row> df = spark.createDataFrame(mappartRdd.map(RowFactory::create), schema);
        df.show(false);
        df.printSchema();

        /**
         * +-----+
         * |value|
         * +-----+
         * |6    |
         * |8    |
         * |6    |
         * +-----+
         *
         * root
         *  |-- value: integer (nullable = true)
         */
    }

    // ############################################################################################################

    @Test
    public void test62398704() {
        String data = "ID | branch | name    | role | age\n" +
                "1  | HQ     | Harry   | DEV  | 24\n" +
                "1  | Berlin | Harry   | QA   | 24\n" +
                "1  | Hungary| Harry   | BA   | 24\n" +
                "2  | HQ     | Chris   | Prod | 39\n" +
                "2  | Chime  | Chris   | Prod | 39\n" +
                "2  | Cornell| Chris   | Acc  | 39\n" +
                "2  | Chappel| Chris   |      | 39";

        List<String> list1 = Arrays.stream(data.split(System.lineSeparator()))
                .map(s -> Arrays.stream(s.split("\\|"))
                        .map(s1 -> s1.replaceAll("^[ \t]+|[ \t]+$", ""))
                        .collect(Collectors.joining(","))
                )
                .collect(Collectors.toList());

        Dataset<Row> dataset = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .option("sep", ",")
                .csv(spark.createDataset(list1, Encoders.STRING()));
        dataset.show(false);
        dataset.printSchema();

        /**
         * +---+-------+-----+----+---+
         * |ID |branch |name |role|age|
         * +---+-------+-----+----+---+
         * |1  |HQ     |Harry|DEV |24 |
         * |1  |Berlin |Harry|QA  |24 |
         * |1  |Hungary|Harry|BA  |24 |
         * |2  |HQ     |Chris|Prod|39 |
         * |2  |Chime  |Chris|Prod|39 |
         * |2  |Cornell|Chris|Acc |39 |
         * |2  |Chappel|Chris|null|39 |
         * +---+-------+-----+----+---+
         *
         * root
         *  |-- ID: integer (nullable = true)
         *  |-- branch: string (nullable = true)
         *  |-- name: string (nullable = true)
         *  |-- role: string (nullable = true)
         *  |-- age: integer (nullable = true)
         */

        dataset.withColumn("role", collect_set(
                when(trim(col("role")).equalTo(lit("")), lit(null)).otherwise(col("role"))
        ).over(Window.partitionBy("ID")))
                .where("branch='HQ'")
                .show(false);
        /**
         * +---+------+-----+-------------+---+
         * |ID |branch|name |role         |age|
         * +---+------+-----+-------------+---+
         * |2  |HQ    |Chris|[Prod, Acc]  |39 |
         * |1  |HQ    |Harry|[QA, BA, DEV]|24 |
         * +---+------+-----+-------------+---+
         */
    }
    // ############################################################################################################

    @Test
    public void test62410606() {
        spark.sql("select ID from values ('1-1'), ('10') T(ID) where ID NOT LIKE '%-%'")
                .show(false);
        /**
         * +---+
         * |ID |
         * +---+
         * |10 |
         * +---+
         */
    }
    // ############################################################################################################

    @Test
    public void test62423748() {
        String data = "   e_key|f_timestamp_day|                 key|               value|f_country|f_os|received_date\n" +
                "  Tryout|     2020-04-01|      item_guid_list|            a^a^a^b |       FR| iOS|   2020-04-01\n" +
                "  Tryout|     2020-04-01|            sku_list|         c^c^d^e^f^f|       FR| iOS|   2020-04-01";

        List<String> list1 = Arrays.stream(data.split(System.lineSeparator()))
                .map(s -> Arrays.stream(s.split("\\|"))
                        .map(s1 -> s1.replaceAll("^[ \t]+|[ \t]+$", ""))
                        .collect(Collectors.joining(","))
                )
                .collect(Collectors.toList());

        Dataset<Row> dataset = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .option("sep", ",")
                .option("nullValue", "null")
                .csv(spark.createDataset(list1, Encoders.STRING()));
        dataset.show(false);
        dataset.printSchema();
        /**
         * +------+-------------------+--------------+-----------+---------+----+-------------------+
         * |e_key |f_timestamp_day    |key           |value      |f_country|f_os|received_date      |
         * +------+-------------------+--------------+-----------+---------+----+-------------------+
         * |Tryout|2020-04-01 00:00:00|item_guid_list|a^a^a^b    |FR       |iOS |2020-04-01 00:00:00|
         * |Tryout|2020-04-01 00:00:00|sku_list      |c^c^d^e^f^f|FR       |iOS |2020-04-01 00:00:00|
         * +------+-------------------+--------------+-----------+---------+----+-------------------+
         *
         * root
         *  |-- e_key: string (nullable = true)
         *  |-- f_timestamp_day: timestamp (nullable = true)
         *  |-- key: string (nullable = true)
         *  |-- value: string (nullable = true)
         *  |-- f_country: string (nullable = true)
         *  |-- f_os: string (nullable = true)
         *  |-- received_date: timestamp (nullable = true)
         */

        dataset.withColumn("value", explode(array_distinct(split(col("value"), "\\^"))))
                .show(false);
        /**
         * +------+-------------------+--------------+-----+---------+----+-------------------+
         * |e_key |f_timestamp_day    |key           |value|f_country|f_os|received_date      |
         * +------+-------------------+--------------+-----+---------+----+-------------------+
         * |Tryout|2020-04-01 00:00:00|item_guid_list|a    |FR       |iOS |2020-04-01 00:00:00|
         * |Tryout|2020-04-01 00:00:00|item_guid_list|b    |FR       |iOS |2020-04-01 00:00:00|
         * |Tryout|2020-04-01 00:00:00|sku_list      |c    |FR       |iOS |2020-04-01 00:00:00|
         * |Tryout|2020-04-01 00:00:00|sku_list      |d    |FR       |iOS |2020-04-01 00:00:00|
         * |Tryout|2020-04-01 00:00:00|sku_list      |e    |FR       |iOS |2020-04-01 00:00:00|
         * |Tryout|2020-04-01 00:00:00|sku_list      |f    |FR       |iOS |2020-04-01 00:00:00|
         * +------+-------------------+--------------+-----+---------+----+-------------------+
         */
    }

    // ############################################################################################################

    @Test
    public void test62424833() {
        LongAccumulator longAccum = spark.sparkContext().longAccumulator("my accum");
        Dataset<Row> df = spark.range(100).withColumn("x", lit("x"));

        //access in map
        df.map((MapFunction<Row, Row>) row -> {
            longAccum.add(1);
            return  row;
        }, RowEncoder.apply(df.schema()))
                .count();

        // accumulator value
        System.out.println(longAccum.value()); // 100

        longAccum.reset();
        // access in for each
        df.foreach((ForeachFunction<Row>) row -> longAccum.add(1));

        // accumulator value
        System.out.println(longAccum.value()); // 100

        longAccum.reset();
        /**
         * streaming dataframe from csv dir
         * test.csv
         * --------
         * csv
         * id,name
         * 1,bob
         * 2,smith
         * 3,jam
         * 4,dwayne
         * 5,mike
         */
        String fileDir = getClass().getResource("/" + "csv").getPath();
        StructType schema = new StructType()
                .add(new StructField("id", DataTypes.LongType, true, Metadata.empty()))
                .add(new StructField("name", DataTypes.StringType, true, Metadata.empty()));
        Dataset<Row> json = spark.readStream().schema(schema).option("header", true).csv(fileDir);

        StreamingQuery streamingQuery = json
                .map((MapFunction<Row, Row>) row -> {
                    longAccum.add(1);
                    return row;
                }, RowEncoder.apply(df.schema()))
                .writeStream()
                .format("console").start();
        streamingQuery.processAllAvailable();

        // accumulator value
        System.out.println(longAccum.value()); // 5

    }

    // ############################################################################################################

    public static MultilayerPerceptronClassificationModel trainModel_MPC(SparkSession session, JavaRDD<LabeledPoint> data)
    {
        // specify layers for the neural network:
        // input layer of size 4 (features), two intermediate of size 5 and 4
        // and output of size 3 (classes)

        int[] layers = {4, 5, 5, 3};
        MultilayerPerceptronClassifier model = new MultilayerPerceptronClassifier().setLayers(layers)
                .setSeed(System.currentTimeMillis()).setBlockSize(128).setMaxIter(200);

        Dataset<Row> dataset = session.createDataFrame(data.rdd(), LabeledPoint.class);

        return model.fit(dataset);

    }
    // ############################################################################################################

    @Test
    public void test62476206() {
        // ref https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/1019862370390522/4413065072037724/latest.html

        Tuple5<PipelineModel, Pipeline, CrossValidatorModel, Dataset<Row>, Dataset<Row>> tuple5 =
                new ProblemSolverMay2020Test().mlOnIrisData();
        // create LabelPoint
        CrossValidatorModel crossValidatorModel = tuple5._3();
        Dataset<Row> trainData =  tuple5._4();
        JavaRDD<LabeledPoint> labeledPointJavaRDD = crossValidatorModel.transform(trainData).toJavaRDD()
                .map(r -> LabeledPoint.apply(r.getAs("indexedLabel"), r.getAs("indexedFeatures")));
        MultilayerPerceptronClassificationModel perceptronClassificationModel = trainModel_MPC(spark, labeledPointJavaRDD);

        Dataset<Row> testData =  tuple5._5();
        Dataset<Row> rowDataset = crossValidatorModel.transform(trainData)
                .selectExpr("indexedLabel as label", "indexedFeatures as features");
        perceptronClassificationModel.transform(rowDataset)
        .show(false);
    }

    // ############################################################################################################
    @Test
    public void test62586603() {

        StructType schema = new StructType()
                .add(new StructField("id", DataTypes.IntegerType, true, Metadata.empty()))
                .add(new StructField("name", DataTypes.StringType, true, Metadata.empty()));

        JavaRDD<Tuple2<Row, Row>> tuple2JavaRDD = new JavaSparkContext(spark.sparkContext())
                .parallelize(
                        Arrays.asList(Tuple2.apply(RowFactory.create(1), RowFactory.create("a")),
                                Tuple2.apply(RowFactory.create(2), RowFactory.create("b")))
                );
        JavaRDD<Row> rowJavaRDD1 = tuple2JavaRDD.map(t -> Row$.MODULE$.merge(
                toScalaSeq(Arrays.asList(t._1, t._2))
        ));
        Dataset<Row> df1 = spark.createDataFrame(rowJavaRDD1, schema);
        df1.show(false);
        df1.printSchema();
        /**
         * +---+----+
         * |id |name|
         * +---+----+
         * |1  |a   |
         * |2  |b   |
         * +---+----+
         *
         * root
         *  |-- id: integer (nullable = true)
         *  |-- name: string (nullable = true)
         */

        JavaRDD<Tuple2<Integer, String>> resultRDD = new JavaSparkContext(spark.sparkContext())
                .parallelize(Arrays.asList(Tuple2.apply(1, "a"), Tuple2.apply(2, "b")));
        JavaRDD<Row> rowJavaRDD = resultRDD.map(Row$.MODULE$::fromTuple);
        Dataset<Row> dataFrame = spark.createDataFrame(rowJavaRDD, schema);
        dataFrame.show(false);
        dataFrame.printSchema();
        /**
         * +---+----+
         * |id |name|
         * +---+----+
         * |1  |a   |
         * |2  |b   |
         * +---+----+
         *
         * root
         *  |-- id: integer (nullable = true)
         *  |-- name: string (nullable = true)
         */
    }

    // ############################################################################################################

    StructType getSchema(){
        StructField[] structFields = new StructField[]{
                new StructField("id", DataTypes.LongType, true, Metadata.empty()),
                new StructField("name", DataTypes.StringType, true, Metadata.empty()),
                new StructField("cat", DataTypes.StringType, true, Metadata.empty()),
                new StructField("tag", DataTypes.createArrayType(DataTypes.StringType), true, Metadata.empty())

        };
        return new StructType(structFields);
    }
    @Test
    public void test62664789() {
        Dataset<Row> dataFrame = spark.createDataFrame(Arrays.asList(
                RowFactory.create(1L, "foo", "cat1", Arrays.asList("tag1", "tag2"))
        ), getSchema());
        dataFrame.show(false);
        dataFrame.printSchema();
        /**
         * +---+----+----+------------+
         * |id |name|cat |tag         |
         * +---+----+----+------------+
         * |1  |foo |cat1|[tag1, tag2]|
         * +---+----+----+------------+
         *
         * root
         *  |-- id: long (nullable = true)
         *  |-- name: string (nullable = true)
         *  |-- cat: string (nullable = true)
         *  |-- tag: array (nullable = true)
         *  |    |-- element: string (containsNull = true)
         */
        Dataset<DealFeedSchema> dealFeedSchemaDataset = dataFrame.as(Encoders.bean(DealFeedSchema.class));
        dealFeedSchemaDataset.show(false);
        dealFeedSchemaDataset.printSchema();
        /**
         * +---+----+----+------------+
         * |id |name|cat |tag         |
         * +---+----+----+------------+
         * |1  |foo |cat1|[tag1, tag2]|
         * +---+----+----+------------+
         *
         * root
         *  |-- id: long (nullable = true)
         *  |-- name: string (nullable = true)
         *  |-- cat: string (nullable = true)
         *  |-- tag: array (nullable = true)
         *  |    |-- element: string (containsNull = true)
         */

        // SO = 62788484
        Dataset <Row> dataframe1 = dataFrame;
        Dataset <Row> dataframe2 = dataFrame;
        Dataset <Row> dataframe3 = dataFrame;
        Dataset <Row> df= dataframe1.filter(when(col("diffDate").lt(3888),
                dataframe1.join(dataframe2,
                dataframe2.col("id_device").equalTo(dataframe1.col("id_device")).
                        and(dataframe2.col("id_vehicule").equalTo(dataframe1.col("id_vehicule"))).
                        and(dataframe2.col("tracking_time").lt(dataframe1.col("tracking_time"))).
                        and(dataframe1.col("diffDate").lt(3888))
                )
                        .orderBy(dataframe2.col("tracking_time").desc())
        ).
                otherwise(dataframe1.join(dataframe3,
                        dataframe3.col("id_device").equalTo(dataframe1.col("id_device")).
                                and(dataframe3.col("id_vehicule").equalTo(dataframe1.col("id_vehicule"))).
                                and(dataframe3.col("tracking_time").lt(dataframe1.col("tracking_time")))).orderBy(dataframe3.col("tracking_time").desc())));

    }
    // ############################################################################################################
    @Test
    public void test62837467() {
        SparkSession sparkSession = SparkSession.builder()
                .config("spark.serializer", KryoSerializer.class.getName())
                .config("spark.kryo.registrator", GeoSparkKryoRegistrator.class.getName())
                .master("local[*]")
                .appName("myGeoSparkSQLdemo")
                .getOrCreate();

        // register all functions from geospark-sql_2.3 to sparkSession
        GeoSparkSQLRegistrator.registerAll(sparkSession);
        try {
            System.out.println(sparkSession.catalog().getFunction("ST_Geomfromtext"));
            // Function[name='ST_GeomFromText', className='org.apache.spark.sql.geosparksql.expressions.ST_GeomFromText$', isTemporary='true']
        } catch (Exception e) {
            e.printStackTrace();
        }
        // https://datasystemslab.github.io/GeoSpark/api/sql/GeoSparkSQL-Function/
        Dataset<Row> dataframe = sparkSession.sql("select ST_GeomFromText('POINT(-7.07378166 33.826661)')");
        dataframe.show(false);
        dataframe.printSchema();
        /**
         * +---------------------------------------------+
         * |st_geomfromtext(POINT(-7.07378166 33.826661))|
         * +---------------------------------------------+
         * |POINT (-7.07378166 33.826661)                |
         * +---------------------------------------------+
         */

        // using longitude and latitude column from existing dataframe
        Dataset<Row> df = sparkSession.sql("select -7.07378166 as longitude, 33.826661 as latitude");
        df.withColumn("ST_Geomfromtext ",
                expr("ST_GeomFromText(CONCAT('POINT(',longitude,' ',latitude,')'))"))
        .show(false);
        /**
         * +-----------+---------+-----------------------------+
         * |longitude  |latitude |ST_Geomfromtext              |
         * +-----------+---------+-----------------------------+
         * |-7.07378166|33.826661|POINT (-7.07378166 33.826661)|
         * +-----------+---------+-----------------------------+
         */
    }

}