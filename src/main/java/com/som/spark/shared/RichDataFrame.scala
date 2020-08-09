package com.som.spark.shared

import org.apache.spark.sql.functions.expr
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{NumericType, StringType, StructField, StructType}

import scala.language.implicitConversions

class RichDataFrame(ds: DataFrame) {
  def statSummary(statistics: String*): DataFrame = {
    val defaultStatistics = Seq("max", "min", "mean", "std", "skewness", "kurtosis")
    val statFunctions = if (statistics.nonEmpty) statistics else defaultStatistics
    val selectedCols = ds.schema
      .filter(a => a.dataType.isInstanceOf[NumericType] || a.dataType.isInstanceOf[StringType])
      .map(_.name)

    val percentiles = statFunctions.filter(a => a.endsWith("%")).map { p =>
      try {
        p.stripSuffix("%").toDouble / 100.0
      } catch {
        case e: NumberFormatException =>
          throw new IllegalArgumentException(s"Unable to parse $p as a percentile", e)
      }
    }
    require(percentiles.forall(p => p >= 0 && p <= 1), "Percentiles must be in the range [0, 1]")
    val aggExprs = selectedCols.flatMap(c => {
      var percentileIndex = 0
      statFunctions.map { stats =>
        if (stats.endsWith("%")) {
          val index = percentileIndex
          percentileIndex += 1
          expr(s"cast(percentile_approx($c, array(${percentiles.mkString(", ")}))[$index] as string)")
        } else {
          expr(s"cast($stats($c) as string)")
        }
      }
    })

    val aggResult = ds.select(aggExprs: _*).head()

    val r = aggResult.toSeq.grouped(statFunctions.length).toArray
      .zip(selectedCols)
      .map{case(seq, column) => column +: seq }
      .map(Row.fromSeq)

    val output = StructField("columns", StringType) +: statFunctions.map(c => StructField(c, StringType))

    val spark = ds.sparkSession
    spark.createDataFrame(spark.sparkContext.parallelize(r), StructType(output))
  }
}

object RichDataFrame {

  trait Enrichment {
    implicit def enrichMetadata(ds: DataFrame): RichDataFrame =
      new RichDataFrame(ds)
  }

  object implicits extends Enrichment

}
