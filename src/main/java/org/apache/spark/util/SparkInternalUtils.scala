package org.apache.spark.util

import org.apache.spark.{SparkContext, SparkEnv}

/**
  * Below objects are not accessible outside of the org.apache.spark.util package.
  * Therefore, we use an encapsulation violation pattern.
  */
object SparkInternalUtils {

  def sparkEnv(sc: SparkContext): SparkEnv = sc.env
  def getThreadUtils: ThreadUtils.type = ThreadUtils

}
