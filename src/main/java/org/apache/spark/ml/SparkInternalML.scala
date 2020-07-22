package org.apache.spark.ml

import org.apache.spark.ml.tuning.ValidatorParams
import org.apache.spark.ml.util._

/**
  * Below objects are not accessible outside of the org.apache.spark.ml package.
  * Therefore, we use an encapsulation violation pattern.
  */
object SparkInternalML {
  def getDefaultParamsReader: DefaultParamsReader.type = DefaultParamsReader
  def getDefaultParamsWriter: DefaultParamsWriter.type = DefaultParamsWriter
  def getMetaAlgorithmReadWrite: MetaAlgorithmReadWrite.type = MetaAlgorithmReadWrite
  def getInstrumentation: Instrumentation.type = Instrumentation
  def getValidatorParams: ValidatorParams.type = ValidatorParams
  def getSchemaUtils: SchemaUtils.type = SchemaUtils
}
