from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

_builder = (
    SparkSession.builder
        .appName("axis_db")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.3.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(_builder).getOrCreate()