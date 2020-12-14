from pyspark.sql import Window, SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import DoubleType, IntegerType, BooleanType

class Data:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load(self, fileUrl: str):
        df = self.spark.read\
            .option("header", True)\
            .csv(fileUrl)
        
        df = df\
            .withColumn("age", f.col("age").cast(DoubleType()))\
            .withColumn("creatinine_phosphokinase", f.col("creatinine_phosphokinase").cast(DoubleType()))\
            .withColumn("ejection_fraction", f.col("ejection_fraction").cast(DoubleType()))\
            .withColumn("high_blood_pressure", f.col("high_blood_pressure").cast(DoubleType()))\
            .withColumn("platelets", f.col("platelets").cast(DoubleType()))\
            .withColumn("serum_creatinine", f.col("serum_creatinine").cast(DoubleType()))\
            .withColumn("serum_sodium", f.col("serum_sodium").cast(DoubleType()))\
            .withColumn("DEATH_EVENT", f.col("DEATH_EVENT").cast(DoubleType()))\
            .withColumn("anaemia", f.col("anaemia").cast(IntegerType()))\
            .withColumn("diabetes", f.col("diabetes").cast(IntegerType()))\
            .withColumn("high_blood_pressure", f.col("high_blood_pressure").cast(IntegerType()))\
            .withColumn("sex", f.col("sex").cast(IntegerType()))\
            .withColumn("smoking", f.col("smoking").cast(IntegerType()))\
            .withColumnRenamed("DEATH_EVENT", "label")\
            .drop("time")

        window = Window.partitionBy("label").orderBy("serum_creatinine")
        udf = f.udf(lambda x: x % 5 == 0, BooleanType())

        df = df\
            .withColumn("_test_set", f.row_number().over(window))\
            .withColumn("_test_set", udf(f.col("_test_set")))
        
        test = df.where(df["_test_set"] == True)
        self.test = test.drop("_test_set")

        train = df.where(df["_test_set"] == False)
        self.train = train.drop("_test_set")

        return self
        