from pyspark.sql import SparkSession
import data
import model

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel

spark = SparkSession\
    .builder\
    .appName("Heart Failure Prediction")\
    .getOrCreate()

data = Data(spark)
model = Model(spark)

data.load("heart_failure.csv")
model.train(data.train, data.test)

model.bestModel.write().overwrite().save("./model.pkl")
model.pipeline.write().overwrite().save("./pipeline.pkl")


#making predictions
pipeline = PipelineModel.load("./pipeline.pkl")
model = RandomForestClassificationModel.load("./model.pkl")

