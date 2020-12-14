from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class Model:
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def train(self, train: DataFrame, test: DataFrame):
        self.pipeline = Pipeline(stages = [
            VectorAssembler(inputCols = train.columns[:-1], 
                            outputCol = "vector"),
            VectorIndexer(inputCol = "vector",
                          outputCol = "features",
                          maxCategories = 4)
        ]).fit(train)

        rf = RandomForestClassifier()
        paramGrid = ParamGridBuilder()\
            .addGrid(rf.maxDepth, [2, 4, 6])\
            .addGrid(rf.numTrees, [10, 20, 30])\
            .build()
        crossval = CrossValidator(estimator = rf,
                                  estimatorParamMaps = paramGrid,
                                  evaluator = BinaryClassificationEvaluator(metricName = "areaUnderPR"),
                                  numFolds = 3)
        self.bestModel = crossval.fit(self.pipeline.transform(train)).bestModel
        self.areaUnderPR = BinaryClassificationEvaluator(metricName = "areaUnderPR")\
            .evaluate(self.bestModel.transform(self.pipeline.transform(test)))
        
        return self
