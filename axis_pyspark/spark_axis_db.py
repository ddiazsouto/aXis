from pyspark.ml.feature import Word2Vec, Tokenizer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType

from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np

from functions.avg_word_vector import avg_word_vector
from functions.get_spark import spark


df = spark.createDataFrame([
    ("Data breaches must be reported within 72 hours.",),
    ("The data subject has the right to access their data.",),
    ("GDPR applies in the UK.",),
    ("Notify authorities of a security incident immediately.",),
    ("Right to be forgotten under Article 17.",),
], ["text"])


class aXisDB:
    def __init__(self, path: str = "axis.db"):
        self.path = path
        self.embedder = Embedder('all-MiniLM-L6-v2')
        self._vector_registry = None
        self.load()

    def search(self):

        query_vec = self.embedder.model.transform(spark.createDataFrame([("How long to report a breach?",)], ["text"])) \
                        .collect()[0]["sentence_vec"]
        query_bc = spark.sparkContext.broadcast(np.array(query_vec))




df_final = df_vec \
    .withColumn("sentence_vec", avg_word_vectors_pandas("word_vecs")) \
    .withColumn("similarity", cosine_similarity_pandas("sentence_vec")) \
    .orderBy("similarity", ascending=False)

df_final.select("text", "similarity").show(5, truncate=False)




# df_vec = 
# df_vec = df_vec.select("text", "sentence_vec")

# df_vec.show(truncate=False)


class Embedder:

    available_models = [
        "Vord2Vec",
    ]

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if model_name not in self.list_available_models():
            raise ValueError(f"Model {model_name} is not available from: {self.list_available_models}.")
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model:

            pipeline = Pipeline(
                stages=[
                    Tokenizer(inputCol="text", outputCol="words"),
                    Word2Vec(vectorSize=100, minCount=1, inputCol="words", outputCol="word_vecs")
                ]
            )
            self._model = pipeline.fit(df)
        else:
            return self._model

    def encode(self, sentence):        
        return self.model.transform(
            sentence
        ).withColumn(
            "sentence_vec",
            avg_word_vector("word_vecs")
        )