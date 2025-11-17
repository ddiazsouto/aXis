from pyspark.ml.feature import Word2Vec, Tokenizer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType

from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np



df = spark.createDataFrame([
    ("Data breaches must be reported within 72 hours.",),
    ("The data subject has the right to access their data.",),
    ("GDPR applies in the UK.",),
    ("Notify authorities of a security incident immediately.",),
    ("Right to be forgotten under Article 17.",),
], ["text"])


tokenizer = Tokenizer(inputCol="text", outputCol="words")


word2vec = Word2Vec(
    vectorSize=100,
    minCount=1,
    inputCol="words",
    outputCol="word_vecs"
)


pipeline = Pipeline(stages=[tokenizer, word2vec])
model = pipeline.fit(df)


@pandas_udf(ArrayType(FloatType()))
def avg_word_vectors(word_vecs: pd.Series) -> pd.Series:
    arrays = np.array(word_vecs.tolist())
    return pd.Series([arr.mean(axis=0).tolist() for arr in arrays])


df_vec = model.transform(df).withColumn("sentence_vec", avg_word_vectors("word_vecs"))
df_vec = df_vec.select("text", "sentence_vec")

df_vec.show(truncate=False)