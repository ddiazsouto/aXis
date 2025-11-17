import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType, ArrayType


@pandas_udf(ArrayType(FloatType()))
def avg_word_vectors(word_vecs: pd.Series) -> pd.Series:
    arrays = np.array(word_vecs.tolist())
    return pd.Series([arr.mean(axis=0).tolist() for arr in arrays])
