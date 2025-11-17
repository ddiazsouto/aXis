import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType
    

@pandas_udf(FloatType())
def cosine_similarity_pandas(vecs: pd.Series) -> pd.Series:
    vecs_array = np.array(vecs.tolist())
    query = query_bc.value
    
    # Vectorized dot product
    dot = np.sum(vecs_array * query, axis=1)
    norm_vecs = np.linalg.norm(vecs_array, axis=1)
    norm_query = np.linalg.norm(query)
    
    return pd.Series(dot / (norm_vecs * norm_query + 1e-8))