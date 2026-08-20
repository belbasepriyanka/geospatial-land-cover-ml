from src.data_generation import generate_landcover
from src.modeling import train_spatial_holdout
def test_model():
    df=generate_landcover(n=300); _,m=train_spatial_holdout(df); assert 0 <= m['accuracy'] <= 1
