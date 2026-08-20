from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_generation import generate_landcover
from src.modeling import train_spatial_holdout
df=generate_landcover(); _,m=train_spatial_holdout(df); print(m)
