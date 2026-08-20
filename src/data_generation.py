import numpy as np
import pandas as pd

def generate_landcover(seed=7,n=1200):
    rng=np.random.default_rng(seed); x=rng.uniform(0,100,n); y=rng.uniform(0,100,n)
    cls=np.where((x<35)&(y<60),'Water',np.where((x>70)&(y>45),'Urban',np.where(y>70,'Forest','Cropland')))
    base={'Water':[.12,.10,.06,.03],'Urban':[.25,.28,.30,.22],'Forest':[.08,.14,.52,.31],'Cropland':[.12,.20,.43,.28]}
    arr=np.array([base[c] for c in cls])+rng.normal(0,.03,(n,4))
    df=pd.DataFrame({'x':x,'y':y,'blue':arr[:,0],'red':arr[:,1],'nir':arr[:,2],'swir':arr[:,3],'class':cls})
    df['ndvi']=(df.nir-df.red)/(df.nir+df.red); return df
