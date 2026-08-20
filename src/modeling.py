from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_spatial_holdout(df):
    features=['blue','red','nir','swir','ndvi']
    train=df.x<70; test=~train
    model=RandomForestClassifier(n_estimators=250,random_state=42).fit(df.loc[train,features],df.loc[train,'class'])
    pred=model.predict(df.loc[test,features])
    return model, {'accuracy':accuracy_score(df.loc[test,'class'],pred),'macro_f1':f1_score(df.loc[test,'class'],pred,average='macro')}
