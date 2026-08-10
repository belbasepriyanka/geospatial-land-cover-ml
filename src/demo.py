from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
from spatial_split import spatial_block_ids

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)
rng = np.random.default_rng(23)
n = 1200
x = rng.uniform(0,100,n)
y = rng.uniform(0,100,n)
classes = np.where(y > 68, "forest", np.where(x < 32, "water", "urban"))
means = {
 "forest": [0.07,0.10,0.05,0.48,0.26],
 "water": [0.08,0.06,0.04,0.03,0.02],
 "urban": [0.16,0.19,0.22,0.24,0.28],
}
bands = np.vstack([rng.normal(means[c], [0.02]*5) for c in classes])
df = pd.DataFrame(bands, columns=["blue","green","red","nir","swir1"])
df["x"], df["y"], df["label"] = x,y,classes
df["NDVI"] = (df["nir"]-df["red"])/(df["nir"]+df["red"]+1e-9)
df["NDWI"] = (df["green"]-df["nir"])/(df["green"]+df["nir"]+1e-9)
df.to_csv(ROOT/"data"/"synthetic_landcover_samples.csv", index=False)
features = ["blue","green","red","nir","swir1","NDVI","NDWI"]
groups = spatial_block_ids(x,y,20)
split = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
train_idx, test_idx = next(split.split(df[features], df["label"], groups))
model = RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced")
model.fit(df.loc[train_idx,features], df.loc[train_idx,"label"])
pred = model.predict(df.loc[test_idx,features])
acc = accuracy_score(df.loc[test_idx,"label"], pred)
imp = pd.Series(model.feature_importances_, index=features).sort_values()
plt.figure(figsize=(6.5,4.5)); plt.barh(imp.index, imp.values)
plt.xlabel("Random Forest importance"); plt.title("Land-cover Feature Importance"); plt.tight_layout()
plt.savefig(ROOT/"outputs"/"feature_importance.png", dpi=180); plt.close()
ConfusionMatrixDisplay.from_predictions(df.loc[test_idx,"label"], pred)
plt.title("Spatial Holdout Confusion Matrix"); plt.tight_layout()
plt.savefig(ROOT/"outputs"/"confusion_matrix.png", dpi=180); plt.close()
(ROOT/"outputs"/"metrics.json").write_text(json.dumps({"spatial_holdout_accuracy": float(acc)}, indent=2))
print({"spatial_holdout_accuracy": acc})
