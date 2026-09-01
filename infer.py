import xgboost as xgb
import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from dataset import Dataset


cfg = OmegaConf.load("config.yaml")

dataset = Dataset(cfg, inference=True)

model = xgb.XGBRegressor()
model.load_model("xgb_final.json")


preds_log = model.predict(dataset.x)
preds = np.expm1(preds_log)

df = pd.read_csv(cfg.validation_template)
df['predicted_rate'] = preds

df.to_csv(cfg.validation_out, index=False)
        