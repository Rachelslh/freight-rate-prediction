import xgboost as xgb
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from dataset import Dataset
from optimize import Optimizer


cfg = OmegaConf.load("config.yaml")

dataset = Dataset(cfg, cfg.training_data)
optimizer = Optimizer(cfg)
    
best_params = optimizer(dataset.x, dataset.y_log, dataset.x_val, dataset.y_val_log)

params = dict(best_params)
params.update({
    "n_estimators": cfg.n_estimators,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "early_stopping_rounds": cfg.early_stopping_rounds,
    "random_state": cfg.RANDOM_SEED,
})
model = xgb.XGBRegressor(**params)

model.fit(dataset.x, dataset.y_log, eval_set=[(dataset.x_val, dataset.y_val_log)], verbose=100)

importance_df = pd.DataFrame({
    'Feature': model.feature_names_in_,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance_df)

preds_log = model.predict(dataset.x_val)
preds = np.expm1(preds_log)

print(f"log-RMSE : {mean_squared_error(dataset.y_val_log, preds_log) ** 0.5:.5f}")
print(f"MAPE     : {mean_absolute_percentage_error(dataset.y_val, preds) * 100:.2f}%")
print(f"R2       : {r2_score(dataset.y_val, preds):.4f}")
print(f"RMSE ($) : {mean_squared_error(dataset.y_val, preds) ** 0.5:.2f}")

model.save_model(cfg.model_out_file)