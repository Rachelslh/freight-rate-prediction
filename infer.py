import tempfile

import xgboost as xgb
import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from dataset import Dataset


def build_december_predictions(cfg, model, validation_df):
    dec_df = pd.read_csv(cfg.test_input).drop(columns=['predicted_rate'])
    dec_df['date'] = pd.to_datetime(dec_df['date'])
    dec_df['load_id'] = range(len(dec_df))

    daily_market = validation_df.groupby('date')[['market_index', 'quote_signal']].mean()
    dec_df = dec_df.merge(daily_market, on='date', how='left')

    coords = pd.concat([
        validation_df[['pickup', 'pickup_lat', 'pickup_lon']].set_axis(['city', 'lat', 'lon'], axis=1),
        validation_df[['delivery', 'delivery_lat', 'delivery_lon']].set_axis(['city', 'lat', 'lon'], axis=1),
    ]).drop_duplicates('city').set_index('city')
    dec_df['pickup_lat'] = dec_df['pickup'].map(coords['lat'])
    dec_df['pickup_lon'] = dec_df['pickup'].map(coords['lon'])
    dec_df['delivery_lat'] = dec_df['delivery'].map(coords['lat'])
    dec_df['delivery_lon'] = dec_df['delivery'].map(coords['lon'])
    
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        dec_df.to_csv(tmp.name, index=False)
        december_dataset = Dataset(cfg, tmp.name, inference=True)

    x = december_dataset.x.reindex(columns=model.get_booster().feature_names, fill_value=0)
    december_dataset.orig_df['predicted_rate'] = np.expm1(model.predict(x))

    output = pd.read_csv(cfg.test_input).drop(columns=['predicted_rate'])
    output['predicted_rate'] = december_dataset.orig_df['predicted_rate'].values
    output.to_csv(cfg.test_input, index=False)

    

if __name__ == "__main__":
        
    cfg = OmegaConf.load("config.yaml")

    dataset = Dataset(cfg, cfg.validation_data, inference=True)

    model = xgb.XGBRegressor()
    model.load_model("xgb_final.json")

    preds_log = model.predict(dataset.x)
    preds = np.expm1(preds_log)

    dataset.orig_df['predicted_rate'] = preds
    output = dataset.orig_df[['load_id', 'predicted_rate']]

    output.to_csv(cfg.validation_out, index=False)

    build_december_predictions(cfg, model, dataset.orig_df)
            
