import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances


class Dataset:
    def __init__(self, config):
        self.dataset_path = config.path
        self.val_rate = config.VAL_FRACTION
        self.cfg = config

        self.split()
        self.frequency_features = None
        self.train_df = self.build_features(self.train_df)
        self.val_df = self.build_features(self.val_df)
                
        self.x = self.train_df.drop(columns=["posted_rate"])
        self.y = self.train_df["posted_rate"]
        self.y_log = np.log1p(self.y)
        
        self.x_val = self.val_df.drop(columns=["posted_rate"])
        self.y_val = self.val_df["posted_rate"]
        self.y_val_log = np.log1p(self.y_val)
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df["date"] = pd.to_datetime(df["date"])

        df['distance_x_quote'] = df['distance'] * df['quote_signal']
        df['distance_x_marked_index'] = df['quote_signal'] * df['market_index']
        df['distance_x_weight'] = df['distance'] * df['weight']
        
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month

        starts = np.radians(df[['pickup_lat', 'pickup_lon']].to_numpy())
        ends = np.radians(df[['delivery_lat', 'delivery_lon']].to_numpy())
        matrix_rad = haversine_distances(starts, ends)
        row_distances_rad = np.diagonal(matrix_rad)
        df["haversine_dist"] = row_distances_rad * 6371.0088

        df["weight_missing"] = df["weight"].isna().astype(int)
        df["market_index_missing"] = df["market_index"].isna().astype(int)
        
        self.weight_median = df["weight"].median()
        df["weight"] = df["weight"].fillna(self.weight_median)
        
        self.market_index_median = df["market_index"].median()
        df["market_index"] = df["market_index"].fillna(self.market_index_median)

        df["route"] = df["pickup"] + "__" + df["delivery"]
        if self.frequency_features is None:
            self.frequency_features = {
            "route": df["route"].value_counts(),
            "pickup": df["pickup"].value_counts(),
            "delivery": df["delivery"].value_counts(),
        }
            
        df["route_freq"] = df["route"].map(self.frequency_features["route"]).fillna(0)
        df["pickup_freq"] = df["pickup"].map(self.frequency_features["pickup"]).fillna(0)
        df["delivery_freq"] = df["delivery"].map(self.frequency_features["delivery"]).fillna(0)

        df = pd.get_dummies(df, columns=["equipment"], prefix="eq")

        new_df = df.drop(columns=["load_id", "pickup", "delivery", "route", "date"])
        return new_df

    def split(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)
        
        df = df.sort_values("date").reset_index(drop=True)
        split_idx = int(len(df) * (1 - self.val_rate))
        cutoff_date = df.iloc[split_idx]["date"]
        self.train_df = df[df["date"] < cutoff_date]
        self.val_df = df[df["date"] >= cutoff_date]
            
    def save_artifacts(self, ):
        data_to_save = {
            "freq_maps": self.frequency_features,
            "impute_values": {
                "weight_median": self.weight_median,
                "market_index_median": self.market_index_median,
            }
        }

        with open("model_artifacts.pkl", "wb") as f:
            pickle.dump(data_to_save, f)