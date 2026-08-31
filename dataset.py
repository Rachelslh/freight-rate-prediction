import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances


class Dataset:
    def __init__(self, config):
        self.dataset_path = config.path
        self.val_rate = config.VAL_FRACTION
        self.cfg = config

        self.build_features()
        self.split()
        
        self.x = self.train_df.drop(columns=["posted_rate"])
        self.y = self.train_df["posted_rate"]
        self.y_log = np.log1p(self.y)
        
        self.x_val = self.val_df.drop(columns=["posted_rate"])
        self.y_val = self.val_df["posted_rate"]
        self.y_val_log = np.log1p(self.y_val)
        
    def build_features(self) -> pd.DataFrame:
        
        df = pd.read_csv(self.dataset_path)
        
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
        df["weight"] = df["weight"].fillna(df["weight"].median())
        df["market_index"] = df["market_index"].fillna(df["market_index"].median())

        df["route"] = df["pickup"] + "__" + df["delivery"]
        df["route_freq"] = df["route"].value_counts().fillna(0)
        df["pickup_freq"] = df["pickup"].value_counts().fillna(0)
        df["delivery_freq"] = df["delivery"].value_counts().fillna(0)

        self.df = pd.get_dummies(df, columns=["equipment"], prefix="eq")


    def split(self) -> pd.DataFrame:
        df = self.df.sort_values("date").reset_index(drop=True)
        split_idx = int(len(df) * (1 - self.val_rate))
        cutoff_date = df.iloc[split_idx]["date"]
        train_df = df[df["date"] < cutoff_date]
        val_df = df[df["date"] >= cutoff_date]
            
        self.train_df = train_df.drop(columns=["load_id", "pickup", "delivery", "route", "date"])
        self.val_df = val_df.drop(columns=["load_id", "pickup", "delivery", "route", "date"])