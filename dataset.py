import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances


class Dataset:
    def __init__(self, config, inference : bool=False):
        self.inference=inference
        self.val_rate = config.VAL_FRACTION
        self.cfg = config
        
        self.frequency_features = None
        self.weight_median = None
        self.market_index_median = None
        
        if not self.inference:    
            train_df, val_df = self.load_and_split()
            train_df = self.build_features(train_df)
            val_df = self.build_features(val_df)
            
            self.x = train_df.drop(columns=["posted_rate"])
            self.y = train_df["posted_rate"]
            self.y_log = np.log1p(self.y)
                    
            self.x_val = val_df.drop(columns=["posted_rate"])
            self.y_val = val_df["posted_rate"]
            self.y_val_log = np.log1p(self.y_val)
            
            self.save_artifacts()
            
        else:
            with open(self.cfg.model_artifacts, "rb") as f:
                model_features = pickle.load(f)
                
            self.frequency_features = model_features["freq_maps"]
            self.weight_median = model_features["impute_values"]["weight_median"]
            self.market_index_median = model_features["impute_values"]["market_index_median"]

            df, _ = self.load_and_split()
            df = self.build_features(df)
            self.x = df
        
        
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
        
        if self.weight_median is None:
            self.weight_median = df["weight"].median()
        df["weight"] = df["weight"].fillna(self.weight_median)
        
        if self.market_index_median is None:
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

    def load_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.cfg.training_data if not self.inference else self.cfg.validation_data)
        
        df = df.sort_values("date").reset_index(drop=True)

        if self.inference:
            return df, None
        
        split_idx = int(len(df) * (1 - self.val_rate))
        cutoff_date = df.iloc[split_idx]["date"]
        train_df = df[df["date"] < cutoff_date]
        val_df = df[df["date"] >= cutoff_date]
        
        return train_df, val_df
    
    
    def save_artifacts(self, ):
        data_to_save = {
            "freq_maps": self.frequency_features,
            "impute_values": {
                "weight_median": self.weight_median,
                "market_index_median": self.market_index_median,
            }
        }

        with open(self.cfg.model_artifacts, "wb") as f:
            pickle.dump(data_to_save, f)