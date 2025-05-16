import numpy as np
import pandas as pd

class SpatialGridder:
    def __init__(self, lat_key='lat', lon_key='lon', time_key='time', feature_cols=None,
                 lat_vals=None, lon_vals=None):
        self.lat_key = lat_key
        self.lon_key = lon_key
        self.time_key = time_key
        self.feature_cols = feature_cols  # features to keep
        self.lat_vals = lat_vals
        self.lon_vals = lon_vals

        # Will be filled after fit
        self.grid_lat = None
        self.grid_lon = None
        self.timestamps = None

    def fit(self, df):
        self.grid_lat = np.sort(df[self.lat_key].unique()) if self.lat_vals is None else np.array(self.lat_vals)
        self.grid_lon = np.sort(df[self.lon_key].unique()) if self.lon_vals is None else np.array(self.lon_vals)
        self.timestamps = pd.to_datetime(np.sort(df[self.time_key].unique()))

        if self.feature_cols is None:
            exclude = {self.lat_key, self.lon_key, self.time_key}
            self.feature_cols = [col for col in df.columns if col not in exclude]

    def transform(self, df):
        if self.grid_lat is None or self.grid_lon is None or self.timestamps is None:
            raise ValueError("Call fit() before transform().")

        lat_to_idx = {lat: i for i, lat in enumerate(self.grid_lat)}
        lon_to_idx = {lon: i for i, lon in enumerate(self.grid_lon)}
        time_to_idx = {time: i for i, time in enumerate(self.timestamps)}

        T = len(self.timestamps)
        H = len(self.grid_lat)
        W = len(self.grid_lon)
        C = len(self.feature_cols)

        data = np.full((T, H, W, C), np.nan, dtype=np.float32)

        for row in df.itertuples(index=False):
            t_val = pd.to_datetime(getattr(row, self.time_key))
            if t_val not in time_to_idx:
                continue  # skip unseen timestamps
            t = time_to_idx[t_val]
            h = lat_to_idx[getattr(row, self.lat_key)]
            w = lon_to_idx[getattr(row, self.lon_key)]
            features = [getattr(row, f) for f in self.feature_cols]
            data[t, h, w] = features

        return data

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, data):
        if self.grid_lat is None or self.grid_lon is None or self.timestamps is None:
            raise ValueError("fit must be called before inverse_transform")

        T, H, W, C = data.shape
        rows = []
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    feature_values = data[t, h, w]
                    if not np.any(np.isnan(feature_values)):
                        row = {
                            self.time_key: self.timestamps[t],
                            self.lat_key: self.grid_lat[h],
                            self.lon_key: self.grid_lon[w],
                        }
                        row.update({self.feature_cols[i]: feature_values[i] for i in range(C)})
                        rows.append(row)
        ret_df =  pd.DataFrame(rows)
        ret_df[self.time_key] = pd.to_datetime(ret_df[self.time_key])
        ret_df.sort_values(by=[self.time_key, self.lat_key,self.lon_key], inplace=True)
        ret_df.reset_index(drop=True, inplace=True)
        return ret_df