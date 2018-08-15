import numpy as np
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
import geopy
from geopy.geocoders import Yandex
import pickle
import os

geolocator = Yandex()

def do_geocode(address, trial=0):
    try:
        location = geolocator.geocode(address)
        if location == None:
            return "NaN", "NaN"
        return location.latitude, location.longitude
    except Exception as e:
        if trial > 3:
            return "NaN", "NaN"
        else:
            return do_geocode(address, trial + 1)
        time.sleep(2)
        
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class GeoDataTransformer(TransformerMixin):
    def __init__(self):
        self.cache = {}
        self.pickle_path = './data/locations.pkl'
        self.imp = Imputer(strategy='median')
        
        if os.path.isfile(self.pickle_path):
            self.cache = load_pickle(self.pickle_path)

    def fit(self, df, y=None):
        return self
                
    def transform(self, df):
        df_copy = df.copy()
        locations = []
	
        for sub_area in df['sub_area']:
            if sub_area in self.cache:
                latitude, longitude = self.cache[sub_area]
                if latitude == 'NA' or longitude == 'NA':
                    locations.append(['NaN', 'NaN'])
                else:
                    locations.append([latitude, longitude])
            else:
                latitude, longitude = do_geocode(sub_area)
                self.cache[sub_area] = [latitude, longitude]
                locations.append([latitude, longitude])
        
        locations = np.array(locations)
        locations = self.imp.fit_transform(locations)

        lat = locations[:, 0]
        lon = locations[:, 1]

        df_copy['x'] = np.cos(lat) * np.cos(lon)
        df_copy['y'] = np.cos(lat) * np.sin(lon)
        df_copy['z'] = np.sin(lat)
        
        df_copy = df_copy.drop(columns=['sub_area'])
        
        if not os.path.isfile(self.pickle_path):
            save_pickle(cache, self.pickle_path)
            
        return df_copy

class MyRegression(TransformerMixin):
    def fit(self, df, y=None):
        self.model = RidgeCV(alphas=(0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500))
        self.model.fit(df, y)
        return self
    
    def transform(self, df):
        df_copy = df.copy()
        values = self.model.predict(df)
        df_copy = np.column_stack([df_copy, values])
        return df_copy
    
    
class MyXGBoost(TransformerMixin):
    def fit(self, df, y=None):
        xgb_model = xgb.XGBRegressor()
        params = {
            'gamma': [0.5, 1, 2],
            'max_depth': [3, 4, 5]
        }
        grid_search = GridSearchCV(
            xgb_model,
            param_grid=params,
            n_jobs=4
        )
        self.model = grid_search
        self.model.fit(df, y)
        return self
    
    def transform(self, df):
        df_copy = df.copy()
        values = self.model.predict(df)
        df_copy = np.column_stack([df_copy, values])
        return df_copy

