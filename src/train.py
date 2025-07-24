import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import logging
import os 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)
path = os.path.join(base_dir, '..', 'data', 'clean_data.csv')

def train_and_save_models(data_file=path):

    try:
        logger.info("Chargement des données prétraitées...")
        df = pd.read_csv(data_file)
        
        if 'price' not in df.columns:
            target_col = df.columns[-1]
        else:
            target_col = 'price'
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}

        logger.info("Entraînement Random Forest...")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='r2', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        rf_pred = rf_grid.predict(X_test)
        rf_score = r2_score(y_test, rf_pred)
        rf_mse = mean_squared_error(y_test, rf_pred)
        
        results['random_forest'] = {
            'r2_score': rf_score,
            'mse': rf_mse,
            'best_params': rf_grid.best_params_
        }
        
        
        joblib.dump(rf_grid.best_estimator_, '../models/random_forest_model.pkl')
        logger.info(f"Random Forest - R²: {rf_score:.4f}, MSE: {rf_mse:.2f}")

        logger.info("Entraînement XGBoost...")
        xgb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='r2', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        
        xgb_pred = xgb_grid.predict(X_test)
        xgb_score = r2_score(y_test, xgb_pred)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        
        results['xgboost'] = {
            'r2_score': xgb_score,
            'mse': xgb_mse,
            'best_params': xgb_grid.best_params_
        }

        joblib.dump(xgb_grid.best_estimator_, '../models/xgboost_model.pkl')
        logger.info(f"XGBoost - R²: {xgb_score:.4f}, MSE: {xgb_mse:.2f}")
        
        logger.info("Entraînement Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        
        lr_pred = lr.predict(X_test_scaled)
        lr_score = r2_score(y_test, lr_pred)
        lr_mse = mean_squared_error(y_test, lr_pred)
        
        results['linear_regression'] = {
            'r2_score': lr_score,
            'mse': lr_mse,
            'best_params': {}
        }
        
        joblib.dump(lr, '../models/linear_regression_model.pkl')
        joblib.dump(scaler, '../models/scaler.pkl')
        logger.info(f"Linear Regression - R²: {lr_score:.4f}, MSE: {lr_mse:.2f}")
     
        feature_names = X.columns.tolist()
        joblib.dump(feature_names, '../models/feature_names.pkl')
        
        logger.info("Tous les modèles ont été entraînés et sauvegardés!")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        raise

if __name__ == "__main__":
    import os
    os.makedirs('../models', exist_ok=True)
    
    train_and_save_models()