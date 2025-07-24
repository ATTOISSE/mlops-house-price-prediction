import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = '..data/data.csv'
def preprocess_data(input_file='data/data.csv', output_file='data/clean_data.csv'):

    try:
        logger.info("Chargement des données...")
        df = pd.read_csv(input_file)
        logger.info(f"Données chargées: {df.shape}")
        
        logger.info("Nettoyage des valeurs manquantes...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        logger.info("Encodage des variables catégorielles...")
        le = LabelEncoder()
        for col in categorical_cols:
            if col != 'price':  
                df[col] = le.fit_transform(df[col].astype(str))

        logger.info("Suppression des outliers...")
        for col in numeric_cols:
            if col != 'price':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        df.to_csv(output_file, index=False)
        logger.info(f"Données prétraitées sauvegardées: {output_file}")
        logger.info(f"Shape finale: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement: {e}")
        raise

if __name__ == "__main__":
    
    preprocess_data()