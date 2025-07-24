from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import json
from datetime import datetime
from typing import List
import uvicorn


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    rf_model = joblib.load('models/random_forest_model.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    lr_model = joblib.load('models/linear_regression_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    models = {
        'random_forest': rf_model,
        'xgboost': xgb_model,
        'linear_regression': lr_model
    }
    
    logger.info("Tous les modèles chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des modèles: {e}")

class PredictionRequest(BaseModel):
    features: List[float]
    model_name: str = "random_forest"

class PredictionResponse(BaseModel):
    prediction: float
    model_used: str
    timestamp: str

@app.get("/")
async def root():
    return {"message": "Prediction du prix d'une maison"}

@app.get("/models")
async def get_available_models():
    return {"models": list(models.keys())}

@app.get("/features")
async def get_features():
    return {"features": feature_names, "count": len(feature_names)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):

    start_time = datetime.now()
    
    try:
        if request.model_name not in models:
            raise HTTPException(
                status_code=400, 
                detail=f"Modèle non disponible. Modèles disponibles: {list(models.keys())}"
            )
    
        if len(request.features) != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Nombre de features incorrect. Attendu: {len(feature_names)}, Reçu: {len(request.features)}"
            )
        
        features_array = np.array(request.features).reshape(1, -1)

        model = models[request.model_name]
        
        if request.model_name == 'linear_regression':
            features_scaled = scaler.transform(features_array)
            prediction = model.predict(features_scaled)[0]
        else:
            prediction = model.predict(features_array)[0]

        duration = (datetime.now() - start_time).total_seconds()

        log_data = {
            'timestamp': start_time.isoformat(),
            'model': request.model_name,
            'features': request.features,
            'prediction': float(prediction),
            'duration': duration
        }
        logger.info(f"Prediction: {json.dumps(log_data)}")
        
        return PredictionResponse(
            prediction=float(prediction),
            model_used=request.model_name,
            timestamp=start_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/loggers")
async def loggers():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)