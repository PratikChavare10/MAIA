import pickle
import pandas as pd
from typing import Dict, Union
from config import (
    YIELD_VR_PATH,
    LE_CROP_PATH,
    LE_SOIL_PATH,
    LE_WEATHER_PATH,
    LE_TRANS_PATH
)


def load_artifacts():
    print("load")
    trans = pickle.load(open(LE_TRANS_PATH, "rb"))
    """Loads all saved encoders, transformers, and the trained VotingRegressor model."""
    trans = pickle.load(open(LE_TRANS_PATH, "rb"))
    le_crop = pickle.load(open(LE_CROP_PATH, "rb"))
    le_soil = pickle.load(open(LE_SOIL_PATH, "rb"))
    le_weather = pickle.load(open(LE_WEATHER_PATH, "rb"))
    model = pickle.load(open(YIELD_VR_PATH, "rb"))

    return trans, le_crop, le_soil, le_weather, model


def predict_yield(
        crop: str,
        soil_type: str,
        rainfall: float,
        temperature: float,
        area: float = 1.0,
        fertilizer: bool = True,
        region: str = "North",  # Fallback default
        weather_condition: str = "Sunny",  # Fallback default
        irrigation_used: bool = True,  # Fallback default
        days_to_harvest: int = 120  # Fallback default
) -> Dict[str, Union[float, str]]:
    """
    Accepts input parameters, builds a single-row DataFrame,
    applies exact training transformations, and runs prediction.
    """
    # 1. Load trained artifacts
    trans, le_crop, le_soil, le_weather, model = load_artifacts()


    # 2. Convert incoming values into a pandas DataFrame matching exact training feature names
    raw_df = pd.DataFrame([{
        'Region': region,
        'Rainfall_mm': rainfall,
        'Temperature_Celsius': temperature,
        'Fertilizer_Used': fertilizer,
        'Irrigation_Used': irrigation_used,
        'Days_to_Harvest': days_to_harvest,
        'Crop': crop,
        'Soil_Type': soil_type,
        'Weather_Condition': weather_condition
    }])
   

    # 3. Label encode categorical string columns
    raw_df['crop_encoded'] = le_crop.transform(raw_df['Crop'])
    raw_df['soil_encoded'] = le_soil.transform(raw_df['Soil_Type'])
    raw_df['weather_encoded'] = le_weather.transform(raw_df['Weather_Condition'])
    print("1")
    # 4. Drop original categorical string columns to match X before ColumnTransformer
    X = raw_df.drop(columns=['Crop', 'Soil_Type', 'Weather_Condition'])
    


    # 5. Transform using ColumnTransformer
    X_trans = trans.transform(X)

    # 6. Predict yield per hectare (Voting Regressor)
    yield_per_hectare = float(model.predict(X_trans)[0])
    


    # 7. Calculate total yield for the total area
    total_yield = yield_per_hectare * area
    

    # Structured result without LSTM fields
    return {
        "final_yield": round(total_yield, 2),
        "yield_per_hectare": round(yield_per_hectare, 2),
        "unit": "tons"
    }
# ── Example Usage ─────────────────────────────────────────
if __name__ == "__main__":
    # Sample raw input dataframe representing new/unseen data
    sample_data = pd.DataFrame([{
        'Region': 'North',
        'Rainfall_mm': 1200.5,
        'Temperature_Celsius': 26.4,
        'Fertilizer_Used': True,      # or 'Yes'/category depending on raw data
        'Irrigation_Used': True,
        'Days_to_Harvest': 110,
        'Crop': 'Wheat',              # must exist in le_crop classes
        'Soil_Type': 'Loam',          # must exist in le_soil classes
        'Weather_Condition': 'Sunny'  # must exist in le_weather classes
    }])

    # Get prediction
    predicted_yield = predict_yield('Wheat','Loam',1200.5,26.4,1,True,'North','Sunny',True,110)
    print(predicted_yield)
    print(f"Predicted Crop Yield: {predicted_yield}")