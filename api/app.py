from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,model_validator,computed_field,field_validator
from typing import List,Dict,Annotated ,Optional,Literal
import pandas as pd
import numpy as np
import pickle
import joblib


class Agridata(BaseModel):
    # N:Annotated[int,Field(...,ge=0,title='Nitrogen',description='Nitrogen contents in the Soil',example='38')]
    # P:Annotated[int,Field(...,ge=0,title='Phosphorous',description='Phosphorous contents in the Soil',example='64')]
    # K:Annotated[int,Field(...,ge=0,title='Potassium',description='Potassium contents in the Soil',example='98')]
    # temperature:Annotated[float,Field(...,ge=8.82,title='Tempreature',description='Tempreature in °C',example='54.1')]
    # humidity:Annotated[float,Field(...,ge=8.82,title='Humidity',description='Enter the Humidity ',example='98.2')]
    # ph:Annotated[float,Field(...,ge=1.2,title='PH',description='pH value of the soil less than 10',example='3.5')]
    # rainfall:Annotated[float,Field(...,ge=20.2,title='RainFall',description='Rainfall in mm',example='84.2mm')]
    N: Annotated[int, Field(..., ge=0, title="Nitrogen", description="Nitrogen contents in the Soil", json_schema_extra={"example": 38})]
    P: Annotated[int, Field(..., ge=0, title="Phosphorous", description="Phosphorous contents in the Soil", json_schema_extra={"example": 64})]
    K: Annotated[int, Field(..., ge=0, title="Potassium", description="Potassium contents in the Soil", json_schema_extra={"example": 98})]
    temperature: Annotated[float, Field(..., ge=8.82, title="Temperature", description="Temperature in °C", json_schema_extra={"example": 54.1})]
    humidity: Annotated[float, Field(..., ge=8.82, title="Humidity", description="Enter the Humidity", json_schema_extra={"example": 98.2})]
    ph: Annotated[float, Field(..., ge=1.2, title="PH", description="pH value of the soil less than 10", json_schema_extra={"example": 3.5})]
    rainfall: Annotated[float, Field(..., ge=20.2, title="RainFall", description="Rainfall in mm", json_schema_extra={"example": 84.2})]
    
    @computed_field
    @property
    def NPK_mean(self)->float:
        return (self.N + self.P + self.K) / 3
    
    @computed_field
    @property
    def THI(self)->float:
        return (self.temperature * self.humidity) / 100
    
    @computed_field
    @property
    def ph_category(self)->str:
        if self.ph < 5.5: return 'Acidic'
        elif self.ph <= 7.5: return 'Neutral'
        else: return 'Alkaline'
        
        
    @computed_field
    @property
    def rainfall_level(self) -> str:
        if self.rainfall <= 50:
            return 'Low'
        elif self.rainfall <= 100:
            return 'Medium'
        elif self.rainfall <= 200:
            return 'High'
        else:
            return 'Very High'
        
        
    @field_validator("N")
    @classmethod
    def check_N(cls, value):
        if value < 0:
            raise ValueError("Nitrogen cannot be negative")
        return value
        
    @field_validator("P")
    @classmethod
    def check_P(cls, value):
        if value < 0:
            raise ValueError("Phosphorous cannot be negative")
        return value
    
        
    @field_validator("K")
    @classmethod
    def check_K(cls, value):
        if value < 0:
            raise ValueError("Potassium cannot be negative")
        return value 
    
    @field_validator("temperature")
    @classmethod
    def check_temp(cls, value):
        if value < 0:
            raise ValueError("Temperature cannot be negative")
        return value 
    
    
    @field_validator("humidity")
    @classmethod
    def check_humidity(cls, value):
        if value < 0:
            raise ValueError("Humidity cannot be negative")
        return value 
    
    @field_validator("ph")
    @classmethod
    def check_ph(cls, value):
        if value < 0:
            raise ValueError("PH Values cannot be negative")
        return value 
    
    @field_validator("rainfall")
    @classmethod
    def check_rainfall(cls, value):
        if value < 0:
            raise ValueError("Rainfall cannot be negative")
        return value 
    
    
    

        
    
    


    


app = FastAPI()

@app.get('/')
def hello():
    return{
        'status':'✅running...'
    }
    

@app.get('/about')
def about():
    return{
        'message':'hello i am Sahil Kumar...😊(AI/ML Engineer).it is 3:05AM now and i have been figuring out to make these API working these is jsut the Crop Recomendation part of the Crop Advisory...'
        
    }
    
@app.post('/predict')
def prediction(data : Agridata):
    try:
        
        df = pd.DataFrame([{
            'N':data.N,
            'P':data.P,
            'K':data.K,
            'temperature': data.temperature,
            'humidity':data.humidity,
            'ph':data.ph,
            'rainfall':data.ph,
            'NPK_mean':data.NPK_mean,
            'THI':data.THI,
            'ph_category':data.ph_category,
            'rainfall_level':data.rainfall_level
            
        }])
            
        #target Encoder  
        te=joblib.load(r'target_encoder_s.pkl')
        
        #label Encoders   
        cat_cols = ['ph_category', 'rainfall_level']
        with open(r"feature_encoders.pkl","rb") as f:
            encoders = pickle.load(f)

        for col in cat_cols:
            df[col] = encoders[col].transform(df[col])
            
        #Model
        model = joblib.load(r'crop_recommendation_rf_model.pkl')

        y  = model.predict(df)[0]
        # print("Y:- ",y)
        # print("Type Y: ",type(y))
        
        preds = te.inverse_transform([y])[0]
        
        return JSONResponse(
            status_code=200,
            content={
                'prediction':preds
            }
        )
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"File not found: {str(e)}"}
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid input: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )


