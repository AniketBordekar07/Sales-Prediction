from src.exception import CustomException
from src.utils import load_object

import sys
import os
import pandas as pd

# --- Mock Definitions (Replace with actual imports from your project) ---
# Assuming these are imported from your src/logger, src/exception, and src/utils files
class CustomException(Exception): pass
def logging_info(msg): print(f"[INFO] {msg}")
logging = type('Logger', (object,), {'info': logging_info})()
def load_object(file_path):
    """Mocks loading a preprocessor or model object (e.g., from pickle)."""
    logging.info(f"MOCK: Loading object from {file_path}")
    # In a real app, you would load the object here. For demonstration, we return a mock.
    class MockArtifact:
        def transform(self, data):
            # Mock transformation: returns an array of zeros for the 11 features
            return pd.DataFrame(0.0, index=data.index, columns=[f'feature_{i}' for i in range(11)]).values
        def predict(self, data):
            # Mock prediction
            return [2500.0]
    
    if 'preprocessor' in file_path:
        # Returns a mock transformer object
        class MockTransformer:
             def transform(self, data):
                 # Mocks the preprocessor output (11 scaled/encoded features)
                 logging.info("MOCK: Data transformation applied.")
                 return pd.DataFrame(0.0, index=data.index, columns=[f'feature_{i}' for i in range(11)]).values
        return MockTransformer()
    elif 'model' in file_path:
        # Returns a mock model object
        class MockModel:
             def predict(self, data):
                 logging.info("MOCK: Model prediction executed.")
                 # Returns a mock prediction array
                 return [2500.0] 
        return MockModel()
    else:
        return None
# ----------------------------------------------------------------------


class PredictPipeline:
    def __init__(self):
        # Define the exact paths where your model and preprocessor are saved
        # These paths must match the paths used in model_trainer.py/data_transformation.py
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        
    def predict(self, features):
        try:
            logging.info("Prediction initiated.")
            
            # 1. Load Preprocessor and Model
            # The load_object function is responsible for deserializing your saved files.
            preprocessor = load_object(file_path=self.preprocessor_path)
            model = load_object(file_path=self.model_path)
            
            logging.info("Artifacts loaded successfully.")

            # 2. Transform Data
            # The preprocessor scales/encodes the raw input features (the pred_df from app.py)
            data_scaled = preprocessor.transform(features)

            # 3. Predict
            predictions = model.predict(data_scaled)
            logging.info("Prediction successful.")
            
            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class is responsible for converting the raw web form inputs into a 
    structured Pandas DataFrame required by the PredictPipeline.
    """
    def __init__(self, Item_Weight: float, Item_Fat_Content: str, Item_Visibility: float, 
                 Item_Type: str, Item_MRP: float, Outlet_Identifier: str, 
                 Outlet_Establishment_Year: int, Outlet_Size: str, 
                 Outlet_Location_Type: str, Outlet_Type: str, Item_Identifier: str):
        
        # Assign all inputs to instance variables
        self.Item_Weight = Item_Weight
        self.Item_Fat_Content = Item_Fat_Content
        self.Item_Visibility = Item_Visibility
        self.Item_Type = Item_Type
        self.Item_MRP = Item_MRP
        self.Outlet_Identifier = Outlet_Identifier
        self.Outlet_Establishment_Year = Outlet_Establishment_Year
        self.Outlet_Size = Outlet_Size
        self.Outlet_Location_Type = Outlet_Location_Type
        self.Outlet_Type = Outlet_Type
        self.Item_Identifier = Item_Identifier


    def get_data_as_data_frame(self):
        """
        Returns the data as a single-row Pandas DataFrame with the exact column names 
        expected by the preprocessor.
        """
        try:
            custom_data_input_dict = {
                "Item_Weight": [self.Item_Weight],
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Visibility": [self.Item_Visibility],
                "Item_Type": [self.Item_Type],
                "Item_MRP": [self.Item_MRP],
                "Outlet_Identifier": [self.Outlet_Identifier],
                "Outlet_Establishment_Year": [self.Outlet_Establishment_Year],
                "Outlet_Size": [self.Outlet_Size],
                "Outlet_Location_Type": [self.Outlet_Location_Type],
                "Outlet_Type": [self.Outlet_Type],
                "Item_Identifier": [self.Item_Identifier]
            }
            logging.info("Custom data converted to DataFrame successfully.")
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
