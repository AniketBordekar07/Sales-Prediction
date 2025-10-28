import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin 

# --- PROJECT IMPORTS ---
# Ensure these files and paths exist in your project structure:
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.custom_transformation import CustomItemWeightImputer, CustomOutletSizeImputer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation pipeline initiated")

            # --- 1. Define Column Groups ---
            scaling_cols = ['Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
            weight_imputer_input_cols = ['Item_Weight', 'Item_Identifier', 'Item_Type']
            
            # Categorical columns for OHE/imputation
            categorical_cols = [
                'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size'
            ]
            
            # --- 2. Define Pipelines ---

            # Pipeline A: Item_Weight Imputation & Scaling
            weight_processing_pipeline = Pipeline(steps=[
                ('custom_imputer', CustomItemWeightImputer()),
                # Nested ColumnTransformer to apply StandardScaler only to 'Item_Weight' 
                ('feature_selector_scaler', ColumnTransformer(
                    transformers=[
                        ('scaler', StandardScaler(), ['Item_Weight']),
                    ],
                    remainder='drop' # Drops auxiliary columns Item_Identifier, Item_Type
                ))
            ])
            logging.info("Weight Processing Pipeline defined.")


            # Pipeline B: Remaining Numerical Columns Scaling
            scaling_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            logging.info("Scaling Pipeline defined.")


            # Pipeline C: Categorical Columns Imputation & Encoding
            cat_pipeline = Pipeline(steps=[
                ('custom_size_imputer', CustomOutletSizeImputer()), 
                ('mode_imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            logging.info("Categorical Pipeline defined.")


            # --- 3. Combine Pipelines in Main ColumnTransformer ---
            preprocessor = ColumnTransformer(
                transformers=[
                    ('weight_transform', weight_processing_pipeline, weight_imputer_input_cols),
                    ('scaling_transform', scaling_pipeline, scaling_cols),
                    ('categorical_transform', cat_pipeline, categorical_cols)
                ],
                remainder='drop' 
            )
            
            logging.info("Final ColumnTransformer object created.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # ******************************************************************
            # CRITICAL FIX: Read the string paths into pandas DataFrames here.
            # ******************************************************************
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Item_Outlet_Sales"
            
            # These lines (115 and below) now correctly call .drop() on a DataFrame
            # Line 115 (based on your traceback)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe.")

            # Apply transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target into numpy arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)]

            logging.info("Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin 

# --- PROJECT IMPORTS ---
# Ensure these files and paths exist in your project structure:
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.custom_transformation import CustomItemWeightImputer, CustomOutletSizeImputer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation pipeline initiated")

            # --- 1. Define Column Groups ---
            scaling_cols = ['Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
            weight_imputer_input_cols = ['Item_Weight', 'Item_Identifier', 'Item_Type']
            
            # Categorical columns for OHE/imputation
            categorical_cols = [
                'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size'
            ]
            
            # --- 2. Define Pipelines ---

            # Pipeline A: Item_Weight Imputation & Scaling
            weight_processing_pipeline = Pipeline(steps=[
                ('custom_imputer', CustomItemWeightImputer()),
                # Nested ColumnTransformer to apply StandardScaler only to 'Item_Weight' 
                ('feature_selector_scaler', ColumnTransformer(
                    transformers=[
                        ('scaler', StandardScaler(), ['Item_Weight']),
                    ],
                    remainder='drop' # Drops auxiliary columns Item_Identifier, Item_Type
                ))
            ])
            logging.info("Weight Processing Pipeline defined.")


            # Pipeline B: Remaining Numerical Columns Scaling
            scaling_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            logging.info("Scaling Pipeline defined.")


            # Pipeline C: Categorical Columns Imputation & Encoding
            cat_pipeline = Pipeline(steps=[
                ('custom_size_imputer', CustomOutletSizeImputer()), 
                ('mode_imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            logging.info("Categorical Pipeline defined.")


            # --- 3. Combine Pipelines in Main ColumnTransformer ---
            preprocessor = ColumnTransformer(
                transformers=[
                    ('weight_transform', weight_processing_pipeline, weight_imputer_input_cols),
                    ('scaling_transform', scaling_pipeline, scaling_cols),
                    ('categorical_transform', cat_pipeline, categorical_cols)
                ],
                remainder='drop' 
            )
            
            logging.info("Final ColumnTransformer object created.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # ******************************************************************
            # CRITICAL FIX: Read the string paths into pandas DataFrames here.
            # ******************************************************************
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Item_Outlet_Sales"
            
            # These lines (115 and below) now correctly call .drop() on a DataFrame
            # Line 115 (based on your traceback)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe.")

            # Apply transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target into numpy arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)]

            logging.info("Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
