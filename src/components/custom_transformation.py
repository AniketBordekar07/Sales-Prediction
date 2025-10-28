import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# --- 1. CustomItemWeightImputer ---
class CustomItemWeightImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing Item_Weight:
    1. Uses the median weight associated with the specific Item_Identifier.
    2. Uses the median weight associated with the Item_Type as a fallback.
    """
    def __init__(self):
        self.item_id_weight_mapping = None
        self.item_type_weight_mapping = None

    def fit(self, X, y=None):
        df = X.copy()
        
        # 1. Mapping based on Item_Identifier (using median for robustness/explicitness)
        # Note: If Item_Identifier has only one weight, median == mean.
        item_id_weight_pivot = df.pivot_table(
            values='Item_Weight', 
            index='Item_Identifier', 
            aggfunc='median' # Explicitly use median 
        )
        self.item_id_weight_mapping = item_id_weight_pivot['Item_Weight'].to_dict()

        # 2. Mapping based on Item_Type (for fallback)
        item_type_weight_pivot = df.pivot_table(
            values='Item_Weight', 
            index='Item_Type', 
            aggfunc='median'
        )
        self.item_type_weight_mapping = item_type_weight_pivot['Item_Weight'].to_dict()

        return self

    def transform(self, X):
        df = X.copy()

        # Check for required columns before transformation
        if any(col not in df.columns for col in ['Item_Weight', 'Item_Identifier', 'Item_Type']):
             raise KeyError("Input DataFrame must contain 'Item_Weight', 'Item_Identifier', and 'Item_Type'.")

        # Step 1: Fill missing Item_Weight using Item_Identifier mapping
        df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Identifier'].map(self.item_id_weight_mapping))

        # Step 2: Fill remaining missing Item_Weight using Item_Type mapping
        df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Type'].map(self.item_type_weight_mapping))

        return df


# --- 2. CustomOutletSizeImputer ---
class CustomOutletSizeImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer for 'Outlet_Size' based on the mode of each 'Outlet_Type'.
    """
    def __init__(self):
        self.outlet_type_size_mapping = None

    def fit(self, X, y=None):
        """
        Learn mapping between Outlet_Type and most frequent Outlet_Size.
        """
        if 'Outlet_Size' in X.columns and 'Outlet_Type' in X.columns:
            # Calculate mode (most frequent value) for each Outlet_Type
            outlet_type_size_pivot = (
                X.pivot_table(
                    values='Outlet_Size',
                    index='Outlet_Type',
                    # aggfunc=lambda x: x.mode()[0] safely gets the first mode
                    aggfunc=lambda x: x.mode()[0] 
                )
                .reset_index()
            )
            self.outlet_type_size_mapping = dict(
                zip(outlet_type_size_pivot['Outlet_Type'], outlet_type_size_pivot['Outlet_Size'])
            )
        return self

    def transform(self, X):
        """
        Fill missing values in 'Outlet_Size' using the learned mapping.
        """
        X = X.copy()
        if self.outlet_type_size_mapping is not None and 'Outlet_Size' in X.columns and 'Outlet_Type' in X.columns:
            # Fill NaNs in Outlet_Size using the mapping applied to Outlet_Type
            X['Outlet_Size'] = X['Outlet_Size'].fillna(
                X['Outlet_Type'].map(self.outlet_type_size_mapping)
            )
        return X
