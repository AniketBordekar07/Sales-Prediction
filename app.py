import numpy as np
import pandas as pd
from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler

# --- Placeholder/Mock Definitions for Required Imports ---
# These mock classes allow the file to run independently without importing your full ML project structure.
class CustomData:
    def __init__(self, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, 
                 Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, 
                 Outlet_Location_Type, Outlet_Type, Item_Identifier):
        
        # 1. Assign all inputs to instance variables
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
        # 2. Create the DataFrame with the exact column names required by the ML pipeline
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
        return pd.DataFrame(custom_data_input_dict)

class PredictPipeline:
    # In a real project, this class loads the saved preprocessor and model.
    def predict(self, features):
        # Mock prediction logic (replace with your actual pipeline logic)
        return [2500.0] 
# ----------------------------------------------------------------------


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        # --- FILLED IN CUSTOM DATA ARGUMENTS (11 features for Big Mart Sales) ---
        data=CustomData(
            # Numerical features (coerced to float/int)
            Item_Weight=float(request.form.get('Item_Weight')),
            Item_Visibility=float(request.form.get('Item_Visibility')),
            Item_MRP=float(request.form.get('Item_MRP')),
            Outlet_Establishment_Year=int(request.form.get('Outlet_Establishment_Year')), 

            # Categorical features (remain as strings)
            Item_Fat_Content=request.form.get('Item_Fat_Content'),
            Item_Type=request.form.get('Item_Type'),
            Outlet_Identifier=request.form.get('Outlet_Identifier'),
            Outlet_Size=request.form.get('Outlet_Size'),
            Outlet_Location_Type=request.form.get('Outlet_Location_Type'),
            Outlet_Type=request.form.get('Outlet_Type'),
            
            # Key identifier needed for imputation
            Item_Identifier=request.form.get('Item_Identifier')
        ) 
        # -----------------------------------------------------------------------

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        
        # Format the result before passing it to the template
        return render_template('home.html',results=f"The predicted sales are: {results[0]:.2f}")
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
