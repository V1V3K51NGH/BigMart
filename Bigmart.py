import pandas as pd
import streamlit as st 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import load

st.title(':blue[BIGMART]')

st.header(':green[Item Sales Prediction]')

st.sidebar.header(':gray[User Input Parameters]')

def user_input_features():
    Item_Weight = st.sidebar.number_input("Item Weight:", value=0.0, step=0.1)

    # Item Fat Content Mapping
    Item_Fat_Content_Mapping = {'Low Fat': 0, 'Regular': 2, 'Others': 1}
    Item_Fat_Content = st.sidebar.selectbox('Item Fat Content:', list(Item_Fat_Content_Mapping.keys()), index=0)
    Item_Fat_Content_Numeric = Item_Fat_Content_Mapping[Item_Fat_Content]

    Item_Visibility = st.sidebar.number_input('Enter Item Visibility:', value=0.0000, step=0.01)

    #Item Type Mapping
    Item_Type_Maping = {'Baking Goods': 0, 'Breads':1,'Breakfast': 2, 'Canned': 3,
                        'Dairy': 4, 'Frozen Foods': 5, 'Fruits and Vegetables': 6, 'Hard Drinks': 7, 'Health and Hygiene':8, 'Household': 9,
                        'Meat': 10,'Others': 11, 'Seafood': 12, 'Snack Foods': 13, 'Soft Drinks': 14, 'Starchy Foods': 15}
    Item_Type = st.sidebar.selectbox("Select Item Type", list(Item_Type_Maping.keys()), index=0)
    Item_Type_Numeric = Item_Type_Maping[Item_Type]

    #Item Type Category Mapping
    Item_Type_Category_Mapping = {'Food': 1, 'Drink': 0, 'Non-Consumable': 3}
    Item_Type_Category = st.sidebar.selectbox('Select Item Categoty Type', list(Item_Type_Category_Mapping.keys()), index=0)
    Item_Type_Category_Numeric = Item_Type_Category_Mapping[Item_Type_Category]

    Item_MRP = st.sidebar.number_input("Item MRP:", value=0.0, step=0.1)

    Outlet_Identifier_Mapping = {'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3,'OUT019': 4, 'OUT027': 5, 
                                 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9}
    Outlet_Identifier = st.sidebar.selectbox('Outlet Identifier:', list(Outlet_Identifier_Mapping.keys()), index=0)
    Outlet_Identifier_Numeric = Outlet_Identifier_Mapping[Outlet_Identifier]

    Outlet_Establishment_Year = st.sidebar.selectbox('Outlet Establishment Year:', list(range(1980, 2023)), index=0)
    Outlet_Age = 2023 - Outlet_Establishment_Year

    # Outlet Size Mapping 
    Outlet_Size_Mapping = {'Small': 2, 'Medium': 1, 'High': 0}
    Outlet_Size = st.sidebar.selectbox('Outlet Size:', list(Outlet_Size_Mapping.keys()), index=0)
    Outlet_Size_Numeric = Outlet_Size_Mapping[Outlet_Size]

    #Outlet Location Type Mapping
    Outlet_Location_Type_Mapping = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
    Outlet_Location_Type = st.sidebar.selectbox('Outlet Location Type:', list(Outlet_Location_Type_Mapping.keys()), index=0)
    Outlet_Location_Type_Numeric = Outlet_Location_Type_Mapping[Outlet_Location_Type]

    #Outlet Type Mapping
    Outlet_Type_Mapping = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
    Outlet_Type = st.sidebar.selectbox('Outlet Type', list(Outlet_Type_Mapping.keys()), index=0)
    Outlet_Type_Numeric = Outlet_Type_Mapping[Outlet_Type]


    data = {'Item_Weight': Item_Weight,
            'Item_Fat_Content': Item_Fat_Content_Numeric,
            'Item_Visibility': Item_Visibility,
            'Item_Type': Item_Type_Numeric,
            'Item_MRP': Item_MRP,
            'Outlet_Identifier': Outlet_Identifier_Numeric,
            'Outlet_Size': Outlet_Size_Numeric,
            'Outlet_Location_Type': Outlet_Location_Type_Numeric,
            'Outlet_Type': Outlet_Type_Numeric,
            'Outlet_Age': Outlet_Age,
            'Item_Type_Category': Item_Type_Category_Numeric
            }
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

with st.form(key='prediction_form'):
    if st.form_submit_button('Predict'):
        loaded_model = load('model.sav')
        scaler = load('scaler.sav')

        scaled_features = scaler.transform(df)
        prediction = loaded_model.predict(scaled_features)

        st.subheader('Predicted Result in INR')
        st.write(prediction)


