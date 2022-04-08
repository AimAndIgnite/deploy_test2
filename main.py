import streamlit as st
import xgboost as xgb
import pandas as pd
# from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
import pickle

st.header("Number of Clicks Prediction")
# st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("data.csv")

xgb_model = xgb.XGBRegressor()

# xgb_model.load_model("xgb_model.json")
file_name = "xgb_reg.pkl"

xgb_model_loaded = pickle.load(open(file_name, "rb"))


# X = data[["Impressions", "CPC Bid", "Cost"]]
# y = data["Clicks"]

# X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, shuffle=False, random_state=1)

# # load model
# xgb_model = xgb.XGBRegressor()
# # xgb_model.load_model("test")

# xgb_model.fit(X_train, y_train)
# predictions = xgb_model.predict(X_test)

# if st.checkbox('Show dataframe'):
#     data


# input1 = st.slider('Select amount of CPC Bid: ', 0.0, max(data["CPC Bid"]), 1.0)
# input2 = st.slider('Select ampunt of Cost: ', 0.0, max(data["Cost"]), 1.0)
# input3 = st.slider('Select amount of Impressions: ', 0, max(data["Impressions"]), 1)


# if st.button('Make Prediction'):
#     prediction = xgb_model.predict(input1, input2, input3)
#     print("final pred", np.squeeze(prediction, -1))
#     st.write(f"Predicted clicks: {np.squeeze(prediction, -1)} clicks")
