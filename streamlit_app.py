import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn import tree

def main():
    st.title('Customer Prediction Model')
    st.header('Predictions')

    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Customer Segmentation Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    #input elements
    city_dict = {'Boston':0, 'Denver':0, 'New York City':0, 'San Mateo':0, 'Seattle':0}

    location = st.selectbox(
        'Select your city. :world_map:',
        ('Others', 'Boston', 'Denver', 'New York City', 'San Mateo', 'Seattle')
    )

    for x in city_dict:
        if(location == x): city_dict.update({x:1});
    #st.write(city_dict) --checking

    avg_qty = st.slider('How many items do you purchase in a typical transaction?', 0, 20, 5)
    avg_amt = st.number_input('How much do you typically spend in a transaction ($)'
                              , min_value = 0.00, step = 0.01, value = 25.00, format = "%.2f")
    
    age = 50 
    gender = 2
    martial = 0
    child_count = 0
    freq_cat = 0 #main
    subcat = 1 #warm
    #input values 
    """
    Location: OHE (might be a challenge, mebbe false everything for now)
    Avg amt & Avg qty
    Age & Gender & Marital & Child count
    Freq catt & Freq subcat
    """
    
    prediction = predict_model(city_dict, avg_amt, avg_qty, age, gender, martial, child_count,freq_cat,subcat);
    st.write(prediction);

#model deployment
model = pickle.load(open('CustAnalyV3.1.pkl','rb'))

def predict_model(city_dict, avg_amt, avg_qty, age, gender, martial, child_count,freq_cat,freq_subcat):
    input=np.array([[gender,martial,child_count,age
                     ,city_dict['Seattle'],city_dict['Boston'],city_dict['New York City'],city_dict['Denver'],city_dict['San Mateo']
                     ,avg_amt,avg_qty,freq_cat,freq_subcat,6]]).astype(np.float64)
    predict_x = pd.DataFrame({
    "GENDER": gender,
    "MARITAL_STATUS": martial,
    "CHILDREN_COUNT": child_count,
    "AVG_AMT": avg_amt,
    "AVG_QUANTITY": avg_qty,
    "FREQ_CATEGORY": freq_cat,
    "FREQ_SUBCAT": freq_subcat,
    "AGE": age,
    "CITY_New York City": city_dict['New York City'],
    "CITY_Seattle": city_dict['Seattle'],
    "CITY_San Mateo": city_dict['San Mateo'],
    "CITY_Denver": city_dict['Denver'],
    "CITY_Boston": city_dict['Boston']
    }, index = [0])

    #might have to scale values
    
    prediction = model.predict(predict_x)
    
    return int(prediction)

main();
