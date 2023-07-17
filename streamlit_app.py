import streamlit as st
import pandas as pd
import pickle

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
    st.write(city_dict)

    avg_qty = st.slider('How many items do you purchase in a typical transaction?', 0, 20, 5)
    avg_amt = st.number_input('How much do you typically spend in a transaction ($)'
                              , min_value = 0.00, step = 0.01, value = 25.00, format = "%f")
    
    age = 'avg age'
    gender = 'mode gender'
    martial = 'mode status'
    child_count = 'mode count'
    #input values 
    """
    Location: OHE (might be a challenge, mebbe false everything for now)
    Avg amt & Avg qty
    Age & Gender & Marital & Child count
    Freq catt & Freq subcat
    """

#model deployment
#model = pickle.load(open('cust_analysis_treeClass.pkl','rb'))

def predict_age(Length,Diameter,Height,Whole_weight,Shucked_weight,
                Viscera_weight,Shell_weight):
    input=np.array([[Length,Diameter,Height,Whole_weight,Shucked_weight,
                     Viscera_weight,Shell_weight]]).astype(np.float64)
    prediction = model.predict(input)
    
    return int(prediction)
