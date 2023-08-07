import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn import tree
import dice_ml

def main():
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Tasty Bytes Customer Propencity Prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    #input elements
    city_dict = {'Boston':0, 'Denver':0, 'New York City':0, 'San Mateo':0, 'Seattle':0}

    location = st.selectbox(
        'Select your city :world_map:',
        ('Others', 'Boston', 'Denver', 'New York City', 'San Mateo', 'Seattle')
    )

    for x in city_dict:
        if(location == x): city_dict.update({x:1});
    #st.write(city_dict) --checking

    avg_qty = st.slider('How many items do you purchase in a typical transaction?', 0, 20, 5)
    avg_amt = st.number_input('How much do you typically spend in a transaction ($)'
                              , min_value = 0.00, step = 0.01, value = 25.00, format = "%.2f")
    
    age = 50 #mean
    gender = 0 #mode (female)
    martial = 0 #mode (single)
    child_count = 0 #mode <but if not single, mode=2>
    freq_cat = 0 #mode (main)
    subcat = 2 #mode (hot)
    
    #input values 
    #Location: OHE (dictionary)
    #Avg amt & Avg qty
    #Age & Gender & Marital & Child count
    #Freq catt & Freq subcat
    
    prediction = predict_model(city_dict, avg_amt, avg_qty, age, gender, martial, child_count,freq_cat,subcat);

    #write a message
    if (prediction == 0):
        st.subheader("You are a LOW spender :man-gesturing-no: :fencer:");
    elif (prediction == 1):
        st.subheader("You are a HIGH spender :moneybag:");
    else: st.subheader(":red[INVALID prediction output]");

#model deployment
model = pickle.load(open('CustAnalyV3.2_Unscaled.pkl','rb'))

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
    #nvm scale is worse
    
    prediction = model.predict(predict_x)
    
    return int(prediction)

def get_counterfactual():
    ohe_customer_us = pd.read_csv("CustAnaly_newCritV1.1.csv")
    
    d = dice_ml.Data(dataframe=ohe_customer_us.drop(columns=["Unnamed: 0", "CUSTOMER_ID","Recency","MEAN_PROFIT","MEMBER_MONTHS","SPEND_MONTH"]),
                      continuous_features=list(X_train.columns), outcome_name='SPEND_RANK')
    backend = 'sklearn'
    m = dice_ml.Model(model=rf, backend=backend)
    exp = dice_ml.Dice(d,m)
    
    # Generate counterfactual examples
    query_instances = predict_x.iloc[[0]]
    dice_exp = exp.generate_counterfactuals(query_instances, total_CFs=4, desired_class="opposite")
    # Visualize counterfactual explanation
    dice_exp.visualize_as_dataframe()

main();
