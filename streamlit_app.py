import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn import tree
import dice_ml
import IPython
import math

def main():
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Tasty Bytes Customer Propencity Prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.sidebar.header("Tell us more about yourself")

    #input elements
    city_dict = {'Boston':0, 'Denver':0, 'New York City':0, 'San Mateo':0, 'Seattle':0}
    location = st.sidebar.selectbox("Select your city",
        ('Others', 'Boston', 'Denver', 'New York City', 'San Mateo', 'Seattle')
    )

    for x in city_dict:
        if(location == x): city_dict.update({x:1});
    #st.write(city_dict) --checking

    avg_qty = st.sidebar.slider('How many items do you purchase in a typical transaction?', 0, 20, 5)
    avg_amt = st.sidebar.number_input('How much do you typically spend in a transaction ($)'
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

    predict_x = pd.DataFrame({
    "GENDER": gender,
    "MARITAL_STATUS": martial,
    "CHILDREN_COUNT": child_count,
    "AVG_AMT": avg_amt,
    "AVG_QUANTITY": avg_qty,
    "FREQ_CATEGORY": freq_cat,
    "FREQ_SUBCAT": subcat,
    "AGE": age,
    "CITY_New York City": city_dict['New York City'],
    "CITY_Seattle": city_dict['Seattle'],
    "CITY_San Mateo": city_dict['San Mateo'],
    "CITY_Denver": city_dict['Denver'],
    "CITY_Boston": city_dict['Boston']
    }, index = [0])
    
    prediction = predict_model(predict_x);
    #write a message
    if (prediction == 0):
        st.header("You are a LOW spender :man-gesturing-no: :fencer:");
    elif (prediction == 1):
        st.header("You are a HIGH spender :moneybag:");
    else: st.subheader(":red[INVALID prediction output]");

    #st.dataframe(predict_x)
    get_counterfactual(predict_x,prediction);

#model deployment
model = pickle.load(open('CustAnalyV3.2_Unscaled.pkl','rb'))

def predict_model(predict_x):
    #might have to scale values
    #nvm scale is worse
    
    prediction = model.predict(predict_x)
    
    return int(prediction)

def get_counterfactual(predict_x, prediction):
    ohe_customer_us = pd.read_csv("CustAnaly_newCritV1.1.csv")
    
    d = dice_ml.Data(dataframe=ohe_customer_us.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "CUSTOMER_ID","Recency","MEAN_PROFIT","MEMBER_MONTHS","SPEND_MONTH"]),
                      continuous_features=list(predict_x.columns), outcome_name='SPEND_RANK')
    backend = 'sklearn'
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d,m)

    # Generate similar examples
    query_instances = predict_x.iloc[[0]]
    dice_same = exp.generate_counterfactuals(query_instances, total_CFs=5, desired_class=prediction)
    # Visualize counterfactual explanation
    same_df = dice_same.cf_examples_list[0].final_cfs_df
    
    
    # Generate counterfactual examples
    dice_exp = exp.generate_counterfactuals(query_instances, total_CFs=5, desired_class="opposite")
    # Visualize counterfactual explanation
    exp_df = dice_exp.cf_examples_list[0].final_cfs_df
    

    #different between user and others
    #calculate % change
    percent_chg_amt = (exp_df["AVG_AMT"].mean() - predict_x["AVG_AMT"].iloc[0]) / predict_x["AVG_AMT"].iloc[0] * 100
    percent_chg_qty = math.ceil(exp_df["AVG_QUANTITY"].mean() - predict_x["AVG_QUANTITY"].iloc[0])

    st.header('Here\'s a breakdown of your spending habits');
    col1,col2 = st.columns(2)
    if (percent_chg_amt >= 0):
        col1.subheader("\nTry spending {:.2f}% more with us :hand_with_index_and_middle_fingers_crossed:".format(percent_chg_amt));
    elif (percent_chg_amt<0):
        st.subheader("\nYou are currently spending {:.2f}% more than others! :muscle:".format(abs(percent_chg_amt)));

    if (percent_chg_qty == 0):
        st.subheader("\nYour cart size is just right :ok_hand:")
    elif (percent_chg_qty > 0):
        st.subheader("\nYou can put {:d} more items in your basket :shopping_trolley:".format(int(percent_chg_qty)));
    elif (percent_chg_qty<0):
        st.subheader("\nYou have {:d} more items than others! :first_place_medal:".format(abs(int(percent_chg_qty))));

    #show examples of other customers
    if (prediction == 0):
        st.subheader("Here are other low spenders... :woman-tipping-hand:");
    elif (prediction == 1):
        st.subheader("Here are other high spenders like you! :money_with_wings:");
    with st.expander ('Guess who!'):
        st.dataframe(same_df.drop(['GENDER','MARITAL_STATUS','CHILDREN_COUNT','AGE'],axis=1))
    
    if (prediction == 0):
        st.subheader("\nHere's how you can be a high spender :gem:");
    elif (prediction == 1):
        st.subheader("\nWatch out! You could become a low spender too! :broken_heart:");
    with st.expander ('Take a peek >_<'):    
        st.dataframe(exp_df.drop(['GENDER','MARITAL_STATUS','CHILDREN_COUNT','AGE'],axis=1))

main();
