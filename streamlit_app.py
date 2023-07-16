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

    #input values 
    """
    Location: OHE (might be a challenge, mebbe false everything for now)
    Avg amt & Avg qty
    Age & Gender & Marital & Child count
    Freq cat & Freq subcat
    """

#model deployment
model = pickle.load(open('cust_analysis_treeClass.pkl','rb'))

'''
def predict_age(Length,Diameter,Height,Whole_weight,Shucked_weight,
                Viscera_weight,Shell_weight):
    input=np.array([[Length,Diameter,Height,Whole_weight,Shucked_weight,
                     Viscera_weight,Shell_weight]]).astype(np.float64)
    prediction = model.predict(input)
    
    return int(prediction)
'''
