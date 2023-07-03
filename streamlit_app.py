import streamlit as st
import pandas as pd

st.title('Sales Report')

st.header('Q1 Results')

q1_sales = {
    'January': 100,
    'February': 110,
    'March': 115
}

st.write('January was the start of the year')
st.write(q1_sales)

st.header('Q2 Results')

q2_sales = {
    'April': 150,
    'May': 200,
    'June': 250
}

'Q2 had better results:smile:'
q2_df = pd.DataFrame(q2_sales.items(),
                     columns=['Month', 'Amount'])

st.table(q2_df)
st.dataframe(q2_df)
