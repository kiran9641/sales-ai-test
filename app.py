import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 AI Sales Prediction App")

data = {
    "Month": [1, 2, 3, 4, 5],
    "Sales": [100, 120, 140, 160, 180]
}

df = pd.DataFrame(data)

st.dataframe(df)

X = df[['Month']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

future_month = st.slider("Select Future Month", 6, 12, 6)

prediction = model.predict([[future_month]])

st.write(f"Predicted Sales: {prediction[0]:.2f}")

st.line_chart(df.set_index("Month"))
