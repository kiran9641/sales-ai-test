import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="AI Sales Predictor", page_icon="📈", layout="wide")

# Title
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>🚀 AI Sales Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Smart forecasting powered by Machine Learning</h4>", unsafe_allow_html=True)

# Dataset selector
st.sidebar.header("📂 Choose Dataset")

dataset_option = st.sidebar.selectbox(
    "Select Sample Dataset",
    ["📊 Basic Growth", "🛒 E-commerce Sales", "🏪 Retail Seasonal"]
)

# Different datasets
if dataset_option == "📊 Basic Growth":
    df = pd.DataFrame({
        "Month": [1,2,3,4,5],
        "Sales": [100,120,140,160,180]
    })

elif dataset_option == "🛒 E-commerce Sales":
    df = pd.DataFrame({
        "Month": [1,2,3,4,5],
        "Sales": [200,240,260,300,350]
    })

elif dataset_option == "🏪 Retail Seasonal":
    df = pd.DataFrame({
        "Month": [1,2,3,4,5],
        "Sales": [150,130,170,160,200]
    })

# Show data
st.subheader("📊 Historical Sales Data")
st.dataframe(df, use_container_width=True)

# Train model
X = df[['Month']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

# Prediction section
st.subheader("🔮 Predict Future Sales")

future_month = st.slider("Select Future Month", 6, 12, 6)

prediction = model.predict([[future_month]])

# Metrics (nice UI)
col1, col2, col3 = st.columns(3)

col1.metric("📅 Selected Month", future_month)
col2.metric("📈 Predicted Sales", f"{prediction[0]:.2f}")
col3.metric("📊 Growth", f"{(prediction[0]-df['Sales'].iloc[-1]):.2f}")

# Graph
st.subheader("📉 Sales Trend + Prediction")

fig, ax = plt.subplots()
ax.plot(df["Month"], df["Sales"], marker='o', label="Actual Sales")

# prediction point
ax.scatter(future_month, prediction, color='red', s=100, label="Prediction")

# extend line visually
extended_months = list(df["Month"]) + [future_month]
extended_sales = list(df["Sales"]) + [prediction[0]]
ax.plot(extended_months, extended_sales, linestyle='dashed', color='orange')

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

# Multi prediction
st.subheader("📅 Predict Next 3 Months")

if st.button("🚀 Generate Forecast"):
    results = []
    for m in range(future_month, future_month+3):
        pred = model.predict([[m]])[0]
        results.append((m, pred))

    forecast_df = pd.DataFrame(results, columns=["Month", "Predicted Sales"])
    st.table(forecast_df)

# AI Insights
st.subheader("🤖 AI Business Insights")

trend = "increasing 📈" if df["Sales"].iloc[-1] > df["Sales"].iloc[0] else "fluctuating ⚠️"

st.info(f"""
✅ Sales trend is {trend}

💡 Recommendations:
- Prepare inventory based on expected growth  
- Plan marketing campaigns  
- Optimize pricing strategy  
""")

# Footer
st.markdown("---")
st.markdown("<center>Developed by Kiran | AI Demo 🚀</center>", unsafe_allow_html=True)
