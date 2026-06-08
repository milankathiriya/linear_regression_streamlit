import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Simple Linear Regression",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction using Simple Linear Regression")
st.markdown("---")

# ----------------------------------
# Sample Dataset
# ----------------------------------

data = {
    "House_Size_SqFt": [
        500, 700, 900, 1100, 1300,
        1500, 1700, 1900, 2100, 2300,
        2500, 2700, 2900, 3100, 3300
    ],
    "Price_Lakh": [
        20, 28, 35, 43, 50,
        58, 66, 74, 82, 90,
        98, 106, 114, 122, 130
    ]
}

df = pd.DataFrame(data)

# ----------------------------------
# Dataset Section
# ----------------------------------

st.header("📊 House Dataset")

st.dataframe(df, use_container_width=True)

# ----------------------------------
# Scatter Plot
# ----------------------------------

st.header("📈 Relationship Between House Size & Price")

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(
    df["House_Size_SqFt"],
    df["Price_Lakh"]
)

ax.set_xlabel("House Size (SqFt)")
ax.set_ylabel("Price (Lakh ₹)")
ax.set_title("House Size vs Price")

st.pyplot(fig)

# ----------------------------------
# Prepare Data
# ----------------------------------

X = df[["House_Size_SqFt"]]
y = df["Price_Lakh"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# Train Model
# ----------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# ----------------------------------
# Predictions
# ----------------------------------

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# ----------------------------------
# Model Metrics
# ----------------------------------

st.header("🤖 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2:.4f}")

with col2:
    st.metric("MAE", f"{mae:.2f} Lakh")

# ----------------------------------
# Regression Equation
# ----------------------------------

st.header("📐 Regression Equation")

slope = model.coef_[0]
intercept = model.intercept_

st.success(
    f"Price = {slope:.4f} × House_Size + {intercept:.4f}"
)

# ----------------------------------
# Regression Line
# ----------------------------------

st.header("📉 Regression Line")

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.scatter(
    X,
    y,
    label="Actual Data"
)

ax2.plot(
    X,
    model.predict(X),
    linewidth=3,
    label="Regression Line"
)

ax2.set_xlabel("House Size (SqFt)")
ax2.set_ylabel("Price (Lakh ₹)")
ax2.legend()

st.pyplot(fig2)

# ----------------------------------
# Prediction Section
# ----------------------------------

st.header("🏡 Predict New House Price")

size = st.slider(
    "Select House Size (SqFt)",
    min_value=500,
    max_value=4000,
    value=2000,
    step=100
)

predicted_price = model.predict([[size]])[0]

st.subheader(
    f"Estimated Price: ₹ {predicted_price:.2f} Lakh"
)

# ----------------------------------
# Actual vs Predicted
# ----------------------------------

st.header("📊 Actual vs Predicted")

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.dataframe(
    comparison_df.round(2),
    use_container_width=True
)

fig3, ax3 = plt.subplots(figsize=(8,5))

ax3.plot(
    comparison_df["Actual"].values,
    marker="o",
    label="Actual"
)

ax3.plot(
    comparison_df["Predicted"].values,
    marker="s",
    label="Predicted"
)

ax3.set_ylabel("Price (Lakh ₹)")
ax3.set_title("Actual vs Predicted Price")
ax3.legend()

st.pyplot(fig3)

# ----------------------------------
# Conclusion
# ----------------------------------

st.markdown("---")

st.header("📚 Learning Summary")

st.info(
    """
    Simple Linear Regression learns a straight-line relationship between:
    
    • Independent Variable (X): House Size
    
    • Dependent Variable (Y): House Price
    
    Formula:
    
    Price = m × Size + c
    
    where:
    - m = Slope (Coefficient)
    - c = Intercept
    """
)
