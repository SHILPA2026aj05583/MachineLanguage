import streamlit as st
import pandas as pd
import numpy as np

st.title("My First AI/ML Explorer App 🚀")

# Sidebar inputs
st.sidebar.header("User Settings")
slider_val = st.sidebar.slider("Select a multiplier:", 1, 10, 5)

# Main area
st.write(f"You selected a multiplier of **{slider_val}**.")

# Generate some dummy data
data = pd.DataFrame({
    'X': range(1, 11),
    'Y': [x * slider_val for x in range(1, 11)]
})

st.subheader("Generated Data & Chart")
st.line_chart(data.set_index('X'))
