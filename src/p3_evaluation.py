import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def evaluation():
    st.image("image/3_evaluation.png")
    st.subheader("Step 1: Metrics evaluation", divider='rainbow') 
    st.image("image/metrics.jpeg")
    st.subheader("Step 2: Classification report", divider='rainbow')
    st.image("image/report.jpg")