import streamlit as st
from src.sidebar import render_sidebar

# set_page_config must be the first Streamlit call in the script
st.set_page_config(
    page_title="A/B Test Simulator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

st.title("First Order Discount — A/B Test Simulator")
st.markdown("Use the sidebar to configure parameters. Navigate pages using the left panel.")