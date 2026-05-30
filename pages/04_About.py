import streamlit as st
from src.sidebar import render_sidebar

render_sidebar()

st.title("About")
st.markdown("""
    **Vlad Lee** Data Scientist (MIDS, UC Berkeley)

    🔗 [LinkedIn](https://www.linkedin.com/in/vlad-lee)  
            
    💻 [GitHub Repository](https://github.com/Vlad-Lee/ab-test-first-order-discount)

    📧 vlad7984@gmail.com
""")