import streamlit as st
from ml_app import run_ml_app

def main():
    st.title("Water Potability Predictor")

    menu = ['Home','Machine Learning']

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to Our Homepage")
        st.write("Click the Menu to the Machine Learning")
    elif choice == "Machine Learning":
        run_ml_app()

        
if __name__ == '__main__':
    main()