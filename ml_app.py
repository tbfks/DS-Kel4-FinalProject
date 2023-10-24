import streamlit as st
import numpy as np

#Machine Learning
import joblib
import os

attribute_info = """
                 - pH: 0-14
                 - Hardness: 47-323
                 - Solids: 320-61127
                 - Chloramines: 0-13
                 - Sulfate: 129-481
                 - Conductivity: 181-753
                 - Organic Carbon: 2-28
                 - Trihalomethanes: 0-124
                 - Turbidity: 1-7
                 """

@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model    

def run_ml_app():
    st.subheader("ML Section")

    with st.expander('Attribute Info'):
         st.markdown(attribute_info)

    st.subheader("Input your Data")
    ph = st.number_input("pH",0,14)
    hardness = st.number_input("Hardness",47,323)
    solids = st.number_input("Solids",320,61127)
    chloramines = st.number_input("Chloramines",0,13)
    sulfate = st.number_input("Sulfate",129,481)
    conductivity = st.number_input("Conductivity",181,753)
    organic_carbon = st.number_input("Organic Carbon",2,28)
    trihalomethanes = st.number_input("Trihalomethanes",0,124)
    turbidity = st.number_input("Turbidity",1,7)

    with st.expander("Your Selected Options"):
        result = {
            "pH":ph,
            "Hardness":hardness,
            "Solids":solids,
            "Chloramines":chloramines,
            "Sulfate":sulfate,
            "Conductivity":conductivity,
            "Organic Carbon":organic_carbon,
            "Trihalomethanes":trihalomethanes,
            "Turbidity":turbidity
        }
    
    st.write(result)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
    
    st.write(encoded_result)

    st.subheader("Prediction Result")
    single_sample = np.array(encoded_result).reshape(1,-1)
    st.write(single_sample)

    model = load_model("model_qda.pkl")

    prediction = model.predict(single_sample)
    pred_prob = model.predict_proba(single_sample)

    st.subheader("Prediction Absolute")
    st.write(prediction)
    st.subheader("Prediction Probability")
    st.write(pred_prob)

    pred_probability = {'Potable':round(pred_prob[0][1]*100,4),
                        'Not Potable':round(pred_prob[0][0]*100,4)}
    if prediction == 1:
        st.success("The Water is Potable")
        st.write(pred_probability)
    else:
        st.warning("The Waster is Not Potable")
        st.write(pred_probability)