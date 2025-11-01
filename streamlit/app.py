import streamlit as st
from model_helper import predict


st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload the File", type=['jpg', 'png'])

if uploaded_file is not None:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="Uploaded File")
        prediction = predict(uploaded_file)
        st.info(f"Prediction Class : {prediction}")