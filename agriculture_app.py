import numpy as np
import pandas as pd
import streamlit as st
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

# Load the models and scalers for Crop and Fertilizer Recommendations
rfc = joblib.load('crop_recommendation_model.h5')
ms = joblib.load('scaler.h5')
pickled_model = pickle.load(open('ferti.pkl', 'rb'))

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
} 

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Soil Testing", "Crop Recommendation", "Fertilizer Recommendation", "Farmers' Schemes", "Bio-Organic Vermicompost"])

if options == "Home":
    st.subheader("WELCOME TO OUR KISAN SERVICES")
    st.write("In this, we provide the Farmers Required Agriculture Services")
    
    # Add content about farming in India
    st.write("### Importance of Farming in India")
    st.write("""Farming in India is a cornerstone of the economy, employing over 50% of the population and contributing significantly to the country's GDP. With diverse agro-climatic zones, India produces a wide variety of crops, ensuring food security and supporting local cultures and traditions. However, the agricultural sector faces challenges such as climate change, limited access to technology, and financial constraints, which hinder productivity and market access. To address these issues, the government has implemented initiatives like the PM-KISAN scheme for direct income support, the Soil Health Card Scheme, and crop insurance through the Pradhan Mantri Fasal Bima Yojana. Looking ahead, there is a growing emphasis on sustainable practices and the integration of technology, positioning Indian agriculture for modernization and resilience in the face of changing environmental and economic landscapes""")
    
    # Images related to farming
    st.image("farmer.jpg", caption="Indian Farmer in the Field", use_column_width=True)
    
    st.write("""### Farming Practices
    Farmers employ various methods of cultivation, including:
    - **Traditional Methods**: Many farmers continue to use traditional practices, relying on indigenous knowledge passed down through generations.
    - **Modern Techniques**: Increasingly, farmers are adopting modern agricultural techniques such as precision farming, drip irrigation, and the use of high-yielding variety (HYV) seeds to enhance productivity.
    - **Organic Farming**: There is a rising trend toward organic farming, where farmers use natural fertilizers and pest control methods to cultivate crops sustainably.""")
    
    st.image("farmer2.jpg", caption="Harvesting Crops", use_column_width=True)
    st.image("farmer4.jpg", caption="Modern Farming Techniques", use_column_width=True)

elif options == "Soil Testing":
    st.subheader("Soil Testing Information")
    st.write("""Soil testing is crucial for understanding soil health and nutrient content. 
    Here are some resources that explain the process and importance of soil testing:""")
    
    # Links to YouTube videos
    st.markdown("[How to Test Your Soil at Home](https://youtu.be/K9LgbrkdVnY?si=hTPqLElrYmg1ALnv)", unsafe_allow_html=True)
    st.markdown("[Understanding Soil Testing](https://youtu.be/H93vEx5UOGc?si=EVX0zNUo4QCLGi_y)", unsafe_allow_html=True)
    st.markdown("[Interpreting Soil Test Results](https://youtu.be/zcZ17CY7njw?si=HOQKsbnxtzBu3o8S)", unsafe_allow_html=True)

    st.write("""Make sure to follow the instructions in these videos to get accurate soil test results. 
    Understanding your soil's nutrient levels will help you make better decisions for crop selection and fertilizer use.""")

elif options == "Crop Recommendation":
    st.subheader("Crop Recommendation System")
    
    # User input form
    with st.form(key='crop_form'):
        N = st.number_input("Nitrogen (N)", min_value=0)
        P = st.number_input("Phosphorus (P)", min_value=0)
        k = st.number_input("Potassium (K)", min_value=0)
        temperature = st.number_input("Temperature (°C)", min_value=0.0)
        humidity = st.number_input("Humidity (%)", min_value=0)
        ph = st.number_input("pH", min_value=0.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
        
        submit_button = st.form_submit_button("Recommend Crop")

    # Recommendation function
    def recommend_crop(N, P, k, temperature, humidity, ph, rainfall):
        features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
        transformed_features = ms.transform(features)
        prediction = rfc.predict(transformed_features)
        return prediction[0]

    if submit_button:
        predict = recommend_crop(N, P, k, temperature, humidity, ph, rainfall)
        
        if predict in crop_dict:
            crop_recommendation = crop_dict[predict]
            st.success(f"The best crop to cultivate is: {crop_recommendation}")
        else:
            st.error("Sorry, we are not able to recommend a proper crop for this environment.")

elif options == "Fertilizer Recommendation":
    st.subheader("Fertilizer Recommendation System")
    
    # Fertilizer prediction UI
    activities1 = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']
    option1 = st.sidebar.selectbox('Which Soil Type would you like to use?', activities1)

    activities2 = ['Sugarcane', 'Cotton', 'Millets', 'rice', 'Pulses', 'Wheat', 'Tobacco', 'Barley', 'Oil seeds', 'Ground Nuts', 'Maize']
    option2 = st.sidebar.selectbox('Which Crop Type would you like to use?', activities2)

    temperature = st.number_input("Enter the temperature in Celsius")
    humidity = st.number_input("Enter the humidity")
    nitrogen = st.number_input("Enter the nitrogen content in soil")
    potassium = st.number_input("Enter the potassium content in soil")
    phosphorous = st.number_input("Enter the phosphorous content in soil")

    # Map soil type and crop type to integers
    soiltype_map = {'Loamy': 2, 'Sandy': 4, 'Clayey': 1, 'Black': 0, 'Red': 3}
    croptype_map = {
        'Sugarcane': 8, 'Cotton': 1, 'Millets': 4, 'rice': 6, 
        'Pulses': 7, 'Wheat': 10, 'Tobacco': 9, 'Barley': 0, 
        'Oil seeds': 5, 'Ground Nuts': 2, 'Maize': 3
    }

    soiltype = soiltype_map.get(option1, 3)
    croptype = croptype_map.get(option2, 3)

    inputs = [[temperature, humidity, soiltype, croptype, nitrogen, potassium, phosphorous]]

    result = ""

    if st.button('Predict'):
        result = pickled_model.predict(inputs)
        fertilizer_dict = {
            6: "Urea", 5: "DAP", 4: "28-28", 3: "20-20", 
            2: "17-17-17", 1: "14-35-14", 0: "10-26-26"
        }
        result = fertilizer_dict.get(result[0], "Unknown Fertilizer")
        st.success('Fertilizer to be used is: {}'.format(result))

elif options == "Farmers' Schemes":
    st.subheader("Farmers' Schemes in India")
    st.write("""The Indian government provides various schemes to support farmers:
                1. Rythu Bandhu Scheme
                2. Rythu Bima Scheme
                3. Kisan Credit Card (KCC)
                4. Pradhan Mantri Fasal Bima Yojana (PMFBY)
                5. Telangana State Agricultural Mission (TSAM)
                6. Sheep Distribution Scheme
                7. Farmers’ Training Programs""")
    
    # Links to YouTube videos
    st.markdown("[RYTHU BANDHU APPLICATION PROCESS](https://youtu.be/hZVhswYdVeY?si=0yW3X26orrZ-gG-o)", unsafe_allow_html=True)
    st.markdown("[RYTHU BHEEMA](https://youtu.be/nYUm2x_drfg?si=MXbUFQBD1rpkJxPD)", unsafe_allow_html=True)
    st.markdown("[Pradhan Mantri Fasal Bima Yojana (PMFBY)](https://youtu.be/46ZSJ4qTvSQ?si=KOS3xa_ULt9HjcHO)", unsafe_allow_html=True)
    st.markdown("[official website to apply for RYTHU BHEMA](https://rythubandhu.telangana.gov.in/Default_LIC1.aspx)", unsafe_allow_html=True)
    st.markdown("[RYTHU BANDHU APPLICATION PROCESS WEBSITE](https://rythubandhu.telangana.gov.in/Default_RB1.aspx)", unsafe_allow_html=True)
    st.markdown("[Pradhan Mantri Fasal Bima Yojana (PMFBY) APPLICATION PROCESS Website](https://pmfby.gov.in/)", unsafe_allow_html=True)

elif options == "Bio-Organic Vermicompost":
    st.subheader("Bio-Organic Vermicompost Manure and Its Use in Farming")
    st.write("""Bio-organic vermicompost manure is a nutrient-rich organic fertilizer created through the decomposition of organic waste by earthworms, particularly the red wiggler (Eisenia fetida). The preparation involves layering kitchen scraps, dry leaves, and earthworms in a suitable container, ensuring proper moisture and aeration. This process typically takes 2 to 3 months, resulting in a dark, crumbly compost that is rich in essential nutrients like nitrogen, phosphorus, and potassium, as well as beneficial microorganisms that enhance soil health.""")
    
    st.image("vermicompostprepare.jpg", caption="Vermicomposting Process", use_column_width=True)

    st.write("""The benefits of vermicompost are manifold; it improves soil structure, increases water retention, and supports healthy root development. Additionally, it reduces the reliance on chemical fertilizers, making it an environmentally friendly option for gardeners and farmers. By repurposing organic waste, vermicomposting not only helps reduce landfill contributions but also promotes sustainable agricultural practices, contributing to a healthier ecosystem.""")
    
    # Links to YouTube videos
    st.markdown("[vermicompost prepare](https://youtu.be/lir2GkiyGyA?si=AFBUml3FnI1UsXlL)", unsafe_allow_html=True)
    st.markdown("[vermicompost ](https://youtu.be/Fc3mtdZzNOs?si=lzJb5Ue07Z2UY_gW)", unsafe_allow_html=True)
    st.markdown("[Vermicompost making](https://youtu.be/qo_pjygu3hU?si=6SiPyBJu1aW3zvwi)", unsafe_allow_html=True)





