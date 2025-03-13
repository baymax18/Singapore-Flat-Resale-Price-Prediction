import streamlit as st
import numpy as np
import pandas as pd
import pickle
import locale
import warnings
import re

def format_singapore_money(amount):
    
    return "{:,.2f}".format(amount)


st.set_page_config(
    page_title="Singapore Flat reslae price prediction",
    page_icon="🏠",
    layout="wide"
)

st.title (':violet[Singapore Flat Reslae Price Prediction using Machine Learning Model]')


tab1, tab2= st.tabs(['About', 'Price Prediction'])

with tab1:
    st.markdown("""
    <h2 style='font-size: 30px;'>Introduction </h2>
    <p style='font-size: 20px;'>In recent years, Singapore’s real estate market has seen significant fluctuations in flat prices, driven by a wide range of factors such as Location, flat model, remaining lease, floor area, and other socioeconomic trends. 
    Accurately predicting the resale price of flats can be a daunting task for buyers and sellers alike. This project aims to build a machine learning model that can accurately predict the resale price of flats in Singapore, providing a powerful tool for both buyers and sellers to evaluate the potential worth of a property.</p>
    
    <p style='font-size: 20px;'>The <strong>Singapore Flat Resale Price Prediction</strong> model uses historical data on past flat transactions to train a machine learning algorithm. By analyzing various features of each transaction, the model learns to predict the resale price based on key factors such as:</p>
    
    <ul style='font-size: 20px;'>
        <li><strong>Location</strong>: The town which the flat is located.</li>
        <li><strong>Flat Model</strong>: The type or design of the flat (e.g., 4-room, 5-room, etc.).</li>
        <li><strong>Remaining Lease</strong>: The number of years remaining on the flat's lease.</li>
        <li><strong>Floor Area</strong>: The total floor area of the flat in square meters.</li>
        <li><strong>Storey Range</strong>: The range of floors the flat is situated on (e.g., high floor, low floor).</li>
    </ul>
    
    <p style='font-size: 20px;'>The goal of this predictive model is to help users estimate the resale price of a flat using these criteria, aiding them in making informed decisions in the buying or selling process. By leveraging machine learning algorithms, the model can uncover complex relationships between these features and predict the resale value with greater accuracy than traditional methods.</p>

    <h3 style='font-size: 25px;'>Key Benefits:</h3>
    <ul style='font-size: 20px;'>
        <li><strong>For Buyers:</strong> The model helps potential buyers evaluate whether a flat is reasonably priced based on its features and current market trends.</li>
        <li><strong>For Sellers:</strong> Sellers can use the model to set a competitive yet realistic price for their flat, maximizing their chances of a successful sale.</li>
        <li><strong>For Real Estate Agencies:</strong> Real estate agents can use the model to provide value estimates to clients, improving their efficiency and decision-making process.</li>
    </ul>
    
    <p style='font-size: 20px;'>The machine learning model will be trained using a dataset of historical resale transactions, where each record contains the key features that influence flat prices. After training, the model will be able to predict the resale price of any given flat based on its characteristics.</p>

    <h3 style='font-size: 25px;'>Why Machine Learning?</h3>
    <p style='font-size: 20px;'>Traditional methods of property valuation often rely on human expertise and can be influenced by biases or outdated market conditions. Machine learning, on the other hand, allows us to model complex, nonlinear relationships between flat features and price, providing more accurate and data-driven predictions. This makes it an ideal approach for predicting flat prices in a dynamic and competitive market like Singapore’s.</p>

    <h3 style='font-size: 25px;'>How to Use This Application:</h3>
    <p style='font-size: 20px;'>This web application provides an intuitive interface where users can input details about a flat, such as its location, floor area, flat model, remaining lease, and storey range. The machine learning model will then predict the resale price based on the provided information. The application aims to be user-friendly, with predictions provided almost instantly after entering the necessary details.</p>
    """, unsafe_allow_html=True)


            

with tab2:

    flat_categories = ['Improved', 'New Generation', 'Model A', 'Premium Apartment',
    'Standard', 'Simplified', 'Model A2', 'Type S1', 'DBSS', 'Terrace',
    'Premium Apartment Loft', '2-room']

    flat_model_map = {
            'Improved': np.int64(2),
            'New Generation': np.int64(5),
            'Model A': np.int64(3),
            'Premium Apartment': np.int64(6),
            'Standard': np.int64(9),
            'Simplified': np.int64(8),
            'Model A2': np.int64(4),
            'Type S1': np.int64(11),
            'DBSS': np.int64(1),
            'Terrace': np.int64(10),
            'Premium Apartment Loft': np.int64(7),
            '2-room': np.int64(0)}
    Town_list = ['ANG MO KIO',
                    'BEDOK',
                    'BISHAN',
                    'BUKIT BATOK',
                    'BUKIT MERAH',
                    'BUKIT PANJANG',
                    'BUKIT TIMAH',
                    'CENTRAL AREA',
                    'CHOA CHU KANG',
                    'CLEMENTI',
                    'GEYLANG',
                    'HOUGANG',
                    'JURONG EAST',
                    'JURONG WEST',
                    'KALLANG/WHAMPOA',
                    'MARINE PARADE',
                    'PASIR RIS',
                    'PUNGGOL',
                    'QUEENSTOWN',
                    'SEMBAWANG',
                    'SENGKANG',
                    'SERANGOON',
                    'TAMPINES',
                    'TOA PAYOH',
                    'WOODLANDS',
                    'YISHUN']
    town_map = {
            'ANG MO KIO': np.int64(0),
            'BEDOK': np.int64(1),
            'BISHAN': np.int64(2),
            'BUKIT BATOK': np.int64(3),
            'BUKIT MERAH': np.int64(4),
            'BUKIT PANJANG': np.int64(5),
            'BUKIT TIMAH': np.int64(6),
            'CENTRAL AREA': np.int64(7),
            'CHOA CHU KANG': np.int64(8),
            'CLEMENTI': np.int64(9),
            'GEYLANG': np.int64(10),
            'HOUGANG': np.int64(11),
            'JURONG EAST': np.int64(12),
            'JURONG WEST': np.int64(13),
            'KALLANG/WHAMPOA': np.int64(14),
            'MARINE PARADE': np.int64(15),
            'PASIR RIS': np.int64(16),
            'PUNGGOL': np.int64(17),
            'QUEENSTOWN': np.int64(18),
            'SEMBAWANG': np.int64(19),
            'SENGKANG': np.int64(20),
            'SERANGOON': np.int64(21),
            'TAMPINES': np.int64(22),
            'TOA PAYOH': np.int64(23),
            'WOODLANDS': np.int64(24),
            'YISHUN': np.int64(25)}


    col1,col2 = st.columns([3,1], gap='small')
    with col1:
        st.markdown("# :violet[Predicting Price based on Trained Model]")
        
        
        a1 = st.selectbox("Select Town",Town_list )
        a1 = town_map[a1]
        b1 = st.text_input("Remaining Lease in Years", placeholder="Value should be between 1 and 99")
        c1 = st.text_input("Floor Area in sqm")
        d1 = st.text_input("Storey Range")
        e1 = st.selectbox("Select Flat Model", flat_categories)
        e1 = flat_model_map[e1]

        
        with open(r"regression_model.pkl", 'rb') as file_1:
            regression_model = pickle.load(file_1)

        
        predict_button = st.button(":green[Predict ReSelling Price]")

        if predict_button:
            try:
            
                a1 = float(a1)  # Town Encoded
                b1 = float(b1)  # Remaining Lease
                c1 = float(c1)  # Floor Area
                d1 = float(d1)  # Storey Range
                e1 = float(e1)  # Flat Model Encoded

            
                new_sample_1 = np.array([[a1, b1, c1, np.log(d1), e1]])

            
                new_pred_1 = regression_model.predict(new_sample_1)[0]

                formatted_price = format_singapore_money(np.exp(new_pred_1))
                
                st.write(f"## :green[Predicted Resale Price in SGD: S$ {formatted_price}]")

            except ValueError:
                
                st.write(f"## :red[Error: Please make sure all inputs are valid numbers.]")

    


    
