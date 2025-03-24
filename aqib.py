import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

image = Image.open("says.jpeg")

st.image(image,width=1920)

# Load the trained model and encoders
model = pickle.load(open("best_model_linear.sav", "rb"))
scaler = pickle.load(open("scaler (1).sav", "rb"))
ohe1 = pickle.load(open("ohe1.sav", "rb"))
ohe3 = pickle.load(open("ohe3.sav", "rb"))
ohe4 = pickle.load(open("ohe4.sav", "rb"))
ohe5 = pickle.load(open("ohe5.sav", "rb"))
ohe6 = pickle.load(open("ohe6.sav", "rb"))
ohe7 = pickle.load(open("ohe7.sav", "rb"))
le = pickle.load(open("le.sav", "rb"))

st.title("Airline Ticket Price Prediction")

# User inputs
travel_class = st.selectbox("Travel Class", ["Economy", "Business"])
duration = st.number_input("Duration (in hours)", min_value=0.0, format="%.2f")
days_left = st.number_input("Days Left for Booking", min_value=0, step=1)
airline = st.selectbox("Airline", ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"])
source_city = st.selectbox("Source City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
destination_city = st.selectbox("Destination City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
departure_time = st.selectbox("Departure Time", ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"])
stops = st.selectbox("Number of Stops", ["zero", "one", "two_or_more"])
arrival_time = st.selectbox("Arrival Time", ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"])

# Function to handle unknown categories
def safe_ohe_transform(encoder, feature_name, value):
    categories = encoder.get_feature_names_out([feature_name])
    df = pd.DataFrame(encoder.transform([[value]]), columns=categories)
    return df.reindex(columns=categories, fill_value=0)

if st.button("Predict Price"):

    
        # Prepare input data
        input_data = pd.DataFrame({
            "duration": [duration],
            "days_left": [days_left]
        })
        
        # Apply One-Hot Encoding safely
        df_airline = safe_ohe_transform(ohe1, "airline", airline)
        df_source = safe_ohe_transform(ohe3, "source_city", source_city)
        df_destination = safe_ohe_transform(ohe7, "destination_city", destination_city)
        df_departure_time = safe_ohe_transform(ohe4, "departure_time", departure_time)
        df_stops = safe_ohe_transform(ohe5, "stops", stops)
        df_arrival_time = safe_ohe_transform(ohe6, "arrival_time", arrival_time)
        input_data['class'] = le.transform([travel_class])
        
        # Combine all input data
        input_data = pd.concat([input_data, df_airline, df_source, df_destination, df_departure_time, df_stops, df_arrival_time], axis=1)
        
        # Ensure column order matches the trained scaler
        expected_columns = scaler.feature_names_in_
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)  # Maintain correct order
        input_data_scaled = scaler.transform(input_data)
          # Make prediction
        price_prediction = model.predict(input_data_scaled)
        
        # Reverse log transformation
        price_prediction = np.expm1(price_prediction)    
            
        # Display result
        st.success(f"The predicted ticket price is ₹{price_prediction[0]:,.2f}")
    
 
    
   