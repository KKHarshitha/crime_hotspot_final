import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static, st_folium
from geopy.distance import geodesic
import pickle
import math
import os
import numpy as np
from sklearn.cluster import DBSCAN

# Custom CSS for styling
st.markdown("""
    <style>
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #f0f2f6;
        color: #333;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-alert {
        color: green;
        font-weight: bold;
    }
    .warning-alert {
        color: orange;
        font-weight: bold;
    }
    .danger-alert {
        color: red;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load City Data
df = pd.read_pickle("cities.pkl")

# Load the Model
model_filename = "train_model.pkl"
if not os.path.exists(model_filename):
    st.error(f"Model file not found: {model_filename}. Please ensure the file exists in the same directory.")
    st.stop()

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the trained model
model = pickle.load(open('model (5).pkl', 'rb'))

# City and crime type mappings
city_names = {
    '0': 'Ahmedabad', '1': 'Bengaluru', '2': 'Chennai', '3': 'Coimbatore', '4': 'Delhi',
    '5': 'Ghaziabad', '6': 'Hyderabad', '7': 'Indore', '8': 'Jaipur', '9': 'Kanpur',
    '10': 'Kochi', '11': 'Kolkata', '12': 'Kozhikode', '13': 'Lucknow', '14': 'Mumbai',
    '15': 'Nagpur', '16': 'Patna', '17': 'Pune', '18': 'Surat'
}

crimes_names = {
    '0': 'Crime Committed by Juveniles', '1': 'Crime against SC', '2': 'Crime against ST',
    '3': 'Crime against Senior Citizen', '4': 'Crime against Children', '5': 'Crime against Women',
    '6': 'Cyber Crimes', '7': 'Economic Offences', '8': 'Kidnapping', '9': 'Murder'
}

population = {
    '0': 63.50, '1': 85.00, '2': 87.00, '3': 21.50, '4': 163.10, '5': 23.60, '6': 77.50,
    '7': 21.70, '8': 30.70, '9': 29.20, '10': 21.20, '11': 141.10, '12': 20.30, '13': 29.00,
    '14': 184.10, '15': 25.00, '16': 20.50, '17': 50.50, '18': 45.80
}

# Crime prevention suggestions
crime_suggestions = {
    '0': "Encourage educational programs and mentorship initiatives for youth.",
    '1': "Strengthen legal protection and create awareness about rights.",
    '2': "Promote inclusivity and ensure strict legal enforcement.",
    '3': "Enhance neighborhood watch programs and personal security for elders.",
    '4': "Increase child safety measures and strengthen family awareness.",
    '5': "Promote gender equality and enforce strict laws against offenders.",
    '6': "Use strong passwords, be cautious online, and report suspicious activities.",
    '7': "Be vigilant about financial frauds, verify sources before transactions.",
    '8': "Educate children about safety, avoid sharing personal details with strangers.",
    '9': "Improve community policing and strengthen law enforcement presence."
}

# Load Crime Data
@st.cache_data
def load_crime_data():
    with open('crime_data.pkl', 'rb') as file:
        return pickle.load(file)

crime_data = load_crime_data()
crime_data['state/ut'] = crime_data['state/ut'].str.title()
crime_data['district'] = crime_data['district'].str.title()

# Load Location Data
@st.cache_data
def load_location_data():
    return pd.read_pickle('state_district_lat_long.pkl')

location_data = load_location_data()
location_data['State'] = location_data['State'].str.title()
location_data['District'] = location_data['District'].str.title()

# Crime Severity Score Calculation
crime_weights = {
    'murder': 5,
    'rape': 4,
    'kidnapping & abduction': 4,
    'robbery': 3,
    'burglary': 3,
    'dowry deaths': 3
}

def calculate_crime_severity(df):
    weighted_sum = sum(df[col].sum() * weight for col, weight in crime_weights.items())
    max_possible = sum(500 * weight for weight in crime_weights.values())
    crime_index = (weighted_sum / max_possible) * 100 if max_possible > 0 else 0
    return round(crime_index, 2)

# Login Page


# City-wise Crime Analysis
def city_wise_analysis():
    st.title("🏙 City-wise Crime Analysis")
    
    # User inputs
    city_code = st.selectbox("🏙 Select City", options=list(city_names.keys()), format_func=lambda x: city_names[x])
    crime_code = st.selectbox("⚖ Select Crime Type", options=list(crimes_names.keys()), format_func=lambda x: crimes_names[x])
    year = st.number_input("📅 Enter Year", min_value=2024, max_value=2050, step=1, value=2024)

    if st.button("🔮 Predict Crime Rate"):
        # Fetch population data for the selected city
        pop = population.get(city_code, 0)
        
        # Adjust population based on the year (using compounded 1% annual growth)
        year_diff = year - 2015
        pop = pop * (1.01 ** year_diff)  # 1% yearly growth using exponential formula
        
        try:
            # Predict crime rate using the model
            crime_rate = model.predict([[int(year), int(city_code), pop, int(crime_code)]])[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Calculate estimated number of cases
        cases = math.ceil(crime_rate * pop)
        st.write(f"🔍 Debug: Predicted Cases = {cases}")

        # 🔴 Improved Crime Severity Categories Based on Predicted Cases
        if cases <= 50:
            crime_status = "🟢 Very Low Crime Area"
            color = "green"
        elif 51 <= cases <= 85:
            crime_status = "🟡 Low Crime Area"
            color = "yellow"
        elif 86 <= cases <= 120:
            crime_status = "🟠 Moderate Crime Area"
            color = "orange"
        elif 121 <= cases <= 170:
            crime_status = "🔴 High Crime Area"
            color = "red"
        else:
            crime_status = "🔥 Extremely High Crime Area"
            color = "darkred"
        
        # Display results with styling
        st.subheader("📊 Prediction Results")
        st.write(f"🏙 **City:** {city_names[city_code]}")
        st.write(f"⚖ **Crime Type:** {crimes_names[crime_code]}")
        st.write(f"📅 **Year:** {year}")
        st.write(f"👥 **Population:** {pop:.2f} Lakhs")
        st.write(f"🔍 Debug Before Classification: Predicted Cases = {cases}")
        st.markdown(f"<h3 style='color:{color};'>🚔 Predicted Cases: {cases}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:{color};'>⚠ Crime Severity: {crime_status}</h3>", unsafe_allow_html=True)

        # Display crime prevention suggestion
        st.markdown("### 💡 Safety Tip:")
        st.write(f"🛑 {crime_suggestions[crime_code]}")

        
# District-wise Crime Analysis
def district_wise_analysis():
    st.title("🌍 District-wise Crime Analysis")
    state = st.selectbox('Select a State/UT:', crime_data['state/ut'].unique())

    if state:
        # Filter data for the selected state
        state_data = crime_data[crime_data['state/ut'] == state]
        
        # Compute crime severity for each district
        district_severity = {}
        trend_data = {}  # To store crime severity trends for each district

        for district in state_data['district'].unique():
            district_data = state_data[state_data['district'] == district]
            
            # Calculate crime severity for 2024
            district_severity[district] = calculate_crime_severity(district_data[district_data['year'] == 2024])
            
            # Calculate crime severity for 2022, 2023, and 2024 (trend data)
            trend_data[district] = {
                year: calculate_crime_severity(district_data[district_data['year'] == year])
                for year in [2023, 2024]
            }
        
        # Display Crime Severity Map
        st.subheader(f'Crime Severity Index for Districts in {state}')
        
        state_location = location_data[location_data['State'] == state]
        if not state_location.empty:
            latitude, longitude = state_location.iloc[0]['Latitude'], state_location.iloc[0]['Longitude']
            m = folium.Map(location=[latitude, longitude], zoom_start=7)

            for district, severity in district_severity.items():
                district_row = location_data[(location_data['State'] == state) & (location_data['District'] == district)]
                if not district_row.empty:
                    lat, lon = district_row.iloc[0]['Latitude'], district_row.iloc[0]['Longitude']
                    color = 'green' if severity < 15 else 'orange' if severity < 25 else 'red'
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=10,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"{district}: {severity}"
                    ).add_to(m)
            
            folium_static(m)
        else:
            st.warning("Coordinates for the selected state were not found.")
        
        # Crime Severity Table
        st.subheader("Crime Severity Index by District")
        df_severity = pd.DataFrame(district_severity.items(), columns=['District', 'Crime Severity Index']).sort_values(by='Crime Severity Index', ascending=False)
        st.dataframe(df_severity)

        # Recommendations for selected district
        selected_district = st.selectbox("Select a District for Detailed Analysis:", list(district_severity.keys()))
        crime_severity_index = district_severity[selected_district]
        st.metric(label="Crime Severity Index (Higher is riskier)", value=crime_severity_index)
        
        # Display Crime Severity Trend
        st.subheader("Crime Severity Trend (2022 - 2024)")
        trend_df = pd.DataFrame(trend_data[selected_district], index=["Crime Severity Index"]).T
        st.line_chart(trend_df)
        
        if crime_severity_index < 10:
            st.markdown("<div class='success-alert'>🟢 This area is relatively safe.</div>", unsafe_allow_html=True)
        elif 11<= crime_severity_index <= 25:
            st.markdown("<div class='warning-alert'>🟠 Moderate risk; stay cautious.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='danger-alert'>🔴 High risk! Precaution is advised.</div>", unsafe_allow_html=True)
# Location-wise Crime Analysis

# Load dataset
import re

# Load and clean the location-specific crime dataset
location_crime_data = pd.read_pickle("eluru_bvrm_mtm.pkl")  # Using cleaned pickle file

# Remove leading/trailing spaces from column names
location_crime_data.columns = location_crime_data.columns.str.strip()

# Function to clean latitude and longitude
def clean_lat_lon(value):
    if isinstance(value, str):
        value = re.sub(r"[^\d.-]", "", value)
    try:
        return float(value)
    except ValueError:
        return None

location_crime_data["Latitude"] = location_crime_data["Latitude"].apply(clean_lat_lon)
location_crime_data["Longitude"] = location_crime_data["Longitude"].apply(clean_lat_lon)

# Drop invalid rows
location_crime_data = location_crime_data.dropna(subset=["Latitude", "Longitude"])

# Convert severity to numerical for clustering
severity_mapping = {"low": 1, "moderate": 2, "high": 3}
location_crime_data["Severity_Score"] = location_crime_data["Crime_severity"].map(severity_mapping)

# Apply DBSCAN
coords = location_crime_data[["Latitude", "Longitude"]].values
db = DBSCAN(eps=10 / 6371, min_samples=2, metric="haversine").fit(np.radians(coords))
location_crime_data["Cluster"] = db.labels_

# Location-wise crime analysis function
   
def location_wise_analysis():
    st.title("📍 Crime Hotspots: Find Risk Level in Your Area")

    # Base map centered around Eluru-BVRM-MTM
    base_map = folium.Map(location=[16.7100, 81.0950], zoom_start=10)
    map_data = st_folium(base_map, height=500, width=700)

    if map_data and "last_clicked" in map_data:
        user_lat = map_data["last_clicked"]["lat"]
        user_lon = map_data["last_clicked"]["lng"]
        st.success(f"✅ Selected Location: ({user_lat:.4f}, {user_lon:.4f})")

        # Find nearby high-severity and moderate-severity hotspots
        high_severity_hotspots = []
        moderate_severity_hotspots = []

        for _, row in location_crime_data.iterrows():
            hotspot_lat, hotspot_lon = row["Latitude"], row["Longitude"]
            if pd.notnull(hotspot_lat) and pd.notnull(hotspot_lon):
                distance_km = geodesic((user_lat, user_lon), (hotspot_lat, hotspot_lon)).km
                if distance_km <= 5:
                    if row["Crime_severity"] == "High":
                        high_severity_hotspots.append((row["Area Name"], hotspot_lat, hotspot_lon))
                    elif row["Crime_severity"] == "Moderate":
                        moderate_severity_hotspots.append((row["Area Name"], hotspot_lat, hotspot_lon))

        if high_severity_hotspots or moderate_severity_hotspots:
            st.subheader("🔥 Crime Hotspots within 5KM")
            hotspot_map = folium.Map(location=[user_lat, user_lon], zoom_start=14)

            # Mark user location
            folium.Marker(
                location=[user_lat, user_lon],
                popup="📍 Your Location",
                icon=folium.Icon(color="blue", icon="user")
            ).add_to(hotspot_map)

            # Mark high-severity hotspots (Red)
            for area, lat, lon in high_severity_hotspots:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.7,
                    popup=f"{area}: High Severity"
                ).add_to(hotspot_map)

            # Mark moderate-severity hotspots (Yellow)
            for area, lat, lon in moderate_severity_hotspots:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    color="yellow",
                    fill=True,
                    fill_color="yellow",
                    fill_opacity=0.7,
                    popup=f"{area}: Moderate Severity"
                ).add_to(hotspot_map)

            folium_static(hotspot_map)
        else:
            st.warning("✅ No high or moderate-severity crime hotspots found within 5KM.")



# Main App Logic
def main():

    option = st.sidebar.radio("Choose an Analysis:", ["City-wise Crime Analysis", "District-wise Crime Analysis", "Location-wise Crime Analysis"])
        
    if option == "City-wise Crime Analysis":
        city_wise_analysis()
    elif option == "District-wise Crime Analysis":
        district_wise_analysis()
    elif option == "Location-wise Crime Analysis":
        location_wise_analysis()

if __name__ == "__main__":
    main()