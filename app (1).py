import streamlit as st
import folium
import pandas as pd
import numpy as np
from streamlit_folium import folium_static, st_folium
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import re

# Load dataset
crime_data = pd.read_csv("eluru_bvrm_mtm.csv")  # Ensure correct file path

# Remove leading/trailing spaces from column names
crime_data.columns = crime_data.columns.str.strip()

# Standardize Crime Severity Column
crime_data["Crime_severity"] = crime_data["Crime_severity"].str.strip().str.lower()

# Function to clean latitude and longitude values
def clean_lat_lon(value):
    if isinstance(value, str):
        value = re.sub(r"[^\d.-]", "", value)  # Remove unwanted characters
    try:
        return float(value)
    except ValueError:
        return None  # Return None if conversion fails

# Apply cleaning function to Latitude & Longitude
crime_data["Latitude"] = crime_data["Latitude"].apply(clean_lat_lon)
crime_data["Longitude"] = crime_data["Longitude"].apply(clean_lat_lon)

# Drop rows where Latitude or Longitude could not be converted
crime_data = crime_data.dropna(subset=["Latitude", "Longitude"])

# Print data preview (for debugging)
print("Cleaned Data:", crime_data.head())

# Save cleaned data (optional)
crime_data.to_csv("cleaned_eluru_bvrm_mtm_data.csv", index=False)

# Convert severity to numerical values for clustering
severity_mapping = {"low": 1, "moderate": 2, "high": 3}
crime_data["Severity_Score"] = crime_data["Crime_severity"].map(severity_mapping)

# Apply DBSCAN clustering (adjusted eps value)
coords = crime_data[["Latitude", "Longitude"]].values

db = DBSCAN(eps=0.02, min_samples=2, metric="haversine").fit(np.radians(coords))  # Adjusted eps
crime_data["Cluster"] = db.labels_

# Function to analyze location-wise crime
def location_wise_analysis():
    st.title("📍 Crime Hotspots: Find Risk Level in Your Area")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=6)
    map_data = st_folium(m, height=500, width=700)

    if map_data and "last_clicked" in map_data:
        user_location = map_data["last_clicked"]
        user_lat, user_lon = user_location["lat"], user_location["lng"]
        st.success(f"✅ Selected Location: ({user_lat}, {user_lon})")

        # Identify crime hotspots near selected location
        nearby_hotspots = []
        for _, row in crime_data.iterrows():
            hotspot_lat, hotspot_lon = row["Latitude"], row["Longitude"]
            distance_km = geodesic((user_lat, user_lon), (hotspot_lat, hotspot_lon)).km

            if distance_km <= 5 and row["Crime_severity"] == "high":
                nearby_hotspots.append((row["Area Name"], hotspot_lat, hotspot_lon))

        st.write(f"Found {len(nearby_hotspots)} high-severity hotspots nearby.")

        if nearby_hotspots:
            st.subheader("🔥 High-Severity Crime Hotspots (within 5KM radius)")
            crime_map = folium.Map(location=[user_lat, user_lon], zoom_start=14)
            
            # Add user location
            folium.Marker(
                location=[user_lat, user_lon],
                popup="📍 Your Location",
                icon=folium.Icon(color="blue", icon="user")
            ).add_to(crime_map)
            
            # Add high-severity hotspots
            for city, lat, lon in nearby_hotspots:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.7,
                    popup=f"{city}: High Severity"
                ).add_to(crime_map)
            
            folium_static(crime_map)
        else:
            st.warning("⚠ No high-severity crime hotspots found within 5KM.")

location_wise_analysis()
