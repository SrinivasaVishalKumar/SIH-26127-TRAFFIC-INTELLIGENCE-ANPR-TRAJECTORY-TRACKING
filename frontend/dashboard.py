import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="City-Wide ANPR Dashboard", layout="wide")

st.title("🚦 City-Wide ANPR & Vehicle Trajectory Tracking")
st.markdown("Real-time optical character recognition & spatial-temporal vehicle tracking.")

CSV_PATH = '../database/plate_log.csv'

# Load data function
def load_data():
    if os.path.exists(CSV_PATH) and os.stat(CSV_PATH).st_size > 0:
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(columns=["timestamp", "camera_id", "plate_number", "latitude", "longitude"])

data = load_data()

# Layout: 2 Columns (Sidebar/Stats & Main Panel)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Search Vehicle Trajectory")
    search_query = st.text_input("Enter License Plate Number (e.g. DL8C)", "").upper().strip()
    
    st.markdown("---")
    st.subheader("📊 Live Log Feed")
    if not data.empty:
        st.dataframe(data.tail(10), use_container_width=True)
    else:
        st.info("No vehicle logs found yet. Run the AI tracker to start capturing data.")

with col2:
    st.subheader("🗺️ Spatial Trajectory Map")
    
    # Initialize base map centered at default coordinates
    city_map = folium.Map(location=[17.3850, 78.4867], zoom_start=13)

    if search_query and not data.empty:
        # Filter by searched plate
        filtered_data = data[data['plate_number'].str.contains(search_query, na=False)]
        
        if not filtered_data.empty:
            st.success(f"Found {len(filtered_data)} sightings for vehicle **{search_query}**")
            
            points = []
            for _, row in filtered_data.iterrows():
                coord = [row['latitude'], row['longitude']]
                points.append(coord)
                folium.Marker(
                    location=coord,
                    popup=f"Camera: {row['camera_id']}<br>Time: {row['timestamp']}",
                    tooltip=f"{row['plate_number']} @ {row['timestamp']}",
                    icon=folium.Icon(color="red", icon="car", prefix="fa")
                ).add_to(city_map)
            
            # Connect the points to show the trajectory path
            if len(points) > 1:
                folium.PolyLine(points, color="blue", weight=3.5, opacity=0.8).add_to(city_map)
        else:
            st.warning(f"No trajectory records found for: {search_query}")
    else:
        # Plot all general camera locations
        if not data.empty:
            for _, row in data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"{row['plate_number']}",
                    color="green",
                    fill=True
                ).add_to(city_map)

    st_folium(city_map, width=800, height=500)