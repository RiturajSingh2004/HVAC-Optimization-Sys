# dashboard.py
# Streamlit dashboard for the Smart HVAC System

import streamlit as st
from datetime import datetime
import base64
from pathlib import Path


def run_dashboard(system):
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Smart HVAC Dashboard",
        page_icon="🌡️",
        layout="wide"
    )
    
    # Dashboard title and description
    st.title("AI-Powered Smart HVAC Dashboard")
    st.markdown("""
    This dashboard monitors and controls an AI-powered HVAC system that optimizes
    energy usage based on occupancy and environmental conditions.
    """)
        
    # System control sidebar
    st.sidebar.title("System Control")
    if st.sidebar.button("Start System"):
        system.start()
        st.sidebar.success("System started!")
    
    if st.sidebar.button("Stop System"):
        system.stop()
        st.sidebar.error("System stopped!")
    
    if st.sidebar.button("Run Once"):
        system.run_once()
        st.sidebar.info("System ran one iteration")
    
    # Main dashboard content in columns
    col1, col2 = st.columns(2)
    
    # Current status in first column
    with col1:
        st.subheader("Current Status")
        status = system.get_system_status()
        
        # Environmental conditions
        st.markdown("### Environmental Conditions")
        env_data = status["environmental_data"]
        if env_data:
            col_env1, col_env2 = st.columns(2)
            with col_env1:
                st.metric("Indoor Temperature", f"{env_data.get('indoor_temperature', 'N/A')}°C")
                st.metric("Outdoor Temperature", f"{env_data.get('outdoor_temperature', 'N/A')}°C")
            with col_env2:
                st.metric("Indoor Humidity", f"{env_data.get('indoor_humidity', 'N/A')}%")
                st.metric("Outdoor Humidity", f"{env_data.get('outdoor_humidity', 'N/A')}%")
            st.text(f"Weather: {env_data.get('weather_conditions', 'N/A')}")
        else:
            st.info("No environmental data available yet.")
        
        # Occupancy info
        st.markdown("### Occupancy")
        st.metric("Current Occupancy", status["occupancy"])
        
        # Display occupancy detection frame if available
        if system.current_frame is not None:
            st.image(system.current_frame, caption="Occupancy Detection", use_column_width=True)
        
        # HVAC settings
        st.markdown("### HVAC Settings")
        hvac_settings = status["hvac_settings"]
        if hvac_settings:
            col_hvac1, col_hvac2 = st.columns(2)
            with col_hvac1:
                st.metric("Mode", hvac_settings.get("mode", "N/A").capitalize())
                st.metric("Target Temperature", f"{hvac_settings.get('target_temperature', 'N/A')}°C")
            with col_hvac2:
                st.metric("Fan Speed", hvac_settings.get("fan_speed", "N/A").capitalize())
                humidifier_status = "On" if hvac_settings.get("humidify", False) else "Off"
                dehumidifier_status = "On" if hvac_settings.get("dehumidify", False) else "Off"
                st.text(f"Humidifier: {humidifier_status}, Dehumidifier: {dehumidifier_status}")
        else:
            st.info("No HVAC settings available yet.")
    
    # Historical data in second column
    with col2:
        st.subheader("Historical Data")
        
        # Get recent log data
        log_data = system.data_logger.get_recent_logs(100)
        
        if not log_data.empty:
            # Plot temperature over time
            st.markdown("### Temperature Trends")
            temp_data = log_data.set_index("timestamp")
            st.line_chart(temp_data[["indoor_temperature", "outdoor_temperature", "target_temperature"]])
            
            # Plot occupancy over time
            st.markdown("### Occupancy Trends")
            occupancy_data = log_data.set_index("timestamp")[["occupancy"]]
            st.line_chart(occupancy_data)
            
            # Display recent log entries
            st.markdown("### Recent System Logs")
            st.dataframe(log_data.tail(10), use_container_width=True)
        else:
            st.info("No historical data available yet.")
    
    # Refresh dashboard
    st.button("Refresh Dashboard")

if __name__ == "__main__":
    # This allows running the dashboard standalone for development
    import sys
    import os
    
    # Add parent directory to path so we can import the SmartHVACSystem
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from main import SmartHVACSystem
    system = SmartHVACSystem()
    run_dashboard(system)
