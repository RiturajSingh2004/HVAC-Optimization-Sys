# config.py
# Configuration settings for the Smart HVAC system

# System configuration
CONFIG = {
    "video_source": 0,  # 0 for webcam, or path to video file
    "detection_interval": 5,  # seconds between occupancy detection runs
    "weather_api_key": "YOUR_API_KEY",  # OpenWeatherMap API key
    "location": {
        "city": "New York",
        "country": "US"
    },
    "indoor_temp_sensor": None,  # Set to None if using simulated data
    "indoor_humidity_sensor": None,  # Set to None if using simulated data
    "model_path": "hvac_model",  # Path to save/load model
    "hvac_control_api": "http://localhost:8000/api/hvac/control",  # Example HVAC control API endpoint
    "simulation_mode": True  # Set to True for simulated sensors and HVAC control
}