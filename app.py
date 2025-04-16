# Smart HVAC Optimization System
# Main application integrating all components

import os
import time
import cv2
import numpy as np
import torch
import pandas as pd
import requests
import json
import threading
import streamlit as st
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
from typing import Dict, List, Tuple, Union, Optional

# Configuration
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

class OccupancyDetector:
    """Handles people detection and counting using YOLOv5n."""
    
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load YOLOv5n model."""
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        return model
    
    def detect_people(self, frame):
        """Detect people in the given frame."""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = self.model(frame_rgb)
        
        # Filter for people (class 0 in COCO dataset)
        people_detections = results.xyxy[0][results.xyxy[0][:, 5] == 0]
        
        # Count people
        people_count = len(people_detections)
        
        # Visualize detections
        annotated_frame = results.render()[0]
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        return people_count, annotated_frame
    
    def get_occupancy(self):
        """Capture frame from video source and detect people."""
        cap = cv2.VideoCapture(self.video_source)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return 0, None
        
        people_count, annotated_frame = self.detect_people(frame)
        return people_count, annotated_frame


class EnvironmentalDataCollector:
    """Collects environmental data from sensors and weather API."""
    
    def __init__(self, config):
        self.config = config
        self.weather_api_key = config["weather_api_key"]
        self.city = config["location"]["city"]
        self.country = config["location"]["country"]
        self.indoor_temp_sensor = config["indoor_temp_sensor"]
        self.indoor_humidity_sensor = config["indoor_humidity_sensor"]
        self.simulation_mode = config["simulation_mode"]
        
    def get_outdoor_weather(self):
        """Fetch weather data from OpenWeatherMap API."""
        if self.simulation_mode:
            # Simulate weather data
            return {
                "temperature": round(np.random.uniform(10, 35), 1),
                "humidity": round(np.random.uniform(30, 90), 1),
                "conditions": np.random.choice(["Clear", "Cloudy", "Rain", "Snow"]),
                "wind_speed": round(np.random.uniform(0, 20), 1)
            }
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "conditions": data["weather"][0]["main"],
                    "wind_speed": data["wind"]["speed"]
                }
            else:
                print(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
                # Return simulated data as fallback
                return self.get_simulated_weather()
        except Exception as e:
            print(f"Exception when fetching weather data: {e}")
            # Return simulated data as fallback
            return self.get_simulated_weather()
    
    def get_simulated_weather(self):
        """Generate simulated weather data."""
        return {
            "temperature": round(np.random.uniform(10, 35), 1),
            "humidity": round(np.random.uniform(30, 90), 1),
            "conditions": np.random.choice(["Clear", "Cloudy", "Rain", "Snow"]),
            "wind_speed": round(np.random.uniform(0, 20), 1)
        }
    
    def get_indoor_conditions(self):
        """Read data from indoor sensors or simulate if not available."""
        if self.simulation_mode or not (self.indoor_temp_sensor and self.indoor_humidity_sensor):
            # Simulate indoor conditions
            return {
                "temperature": round(np.random.uniform(18, 28), 1),
                "humidity": round(np.random.uniform(30, 70), 1)
            }
        
        # Here you would add code to read from actual sensors
        # This is placeholder code
        try:
            temperature = self.indoor_temp_sensor.read_temperature()
            humidity = self.indoor_humidity_sensor.read_humidity()
            return {"temperature": temperature, "humidity": humidity}
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            # Return simulated data as fallback
            return {
                "temperature": round(np.random.uniform(18, 28), 1),
                "humidity": round(np.random.uniform(30, 70), 1)
            }
    
    def get_all_environmental_data(self):
        """Collect all environmental data."""
        outdoor = self.get_outdoor_weather()
        indoor = self.get_indoor_conditions()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "timestamp": timestamp,
            "indoor_temperature": indoor["temperature"],
            "indoor_humidity": indoor["humidity"],
            "outdoor_temperature": outdoor["temperature"],
            "outdoor_humidity": outdoor["humidity"],
            "weather_conditions": outdoor["conditions"],
            "wind_speed": outdoor["wind_speed"]
        }


class HVACOptimizer:
    """Processes environmental and occupancy data to optimize HVAC settings using a pre-trained model."""
    
    def __init__(self, config):
        self.config = config
        self.hvac_control_api = config["hvac_control_api"]
        self.simulation_mode = config["simulation_mode"]
        self.model = self.load_pretrained_model()
        
    def load_pretrained_model(self):
        """Load a pre-trained model for HVAC optimization."""
        try:
            # Use a pre-trained tabular model from Hugging Face
            # TabNet is a good candidate for tabular data
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            
            # Here we're using DistilBERT as an example, but for tabular data,
            # a dedicated tabular model would be more appropriate
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            
            # Create a pipeline for easier inference
            classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
            
            print("Pre-trained model loaded successfully")
            return classifier
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Falling back to rule-based approach")
            return None
    
    def preprocess_data(self, env_data, occupancy):
        """Prepare data for model inference."""
        # Format the environmental data and occupancy as a text description
        # This approach works with transformer models that expect text input
        
        time_of_day = datetime.now().hour
        day_period = "morning" if 5 <= time_of_day < 12 else "afternoon" if 12 <= time_of_day < 17 else "evening" if 17 <= time_of_day < 22 else "night"
        
        description = (
            f"Room with {occupancy} people. "
            f"Indoor temperature is {env_data['indoor_temperature']}°C with {env_data['indoor_humidity']}% humidity. "
            f"Outdoor temperature is {env_data['outdoor_temperature']}°C with {env_data['outdoor_humidity']}% humidity. "
            f"Weather conditions are {env_data['weather_conditions']}. "
            f"It is {day_period} time."
        )
        
        return description
    
    def determine_optimal_settings(self, env_data, occupancy):
        """Determine optimal HVAC settings based on current conditions."""
        if self.model is None:
            # Fall back to rule-based approach if model loading failed
            return self.rule_based_settings(env_data, occupancy)
        
        try:
            # Preprocess data for the model
            input_data = self.preprocess_data(env_data, occupancy)
            
            # For demonstration purposes, we'll use the model but still interpret the results ourselves
            # In a production system, you'd want the model to directly output HVAC settings
            result = self.model(input_data)
            
            # Here we're using a text classification model as an example
            # The model isn't specifically trained for HVAC control, so we're interpreting its output
            
            # Extract the predicted class or score
            # This is a placeholder - in reality you'd use a model that outputs HVAC settings directly
            label = result[0]['label']
            score = result[0]['score']
            
            print(f"Model prediction: {label} with confidence {score:.2f}")
            
            # Translate the model output to HVAC settings
            # This is highly simplified and would be replaced with proper model output interpretation
            if "LABEL_0" in label:
                mode = "cool"
            elif "LABEL_1" in label:
                mode = "heat"
            else:
                mode = "fan"
                
            # Combine model prediction with rule-based approach
            base_settings = self.rule_based_settings(env_data, occupancy)
            base_settings["mode"] = mode
            
            return base_settings
            
        except Exception as e:
            print(f"Error using pre-trained model for inference: {e}")
            print("Falling back to rule-based approach")
            return self.rule_based_settings(env_data, occupancy)
    
    def rule_based_settings(self, env_data, occupancy):
        """Rule-based fallback for determining HVAC settings."""
        indoor_temp = env_data["indoor_temperature"]
        outdoor_temp = env_data["outdoor_temperature"]
        indoor_humidity = env_data["indoor_humidity"]
        
        # Target temperature based on occupancy
        if occupancy == 0:
            # No one in the room - energy saving mode
            target_temp = 24 if outdoor_temp > 24 else 20
            fan_speed = "low"
        elif occupancy <= 3:
            # Few people - comfortable temperature
            target_temp = 22
            fan_speed = "medium"
        else:
            # Many people - cooler temperature
            target_temp = 21
            fan_speed = "high"
        
        # Adjust for extreme outdoor conditions
        if outdoor_temp > 30:
            target_temp -= 1
        elif outdoor_temp < 5:
            target_temp += 1
            
        # Determine if heating or cooling is needed
        if indoor_temp < target_temp - 1:
            mode = "heat"
        elif indoor_temp > target_temp + 1:
            mode = "cool"
        else:
            mode = "fan"  # Maintain current temperature
            
        # Configure humidity control if needed
        if indoor_humidity > 70:
            dehumidify = True
        elif indoor_humidity < 30:
            humidify = True
        else:
            humidify = False
            dehumidify = False
            
        return {
            "mode": mode,
            "target_temperature": target_temp,
            "fan_speed": fan_speed,
            "humidify": humidify,
            "dehumidify": dehumidify
        }
    
    def apply_hvac_settings(self, settings):
        """Send commands to the HVAC system."""
        if self.simulation_mode:
            print(f"Simulated HVAC control: {settings}")
            return True
        
        try:
            # Send settings to HVAC control API
            response = requests.post(
                self.hvac_control_api,
                json=settings,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print("HVAC settings applied successfully")
                return True
            else:
                print(f"Failed to apply HVAC settings: {response.text}")
                return False
        except Exception as e:
            print(f"Error applying HVAC settings: {e}")
            return False


class DataLogger:
    """Logs system data for analysis and visualization."""
    
    def __init__(self, log_file="hvac_system_log.csv"):
        self.log_file = log_file
        self.initialize_log()
        
    def initialize_log(self):
        """Create or check log file with headers."""
        if not os.path.exists(self.log_file):
            headers = [
                "timestamp", 
                "occupancy", 
                "indoor_temperature", 
                "indoor_humidity", 
                "outdoor_temperature", 
                "outdoor_humidity", 
                "weather_conditions", 
                "hvac_mode", 
                "target_temperature", 
                "fan_speed"
            ]
            with open(self.log_file, "w") as f:
                f.write(",".join(headers) + "\n")
    
    def log_data(self, data):
        """Log a data point to the CSV file."""
        row = [
            data["timestamp"],
            data["occupancy"],
            data["indoor_temperature"],
            data["indoor_humidity"],
            data["outdoor_temperature"],
            data["outdoor_humidity"],
            data["weather_conditions"],
            data["hvac_settings"]["mode"],
            data["hvac_settings"]["target_temperature"],
            data["hvac_settings"]["fan_speed"]
        ]
        
        with open(self.log_file, "a") as f:
            f.write(",".join(map(str, row)) + "\n")
    
    def get_recent_logs(self, n=100):
        """Retrieve the most recent log entries."""
        try:
            df = pd.read_csv(self.log_file)
            return df.tail(n)
        except Exception as e:
            print(f"Error reading log file: {e}")
            return pd.DataFrame()


class SmartHVACSystem:
    """Main class that integrates all components of the system."""
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.occupancy_detector = OccupancyDetector(video_source=config["video_source"])
        self.env_data_collector = EnvironmentalDataCollector(config)
        self.hvac_optimizer = HVACOptimizer(config)
        self.data_logger = DataLogger()
        self.current_frame = None
        self.current_occupancy = 0
        self.current_env_data = {}
        self.current_hvac_settings = {}
        self.running = False
        
    def update_occupancy(self):
        """Update occupancy count."""
        occupancy, frame = self.occupancy_detector.get_occupancy()
        self.current_occupancy = occupancy
        self.current_frame = frame
        return occupancy
    
    def update_environmental_data(self):
        """Update environmental sensor data."""
        self.current_env_data = self.env_data_collector.get_all_environmental_data()
        return self.current_env_data
    
    def update_hvac_settings(self):
        """Update HVAC settings based on current conditions."""
        settings = self.hvac_optimizer.determine_optimal_settings(
            self.current_env_data, 
            self.current_occupancy
        )
        self.current_hvac_settings = settings
        self.hvac_optimizer.apply_hvac_settings(settings)
        return settings
    
    def log_current_state(self):
        """Log the current system state."""
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "occupancy": self.current_occupancy,
            "indoor_temperature": self.current_env_data.get("indoor_temperature"),
            "indoor_humidity": self.current_env_data.get("indoor_humidity"),
            "outdoor_temperature": self.current_env_data.get("outdoor_temperature"),
            "outdoor_humidity": self.current_env_data.get("outdoor_humidity"),
            "weather_conditions": self.current_env_data.get("weather_conditions"),
            "hvac_settings": self.current_hvac_settings
        }
        self.data_logger.log_data(log_data)
        return log_data
    
    def run_once(self):
        """Run one iteration of the system."""
        self.update_occupancy()
        self.update_environmental_data()
        self.update_hvac_settings()
        return self.log_current_state()
    
    def run_continuous(self):
        """Run the system continuously in the background."""
        self.running = True
        
        while self.running:
            try:
                self.run_once()
                time.sleep(self.config["detection_interval"])
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(5)  # Sleep and retry on error
    
    def start(self):
        """Start the system in a background thread."""
        self.thread = threading.Thread(target=self.run_continuous)
        self.thread.daemon = True
        self.thread.start()
        print("Smart HVAC system started")
    
    def stop(self):
        """Stop the system."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5)
        print("Smart HVAC system stopped")
    
    def get_system_status(self):
        """Get the current status of the system."""
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "running": self.running,
            "occupancy": self.current_occupancy,
            "environmental_data": self.current_env_data,
            "hvac_settings": self.current_hvac_settings
        }


# Dashboard Application
def run_dashboard():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Smart HVAC Dashboard",
        page_icon="🌡️",
        layout="wide"
    )
    
    # Initialize system
    system = SmartHVACSystem()
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart HVAC System")
    parser.add_argument("--dashboard", action="store_true", help="Run the Streamlit dashboard")
    parser.add_argument("--headless", action="store_true", help="Run the system without dashboard")
    
    args = parser.parse_args()
    
    if args.dashboard:
        run_dashboard()
    elif args.headless:
        system = SmartHVACSystem()
        system.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping system...")
            system.stop()
    else:
        print("Please specify either --dashboard or --headless mode")
        print("For dashboard: python app.py --dashboard")
        print("For headless operation: python app.py --headless")