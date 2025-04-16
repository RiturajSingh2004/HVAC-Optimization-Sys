# sensing.py
# Collects environmental data from sensors and weather API

import numpy as np
import requests
from datetime import datetime

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