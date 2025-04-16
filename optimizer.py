# optimizer.py
# Processes environmental and occupancy data to optimize HVAC settings

import os
import numpy as np
import requests
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

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
            # For demonstration, we're using a text classification model
            # In production, use a model better suited for tabular data
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