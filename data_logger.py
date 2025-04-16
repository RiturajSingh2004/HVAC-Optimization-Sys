# data_logger.py
# Logs system data for analysis and visualization

import os
import pandas as pd

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