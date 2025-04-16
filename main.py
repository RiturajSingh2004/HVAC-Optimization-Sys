# main.py
# Main application that integrates all components of the system

import os
import time
import threading
from datetime import datetime
import argparse

# Import system components
from config import CONFIG
from detection import OccupancyDetector
from sensing import EnvironmentalDataCollector
from optimizer import HVACOptimizer
from data_logger import DataLogger
from dashboard import run_dashboard

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


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Smart HVAC System")
    parser.add_argument("--dashboard", action="store_true", help="Run the Streamlit dashboard")
    parser.add_argument("--headless", action="store_true", help="Run the system without dashboard")
    
    args = parser.parse_args()
    
    # Create the system
    system = SmartHVACSystem()
    
    if args.dashboard:
        # Import here to avoid circular imports
        import streamlit
        from dashboard import run_dashboard
        run_dashboard(system)
    elif args.headless:
        system.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping system...")
            system.stop()
    else:
        print("Please specify either --dashboard or --headless mode")
        print("For dashboard: python main.py --dashboard")
        print("For headless operation: python main.py --headless")


if __name__ == "__main__":
    main()