# detection.py
# Handles people detection and counting using YOLOv5n

import cv2
import torch
import numpy as np

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