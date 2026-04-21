'''
File: prediction.py
Author: Andrea Medina
Description: This is a user interface for the stress prediction model.
    It asks the user for input data, preprocesses it, and uses the 
    trained model to predict the stress level.
'''

import numpy as np
from tensorflow import keras
from Transformation import Transformation

model = keras.models.load_model('model_baseline.keras')
trans = Transformation()

def get_user_input():
    user_data = {}

    print("Hi! Welcome to the stress level predictor. Please provide the following information:")

    user_data["age"] = int(input("Age: "))
    user_data["gender"] = int(input("Assigned sex at birth:\n 1: Male\n 2: Female"))
    user_data["occupation"] = int(input("Occupation:\n 1: Designer\n 2: Teacher\n 3: Software Engineer\n 4: Manager\n" \
                                      "  5: Student\n 6: Freelancer\n 7: Doctor\n 8: Researcher\n"))
    
    user_data["daily_screen_time_hours"] = float(input("Daily screen time (hours): "))
    user_data["phone_usage_before_sleep_minutes"] = float(input("Phone usage before sleep (minutes): "))
    
    user_data["sleep_duration_hours"] = float(input("Sleep duration (hours): "))
    user_data["sleep_quality_score"] = float(input("Sleep quality (0-10): "))
    
    user_data["caffeine_intake_cups"] = float(input("Caffeine intake (cups per day): "))
    
    user_data["physical_activity_minutes"] = float(input("Physical activity (minutes per day): "))
    user_data["notifications_received_per_day"] = int(input("Notifications received per day: "))
    
    user_data["mental_fatigue_score"] = float(input("Mental fatigue (0-10): "))

    return user_data

def preprocess_input(user_data):
    trans.minutes_to_hours(['phone_usage_before_sleep_minutes', 'physical_activity_minutes'])
    trans.rename_columns({
    'phone_usage_before_sleep_minutes': 'phone_usage_before_sleep_hours',
    'physical_activity_minutes': 'physical_activity_hours'
    })