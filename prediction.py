'''
File: prediction.py
Author: Andrea Medina
Description: This is a user interface for the stress prediction model.
    It asks the user for input data, preprocesses it, and uses the 
    trained model to predict the stress level.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from classes.Transformation import Transformation

# Initial imports
model = load_model('models/model_improved.keras')
scaler = joblib.load('models/scaler.pkl')
features = joblib.load('models/feature_cols.pkl')

def get_user_input():
    user_data = {}

    print("\n\nHi! Welcome to the stress level predictor. Please provide the following information:\n")

    user_data["age"] = int(input("\nAge: "))
    user_data["gender"] = int(input("\nAssigned sex at birth:\n 0: Male\n 1: Female\n"))
    user_data["occupation"] = int(input("\nOccupation:\n 1: Designer\n 2: Teacher\n 3: Software Engineer\n 4: Manager\n" \
                                      "  5: Student\n 6: Freelancer\n 7: Doctor\n 8: Researcher\n"))
    
    user_data["daily_screen_time_hours"] = float(input("\nDaily screen time (hours): "))
    user_data["phone_usage_before_sleep_minutes"] = float(input("\nPhone usage before sleep (minutes): "))
    
    user_data["sleep_duration_hours"] = float(input("\nSleep duration (hours): "))
    user_data["sleep_quality_score"] = float(input("\nSleep quality (0-10): "))
    
    user_data["caffeine_intake_cups"] = float(input("\nCaffeine intake (cups per day): "))
    
    user_data["physical_activity_minutes"] = float(input("\nPhysical activity (minutes per day): "))
    user_data["notifications_received_per_day"] = int(input("\nNotifications received per day: "))
    
    user_data["mental_fatigue_score"] = float(input("\nMental fatigue (0-10): "))

    return user_data

def preprocess_input(user_data):
    df = pd.DataFrame([user_data])
    trans = Transformation(df)

    OCCUPATION_MAP = {
        1: "Designer",
        2: "Teacher",
        3: "Software Engineer",
        4: "Manager",
        5: "Student",
        6: "Freelancer",
        7: "Doctor",
        8: "Researcher"
    }

    # One-hot encoding for occupation
    trans.one_hot_encoding_map('occupation', OCCUPATION_MAP, 'Designer')

    # Convert minuts to hours
    trans.minutes_to_hours(['phone_usage_before_sleep_minutes', 'physical_activity_minutes'])
    trans.rename_columns({
    'phone_usage_before_sleep_minutes': 'phone_usage_before_sleep_hours',
    'physical_activity_minutes': 'physical_activity_hours'
    })

    # Ensure final dataframe has same columns as training data
    trans.ensure_same_columns(features)

    # Normalize data
    trans.normalize(scaler)

    final_df = trans.data
    return final_df


user_data = get_user_input()
preprocessed_data = preprocess_input(user_data)

prediction = model.predict(preprocessed_data)

stress_level = prediction[0][0]
print(f"\nPredicted stress level: {stress_level:.2f}")