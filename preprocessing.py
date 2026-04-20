'''
File: preprocessing.py
Author: Andrea Medina
Description: This file performs data preprocessing necessary to 
    prepare the data for being an input to the model.
'''
from Transformation import Transformation
from Visualization import Visualization
import pandas as pd

# Console display adjustments
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 150)    

df = pd.read_csv('data/sleep_mobile_stress_dataset_15000.csv')
print(df.head())
print(df.info())

# ---- Data Preprocessing ----
trans = Transformation(df)

trans.drop_columns(['user_id'])
trans.drop_rows('gender', 'other')
trans.cat_to_num('gender', 'Male', 'Female')
trans.one_hot_encoding('occupation', 'Designer')

trans.minutes_to_hours(['phone_usage_before_sleep_minutes', 'physical_activity_minutes'])
trans.rename_columns({
    'phone_usage_before_sleep_minutes': 'phone_usage_before_sleep_hours',
    'physical_activity_minutes': 'physical_activity_hours'
})


df = trans.data

print("\nPreprocessed df:")
print(df.head())
print(df.info())

# ---- Data Visualization ----
vis = Visualization()
vis.correlation_matrix(df)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    vis.histogram(df, col)


# ---- Save clean data ----
df.to_csv('data/clean_dataset.csv', index=False)