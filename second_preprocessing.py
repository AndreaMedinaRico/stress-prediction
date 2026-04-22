import pandas as pd

# Console display adjustments
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 150)    

df = pd.read_csv('data/clean_dataset.csv')
print(df.head())
print(df.info())

df['sleep_deficit'] = 8 - df['sleep_duration_hours']
df['screen_sleep_ratio'] = df['daily_screen_time_hours'] / df['sleep_duration_hours']
df['fatigue_per_screen'] = df['mental_fatigue_score'] / df['daily_screen_time_hours']
df['activity_balance'] = df['physical_activity_hours'] - df['daily_screen_time_hours']

df.to_csv('data/engineered_dataset.csv', index=False)