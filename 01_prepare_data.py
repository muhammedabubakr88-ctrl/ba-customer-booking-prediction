"""
Task 2 - Step 1: Data preparation and feature engineering
"""
import pandas as pd
import numpy as np

# Load
df = pd.read_csv('data/customer_booking.csv', encoding='latin-1')
print("Original shape:", df.shape)

# ---- Feature engineering ----

# 1. Total extras wanted (intent/engagement signal)
df['total_extras_wanted'] = (
    df['wants_extra_baggage'] + df['wants_preferred_seat'] + df['wants_in_flight_meals']
)

# 2. Weekend flight flag
df['is_weekend_flight'] = df['flight_day'].isin(['Sat', 'Sun']).astype(int)

# 3. Frequency encode high-cardinality categoricals (route, booking_origin)
route_freq = df['route'].value_counts(normalize=True)
df['route_frequency'] = df['route'].map(route_freq)

origin_freq = df['booking_origin'].value_counts(normalize=True)
df['booking_origin_frequency'] = df['booking_origin'].map(origin_freq)

# 4. One-hot encode low-cardinality categoricals
df = pd.get_dummies(df, columns=['sales_channel', 'trip_type', 'flight_day'], drop_first=True)

# Drop original high-cardinality text columns (now represented by frequency features)
df = df.drop(columns=['route', 'booking_origin'])

print("Final shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

df.to_csv('outputs/prepared_data.csv', index=False)
print("\nSaved outputs/prepared_data.csv")