import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv(r"C:\Users\abdul\PycharmProjects\NYC-taxi\data\train.csv")
    print(f"Raw shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst row:\n{df.head(1)}")
    return df

def remove_outliers(df):
    original_size = len(df)

    # Remove trips with irrational durations
    # Under 10 seconds — not a real trip
    # Over 24 hours —  bad data
    df = df[df["trip_duration"] >= 10]
    df = df[df["trip_duration"] <= 86400]

    # Removing trips where the taxi didn't move
    # Pickup and dropoff coordinates are identical
    df = df[
        (df["pickup_longitude"] != df["dropoff_longitude"]) |
        (df["pickup_latitude"] != df["dropoff_latitude"])
    ]

    # Removing coordinates outside NYC bounding box
    # Anything outside these bounds is bad GPS data
    df = df[
        (df["pickup_longitude"].between(-74.05, -73.75)) &
        (df["pickup_latitude"].between(40.63, 40.85)) &
        (df["dropoff_longitude"].between(-74.05, -73.75)) &
        (df["dropoff_latitude"].between(40.63, 40.85))
    ]

    # Removing trips with 0 passengers
    df = df[df["passenger_count"] > 0]

    # Removing trips with more than 6 passengers
    df = df[df["passenger_count"] <= 6]

    removed = original_size - len(df)
    print(f"\nOutliers removed: {removed:,} rows")
    print(f"Clean shape: {df.shape}")

    return df

def parse_dates(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df

def print_target_stats(df):
    print(f"\n=== TARGET VARIABLE: trip_duration ===")
    print(f"Mean:   {df['trip_duration'].mean():.0f} seconds ({df['trip_duration'].mean()/60:.1f} mins)")
    print(f"Median: {df['trip_duration'].median():.0f} seconds ({df['trip_duration'].median()/60:.1f} mins)")
    print(f"Min:    {df['trip_duration'].min():.0f} seconds")
    print(f"Max:    {df['trip_duration'].max():.0f} seconds ({df['trip_duration'].max()/3600:.1f} hrs)")

if __name__ == "__main__":
    df = load_data()
    df = remove_outliers(df)
    df = parse_dates(df)
    print_target_stats(df)
    print("\nPreprocessing complete.")