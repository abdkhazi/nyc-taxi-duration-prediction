import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two GPS coordinates.
    Returns distance in kilometres.
    """
    R = 6371  # Earth's radius in kilometres

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def add_distance_features(df):
    print("Adding distance features...")

    # Straight line distance between pickup and dropoff
    df["haversine_distance"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )

    # Direction of travel in degrees
    df["direction"] = np.degrees(
        np.arctan2(
            df["dropoff_latitude"] - df["pickup_latitude"],
            df["dropoff_longitude"] - df["pickup_longitude"]
        )
    )

    # Absolute lat and long differences
    df["lat_diff"] = abs(df["dropoff_latitude"] - df["pickup_latitude"])
    df["lon_diff"] = abs(df["dropoff_longitude"] - df["pickup_longitude"])

    return df

def add_time_features(df):
    print("Adding time features...")

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_date"] = df["pickup_datetime"].dt.day

    # Is the trip during rush hour?
    # Morning rush: 7-10am, Evening rush: 4-8pm
    df["is_rush_hour"] = (
        (df["pickup_hour"].between(7, 10)) |
        (df["pickup_hour"].between(16, 20))
    ).astype(int)

    # Is the trip at night?
    df["is_night"] = (
        (df["pickup_hour"] >= 22) |
        (df["pickup_hour"] <= 6)
    ).astype(int)

    # Is the trip on a weekend?
    df["is_weekend"] = (df["pickup_day"] >= 5).astype(int)

    return df

def add_location_features(df):
    print("Adding location features...")

    # NYC airport coordinates
    JFK_LAT, JFK_LON = 40.6413, -73.7781
    LGA_LAT, LGA_LON = 40.7769, -73.8740
    EWR_LAT, EWR_LON = 40.6895, -74.1745

    # Distance from pickup to each airport
    df["pickup_dist_jfk"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        JFK_LAT, JFK_LON
    )
    df["pickup_dist_lga"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        LGA_LAT, LGA_LON
    )
    df["pickup_dist_ewr"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        EWR_LAT, EWR_LON
    )

    # Is the trip going to or from an airport?
    df["is_airport_trip"] = (
        (df["pickup_dist_jfk"] < 2) |
        (df["pickup_dist_lga"] < 2) |
        (df["pickup_dist_ewr"] < 2) |
        (haversine_distance(df["dropoff_latitude"], df["dropoff_longitude"], JFK_LAT, JFK_LON) < 2) |
        (haversine_distance(df["dropoff_latitude"], df["dropoff_longitude"], LGA_LAT, LGA_LON) < 2) |
        (haversine_distance(df["dropoff_latitude"], df["dropoff_longitude"], EWR_LAT, EWR_LON) < 2)
    ).astype(int)

    return df

def encode_categorical(df):
    print("Encoding categorical features...")

    # store_and_fwd_flag: N=0, Y=1
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0, "Y": 1})

    return df

def select_features(df):
    # Drop columns not useful for modelling
    drop_cols = ["id", "pickup_datetime", "dropoff_datetime"]
    df = df.drop(columns=drop_cols)
    return df

def build_features(df):
    df = add_distance_features(df)
    df = add_time_features(df)
    df = add_location_features(df)
    df = encode_categorical(df)
    df = select_features(df)

    print(f"\nFinal feature set: {df.shape[1] - 1} features")
    print(f"Features: {[c for c in df.columns if c != 'trip_duration']}")

    return df

if __name__ == "__main__":
    from preprocess import load_data, remove_outliers, parse_dates

    df = load_data()
    df = remove_outliers(df)
    df = parse_dates(df)
    df = build_features(df)

    print(f"\nSample row:\n{df.head(1)}")
    print("\nFeature engineering complete.")

