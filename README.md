# NYC Taxi Trip Duration Prediction

Predicting taxi trip duration in New York City using machine learning.
Built as a portfolio project targeting commercial ML and LLM roles in London.

## Problem
Given pickup/dropoff coordinates, time of day, and passenger information,
predict how long a NYC taxi trip will take. Accurate duration prediction
improves passenger experience and fleet efficiency.

## Dataset
NYC Taxi Trip Duration — Kaggle competition dataset (2016).
1,458,644 raw trips · 1,433,177 after outlier removal.
Download: https://www.kaggle.com/competitions/nyc-taxi-trip-duration

## Feature Engineering
22 features engineered from raw GPS coordinates and timestamps:

- **Haversine distance** — straight-line distance between pickup and dropoff (km)
- **Direction of travel** — bearing in degrees from pickup to dropoff
- **Time features** — hour, day, month, rush hour flag, night flag, weekend flag
- **Airport proximity** — distance to JFK, LGA, and EWR; airport trip flag
- **Coordinate differences** — absolute lat/lon differences

## Models and Results

| Model              | RMSLE  | MAE          |
|--------------------|--------|--------------|
| Linear Regression  | 0.5417 | 7.1 minutes  |
| Random Forest      | 0.3760 | 4.8 minutes  |
| XGBoost            | 0.3669 | 4.7 minutes  |

XGBoost predictions are off by an average of **4.7 minutes** on unseen data.
Linear Regression underperforms significantly, confirming the non-linear
relationships between distance, time, and trip duration.

## Key Insights (SHAP analysis)

| Driver               | Finding |
|----------------------|---------|
| Haversine distance   | Single strongest predictor — nearly 4x more important than any other feature |
| Pickup hour          | Time of day has strong effect — rush hour trips significantly longer |
| Direction of travel  | East-west crosstown trips slower than north-south — reflects NYC grid |
| JFK proximity        | Airport trips show distinct duration patterns — feature engineering validated |
| Night flag           | Night trips measurably faster — less traffic captured by model |


## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · matplotlib