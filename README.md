# Predictive Modeling and Optimization in Logistics Systems

This is the code for my Week 4 task in the Yuva Internship. I built models that predict delivery times and shipment volumes for RegionalKart, the made up ecommerce company from my earlier reports. Then I used the predictions to plan staff, choose which orders to expedite and improve van routes.

## What each script does
1. `generate_logistics_data.py` creates 38,000 simulated shipments from October 2024 to March 2026.
2. `01_prepare_features.py` builds features that are known at order time and splits the data by date.
3. `02_delivery_time_model.py` compares five models with time series cross validation, tunes the best one and tests it on the last six months.
4. `03_volume_forecast.py` forecasts daily shipments for each hub.
5. `04_optimization.py` uses the forecasts for staffing (integer programming), expediting late orders and route planning (savings algorithm).

## How to run it
```
pip install -r requirements.txt
python generate_logistics_data.py
python 01_prepare_features.py
python 02_delivery_time_model.py
python 03_volume_forecast.py
python 04_optimization.py
```
Run every command from the main project folder. Fixed random seeds are used so you should get the same numbers as in my report.

## Main results
The tuned gradient boosting model predicts delivery time with a mean absolute error of 5.3 hours on future data. That is about 45% lower than guessing the average. A staffing plan based on coverage lifts the lowest hub coverage from 0.60 to 0.88. Expediting orders that the model flags as late avoids 792 late deliveries in six months. The savings algorithm cuts van distance by 20.7%.

## Folders
```
figures/    charts used in the report
outputs/    printed results and saved numbers from each script
data/       generated files (created when you run the scripts)
```

## A note about the data and assumptions
The data is simulated and the costs used in the optimisation (Rs 25 to expedite and Rs 150 for a late delivery) are my own assumptions. The results show how the method works and do not describe a real company.

## About me
Nayan Deepak Borse, B.E. Computer Engineering (2026)
