
===========================================================
📄 DATASET GENERATION SCRIPT FOR PASSENGER SCHEDULE DATA
===========================================================

This script generates two datasets for testing machine learning models that predict car assignments based on passenger schedules.

-----------------------------------------------------------
📂 DATASETS CREATED:
-----------------------------------------------------------
1. **data2_complex.csv** - Historical data with the following columns:
   - `Passenger`: Unique passenger identifier
   - `Schedule`: Assigned schedule (8:00, 9:00, 10:00)
   - `day_of_week`: Day of the week (1 for Monday, 7 for Sunday)
   - `Car`: Assigned car (values from 1 to 5)

2. **data1_complex.csv** - New data for prediction containing:
   - `Passenger`: Unique passenger identifier
   - `Schedule`: Assigned schedule

-----------------------------------------------------------
📋 SCRIPT FUNCTIONALITY:
-----------------------------------------------------------
1. **Historical Data Creation (data2)**
   - Generates 5000 historical records with randomly grouped passengers.
   - Schedule, car, and day-of-week selections are based on specified biases.

2. **New Data Creation (data1)**
   - Creates a dataset for prediction with each passenger assigned a random schedule.

3. **Bias Weights**
   - Allows for bias in selecting schedules, cars, and days of the week.
   - For example, Fridays (day 5) are more frequent, and certain cars are more likely to be assigned.

4. **CSV File Output**
   - Saves both datasets as `data2_complex.csv` and `data1_complex.csv` for easy use in machine learning models.

-----------------------------------------------------------
⚡ HOW TO USE:
-----------------------------------------------------------
1. Run the script to generate the datasets:
   ```
   python generate_data.py
   ```
2. Use the generated `data2_complex.csv` for training and `data1_complex.csv` for testing or prediction.

-----------------------------------------------------------
📢 SAMPLE OUTPUT:
-----------------------------------------------------------
**Data2 (Historical) sample:**
```
    Passenger  Schedule  day_of_week  Car
0  Passenger_12    9:00           5    1
1  Passenger_23    8:00           3    4
2  Passenger_45    10:00          5    2
3  Passenger_7     9:00           1    1
4  Passenger_99    8:00           4    3
```

**Data1 (New) sample:**
```
    Passenger  Schedule
0  Passenger_34    9:00
1  Passenger_21    10:00
2  Passenger_56    8:00
3  Passenger_78    9:00
4  Passenger_12    10:00
```

