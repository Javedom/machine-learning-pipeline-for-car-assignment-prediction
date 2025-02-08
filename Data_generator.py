import pandas as pd
import random

# Parameters
num_passengers = 200            # Total unique passengers
num_historical_entries = 5000   # Total historical schedule entries (rows in dataset 2)
group_size_range = (3, 5)       # Each group will have between 2 and 5 passengers
cars = [1, 2, 3, 4, 5]          # Available cars

# Define available schedules and bias for schedule selection.
schedules = ['8:00', '9:00', '10:00']
schedule_bias = {
    '8:00': 1,
    '9:00': 1,   # "9:00" is three times as likely as "8:00" or "10:00"
    '10:00': 1
}

# Define bias for car selection. (For example, car 1 might be twice as likely.)
car_bias = {
    1: 10,
    2: 1,
    3: 1,
    4: 5,
    5: 1
}

# Define bias for day of week selection.
# For example, day 5 (Friday) might be more frequent.
day_of_week_bias = {
    1: 5,
    2: 1,
    3: 1,
    4: 1,
    5: 10,
    6: 1,
    7: 1
}

# Generate unique passenger names
passenger_names = [f'Passenger_{i}' for i in range(1, num_passengers + 1)]
# (No need to shuffle here since we will sample from this list later)

# -----------------------------------------------------------------------------
# Create Data Set 2 (Historical Data)
# -----------------------------------------------------------------------------
data2_entries = []

# Continue generating groups until we have enough historical entries
while len(data2_entries) < num_historical_entries:
    # Determine a random group size (between group_size_range values)
    group_size = random.randint(*group_size_range)
    # Randomly select group_size passengers from our pool (sample without replacement)
    group_passengers = random.sample(passenger_names, group_size)
    
    # Choose a car using the bias weights
    car = random.choices(cars, weights=[car_bias[c] for c in cars], k=1)[0]
    
    # Choose a schedule using the bias
    schedule = random.choices(schedules, weights=[schedule_bias[s] for s in schedules], k=1)[0]
    
    # Choose a day of the week (1-7) using the bias
    days = list(range(1, 8))
    day = random.choices(days, weights=[day_of_week_bias[d] for d in days], k=1)[0]
    
    # Create an entry for each passenger in the group
    for passenger in group_passengers:
        data2_entries.append({
            'Passenger': passenger,
            'Schedule': schedule,
            'day_of_week': day,
            'Car': car
        })

# Trim any extra entries to match exactly num_historical_entries
if len(data2_entries) > num_historical_entries:
    data2_entries = data2_entries[:num_historical_entries]

# Create DataFrame for Data Set 2
data2 = pd.DataFrame(data2_entries)

# (Optional) Ensure that all cars are used at least once.
used_cars = data2['Car'].unique()
unused_cars = list(set(cars) - set(used_cars))
if unused_cars:
    for car in unused_cars:
        passenger = random.choice(passenger_names)
        schedule = random.choices(schedules, weights=[schedule_bias[s] for s in schedules], k=1)[0]
        day = random.choices(days, weights=[day_of_week_bias[d] for d in days], k=1)[0]
        new_entry = {
            'Passenger': passenger,
            'Schedule': schedule,
            'day_of_week': day,
            'Car': car
        }
        data2 = pd.concat([data2, pd.DataFrame([new_entry])], ignore_index=True)

# -----------------------------------------------------------------------------
# Create Data Set 1 (New Data)
# -----------------------------------------------------------------------------
# For the new dataset, each passenger is assigned a schedule (using unbiased random selection).
data1 = pd.DataFrame({
    'Passenger': random.choices(passenger_names, k=num_passengers),
    'Schedule': random.choices(schedules, k=num_passengers)
})

# -----------------------------------------------------------------------------
# Save datasets as CSV for testing
# -----------------------------------------------------------------------------
data2.to_csv("data2_complex.csv", index=False)
data1.to_csv("data1_complex.csv", index=False)

print("Test datasets generated:")
print("Data2 (Historical) sample:")
print(data2.head())
print("\nData1 (New) sample:")
print(data1.head())
