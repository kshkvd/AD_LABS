import pandas as pd
import numpy as np
import time
from datetime import datetime

# --- NumPy версія ---
print("--- NumPy версія ---")

# Задання типів для кожної колонки
types = [
    ("Date", "U10"),
    ("Time", "U8"),
    ("Global_active_power", "f8"),
    ("Global_reactive_power", "f8"),
    ("Voltage", "f8"),
    ("Global_intensity", "f8"),
    ("Sub_metering_1", "f8"),
    ("Sub_metering_2", "f8"),
    ("Sub_metering_3", "f8")
]

# Завантаження та очищення
start = time.time()
df_np = np.genfromtxt("household_power_consumption.txt",
                      missing_values=["?", ""],
                      delimiter=';',
                      dtype=types,
                      encoding="UTF-8",
                      names=True)
df_np = df_np[~np.isnan(df_np['Global_active_power'])]
end = time.time()
print("Час завантаження та очищення NumPy:", end - start, "сек")

# 1. Sub_metering_2 > Sub_metering_3
start = time.time()
selected_np = df_np[df_np['Sub_metering_2'] > df_np['Sub_metering_3']]
end = time.time()
print("NumPy sub_metering_2 > sub_metering_3:", end - start, "сек")

# 2. Випадкові 5000 записів
start = time.time()
random_np = df_np[np.random.choice(len(df_np), size=5000, replace=False)]
end = time.time()
print("NumPy 5000 випадкових записів:", end - start, "сек")

# 3. Складна умова
start = time.time()
complex_np = df_np[(df_np['Sub_metering_1'] > 1.0) & (df_np['Voltage'] > 240)]
end = time.time()
print("NumPy складна умова:", end - start, "сек")

# 4. Вибір колонок
start = time.time()
columns_np = df_np[['Voltage', 'Global_intensity']]
end = time.time()
print("NumPy вибір колонок:", end - start, "сек")

# 5. Потужність > 5 кВт
start = time.time()
power_np = df_np[df_np['Global_active_power'] > 5]
end = time.time()
print("NumPy Global_active_power > 5 кВт:", end - start, "сек")

# 6. Напруга > 235 В
start = time.time()
volt_np = df_np[df_np['Voltage'] > 235]
end = time.time()
print("NumPy Voltage > 235 В:", end - start, "сек")

# 7. Потужність 19-20А, група 2 найбільша
start = time.time()
current_np = df_np[(df_np['Global_intensity'] >= 19) & (df_np['Global_intensity'] <= 20)]
cond_np = current_np[(current_np['Sub_metering_2'] > current_np['Sub_metering_1']) &
                     (current_np['Sub_metering_2'] > current_np['Sub_metering_3'])]
end = time.time()
print("NumPy потужність 19-20А, група 2 найбільша:", end - start, "сек")

# 8. Вибірка 500000 записів, середнє
start = time.time()
sample_np = df_np[np.random.choice(len(df_np), size=500000, replace=False)]
means_np = np.array([
    np.mean(sample_np['Sub_metering_1']),
    np.mean(sample_np['Sub_metering_2']),
    np.mean(sample_np['Sub_metering_3'])
])
end = time.time()
print("NumPy 500000 записів, середнє:", end - start, "сек")
print("Середні значення NumPy:", means_np)

# 9. Завдання 5 — після 18:00, >6 кВт, група 2 найбільша, кожен 3-й/4-й
start = time.time()
times_np = np.array([datetime.strptime(t, "%H:%M:%S").time() for t in df_np['Time']])
after_18_mask = np.array([t >= datetime.strptime("18:00:00", "%H:%M:%S").time() for t in times_np])
filtered_np = df_np[(df_np['Global_active_power'] > 6) & after_18_mask]
group2_mask = (filtered_np['Sub_metering_2'] > filtered_np['Sub_metering_1']) & \
              (filtered_np['Sub_metering_2'] > filtered_np['Sub_metering_3'])
filtered_group2_np = filtered_np[group2_mask]
half_np = len(filtered_group2_np) // 2
first_half_np = filtered_group2_np[:half_np]
second_half_np = filtered_group2_np[half_np:]
final_np_selection = np.concatenate([first_half_np[::3], second_half_np[::4]])
end = time.time()
print("NumPy після 18:00, >6кВт, група 2 max, кожен 3-й/4-й:", end - start, "сек")

# --- Pandas версія ---
print("\n--- Pandas версія ---")

# Завантаження та очищення
start = time.time()
pandas_df = pd.read_csv("household_power_consumption.txt", sep=';', na_values=['?'])
pandas_df.dropna(inplace=True)
end = time.time()
print("Час завантаження та очищення Pandas:", end - start, "сек")

# 1. Sub_metering_2 > Sub_metering_3
start = time.time()
selected = pandas_df[pandas_df['Sub_metering_2'] > pandas_df['Sub_metering_3']]
end = time.time()
print("Pandas sub_metering_2 > sub_metering_3:", end - start, "сек")

# 2. Випадкові 5000 записів
start = time.time()
random_sample = pandas_df.sample(n=5000, replace=False)
end = time.time()
print("Pandas 5000 випадкових записів:", end - start, "сек")

# 3. Складна умова
start = time.time()
complex_selection = pandas_df[(pandas_df['Sub_metering_1'] > 1.0) & (pandas_df['Voltage'] > 240)]
end = time.time()
print("Pandas складна умова:", end - start, "сек")

# 4. Вибір колонок
start = time.time()
selected_columns = pandas_df.iloc[:, [4, 5]]
end = time.time()
print("Pandas вибір колонок:", end - start, "сек")

# 5. Потужність > 5 кВт
start = time.time()
power_gt_5 = pandas_df[pandas_df['Global_active_power'] > 5]
end = time.time()
print("Pandas Global_active_power > 5 кВт:", end - start, "сек")

# 6. Напруга > 235 В
start = time.time()
voltage_gt_235 = pandas_df[pandas_df['Voltage'] > 235]
end = time.time()
print("Pandas Voltage > 235 В:", end - start, "сек")

# 7. Потужність 19-20А, група 2 найбільша
start = time.time()
current_range = pandas_df[(pandas_df['Global_intensity'] >= 19) & (pandas_df['Global_intensity'] <= 20)]
condition_groups = current_range[(current_range['Sub_metering_2'] > current_range['Sub_metering_1']) &
                                 (current_range['Sub_metering_2'] > current_range['Sub_metering_3'])]
end = time.time()
print("Pandas потужність 19-20А, група 2 найбільша:", end - start, "сек")

# 8. Вибірка 500000 записів, середнє
start = time.time()
sample_500k = pandas_df.sample(n=500000, replace=False)
means = sample_500k[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
end = time.time()
print("Pandas 500000 записів, середнє:", end - start, "сек")
print("Середні значення Pandas:", means)

# 9. Завдання 5 — після 18:00, >6 кВт, група 2 найбільша, кожен 3-й/4-й
start = time.time()
pandas_df['Time'] = pd.to_datetime(pandas_df['Time'], format='%H:%M:%S', errors='coerce')
filtered_time = pandas_df[(pandas_df['Time'].dt.hour >= 18) & (pandas_df['Global_active_power'] > 6)]
filtered_group2 = filtered_time[(filtered_time['Sub_metering_2'] > filtered_time['Sub_metering_1']) &
                                (filtered_time['Sub_metering_2'] > filtered_time['Sub_metering_3'])]
half = len(filtered_group2) // 2
first_half = filtered_group2.iloc[:half]
second_half = filtered_group2.iloc[half:]
final_selection = pd.concat([first_half.iloc[::3], second_half.iloc[::4]])
end = time.time()
print("Pandas після 18:00, >6кВт, група 2 max, кожен 3-й/4-й:", end - start, "сек")
