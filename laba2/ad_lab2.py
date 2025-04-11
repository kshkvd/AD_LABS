import os
import urllib.request
from datetime import datetime
import pandas as pd

# === Глобальні змінні ===
BASE_URL = "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php"
DATA_DIR = "data"

# === 1. Створення директорії ===
def create_data_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

# === 2. Завантаження даних по області ===
def download_vhi_data(region_id):
    now = datetime.now().strftime("%d%m%Y%H%M%S")
    filename = f"NOAA_ID{region_id}_{now}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    url = f"{BASE_URL}?country=UKR&provinceID={region_id}&year1=1981&year2=2024&type=Mean"
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()

        with open(filepath, 'wb') as file:
            file.write(data)

        print(f"[OK] Дані для області {region_id} збережено як {filename}")
    except Exception as e:
        print(f"[ERR] Помилка при завантаженні для області {region_id}: {e}")

# === 3. Зчитування всіх CSV у один DataFrame ===
def read_all_data(directory):
    headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']
    all_frames = []

    for file in os.listdir(directory):
        if file.startswith("NOAA_ID") and file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(directory, file), header=1, names=headers)

                df = df[pd.to_numeric(df['Year'], errors='coerce').notnull()]
                df['Year'] = df['Year'].astype(int)

                df = df[df['VHI'] != -1]

                region_id = int(file.split("ID")[1].split("_")[0])
                df['filename'] = file
                df['area'] = region_id
                all_frames.append(df)
            except Exception as e:
                print(f"[WARN] Пропущено файл {file}: {e}")

    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    else:
        print("[ERR] Жоден файл не був прочитаний!")
        return pd.DataFrame()

# === 4. Перейменування індексів областей ===
def rename_areas(df):
    index_map = {
        1: 'Вінницька', 2: 'Волинська', 3: 'Дніпропетровська', 4: 'Донецька', 5: 'Житомирська',
        6: 'Закарпатська', 7: 'Запорізька', 8: 'Івано-Франківська', 9: 'Київська',
        10: 'Кіровоградська', 11: 'Луганська', 12: 'Львівська', 13: 'Миколаївська',
        14: 'Одеська', 15: 'Полтавська', 16: 'Рівенська', 17: 'Сумська', 18: 'Тернопільська',
        19: 'Харківська', 20: 'Херсонська', 21: 'Хмельницька', 22: 'Черкаська',
        23: 'Чернівецька', 24: 'Чернігівська', 25: 'Крим'
    }
    df['region_name'] = df['area'].replace(index_map)
    return df

# === 5. Функції вибірки та аналізу ===
def get_vhi_by_year(df, region, year):
    result = df[(df['region_name'] == region) & (df['Year'] == year)][['Week', 'VHI']]
    print(f"\nVHI для області {region} за рік {year}:\n", result)
    return result

def vhi_statistics(df, region, year):
    filtered = df[(df['region_name'] == region) & (df['Year'] == year)]['VHI']
    print(f"\nСтатистика VHI для {region} у {year} році:")
    print(f"  Мінімум: {filtered.min()}")
    print(f"  Максимум: {filtered.max()}")
    print(f"  Середнє: {filtered.mean():.2f}")
    print(f"  Медіана: {filtered.median()}")

def get_vhi_range(df, region, year_start, year_end):
    result = df[(df['region_name'] == region) & (df['Year'].between(year_start, year_end))][['Year', 'Week', 'VHI']]
    print(f"\nVHI для області {region} з {year_start} по {year_end}:\n", result)
    return result

def detect_droughts(df, min_regions=5):
    droughts = df[df['VHI'] < 15]
    drought_years = droughts.groupby('Year')['region_name'].nunique()
    critical_years = drought_years[drought_years >= min_regions]

    print(f"\nРоки з екстремальною посухою (областей >= {min_regions}):")
    for year in critical_years.index:
        regions = droughts[droughts['Year'] == year]['region_name'].unique()
        print(f"  {year}: {len(regions)} областей -> {', '.join(regions)}")

# === MAIN ===
if __name__ == "__main__":
    create_data_directory()

    print("Завантаження даних для всіх областей...")
    for i in range(1, 26):
        download_vhi_data(i)

    print("\nОбробка збережених файлів...")
    df = read_all_data(DATA_DIR)
    df = rename_areas(df)

    while True:
        print("\n--- Виберіть дію ---")
        print("1 - Показати VHI для області та року")
        print("2 - Статистика VHI по області та року")
        print("3 - Показати VHI для області за діапазон років")
        print("4 - Виявити роки з екстремальною посухою")
        print("0 - Вийти")

        choice = input("Введіть номер опції: ")

        if choice == "1":
            region = input("Введіть назву області (наприклад, Київська): ")
            year = int(input("Введіть рік: "))
            get_vhi_by_year(df, region, year)

        elif choice == "2":
            region = input("Введіть назву області: ")
            year = int(input("Введіть рік: "))
            vhi_statistics(df, region, year)

        elif choice == "3":
            region = input("Введіть назву області: ")
            year_start = int(input("Початковий рік: "))
            year_end = int(input("Кінцевий рік: "))
            get_vhi_range(df, region, year_start, year_end)

        elif choice == "4":
            print("Аналізуємо роки, коли посуха охопила 5 або більше областей...")
            detect_droughts(df, min_regions=5)

        elif choice == "0":
            print("До побачення!")
            break

        else:
            print("Невірна опція. Спробуйте ще раз.")
