import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# === Завантаження даних ===
@st.cache_data
def load_data():
    df = pd.read_csv("lab2/data/final_vhi_data.csv")
    return df

df = load_data()

# === Унікальні значення ===
area_names = {i: name for i, name in enumerate(sorted(df['region_name'].unique()))}
years = sorted(df["Year"].unique())
weeks = list(range(1, 53))

# === Ініціалізація session_state ===
if "selected_area" not in st.session_state:
    st.session_state.selected_area = list(area_names.values())[0]

if "indicator" not in st.session_state:
    st.session_state.indicator = "VCI"

if "week_range" not in st.session_state:
    st.session_state.week_range = (1, 52)

if "year_range" not in st.session_state:
    st.session_state.year_range = (min(years), max(years))

if "ascending_order" not in st.session_state:
    st.session_state.ascending_order = False

if "descending_order" not in st.session_state:
    st.session_state.descending_order = False

# === Функція скидання фільтрів ===
def reset_filters():
    st.session_state.selected_area = list(area_names.values())[0]
    st.session_state.indicator = "VCI"
    st.session_state.week_range = (1, 52)
    st.session_state.year_range = (min(years), max(years))
    st.session_state.ascending_order = False
    st.session_state.descending_order = False

# === Макет: дві колонки ===
col1, col2 = st.columns([1, 3])

with col1:
    st.title("Фільтри")

    st.session_state.indicator = st.selectbox(
        "Оберіть індекс", ["VCI", "TCI", "VHI"],
        index=["VCI", "TCI", "VHI"].index(st.session_state.indicator)
    )

    st.session_state.selected_area = st.selectbox(
        "Оберіть область", options=list(area_names.values()),
        index=list(area_names.values()).index(st.session_state.selected_area)
    )

    st.session_state.week_range = st.slider(
        "Інтервал тижнів", 1, 52, st.session_state.week_range
    )

    st.session_state.year_range = st.slider(
        "Інтервал років", min(years), max(years), st.session_state.year_range
    )

    if st.button("Скинути фільтри"):
        reset_filters()

    st.checkbox(
        "Сортувати за зростанням", key="ascending_order"
    )
    st.checkbox(
        "Сортувати за спаданням", key="descending_order"
    )

with col2:
    # === Фільтрація даних ===
    df_filtered = df[
        (df['region_name'] == st.session_state.selected_area) &
        (df['Week'].between(*st.session_state.week_range)) &
        (df['Year'].between(*st.session_state.year_range))
    ]

    # === Сортування ===
    if st.session_state.ascending_order and not st.session_state.descending_order:
        df_filtered = df_filtered.sort_values(by=st.session_state.indicator, ascending=True)
    elif st.session_state.descending_order and not st.session_state.ascending_order:
        df_filtered = df_filtered.sort_values(by=st.session_state.indicator, ascending=False)
    elif st.session_state.ascending_order and st.session_state.descending_order:
        st.warning("Не можна обрати одночасно зростання і спадання.")

    # === Вкладки ===
    tab1, tab2, tab3 = st.tabs(["Таблиця", "Графік", "Порівняння"])

    with tab1:
        st.subheader("📋 Відфільтровані дані")
        st.dataframe(df_filtered)

    with tab2:
        st.subheader("📈 Графік обраної області")
        fig, ax = plt.subplots()
        x_labels = df_filtered["Year"].astype(str) + "-" + df_filtered["Week"].astype(str)
        ax.plot(x_labels, df_filtered[st.session_state.indicator], label=st.session_state.indicator)
        ax.set_xlabel("Рік-Тиждень")
        ax.set_ylabel(st.session_state.indicator)
        ax.set_title(f"{st.session_state.indicator} для {st.session_state.selected_area}")
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xticks(ax.get_xticks()[::10])

        st.pyplot(fig)

    with tab3:
        st.subheader("📊 Порівняння з іншими областями")
        compare_df = df[
            (df['Week'].between(*st.session_state.week_range)) &
            (df['Year'].between(*st.session_state.year_range))
        ]
        avg_data = compare_df.groupby("region_name")[st.session_state.indicator].mean().sort_values()
        st.bar_chart(avg_data)

