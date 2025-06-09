import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Прогноз продолжительности жизни", page_icon="🌍")

# === Загрузка и подготовка данных ===
@st.cache_data
def load_data():
    df = pd.read_csv("LifeData.csv")
    df = df.dropna()
    df['Status'] = LabelEncoder().fit_transform(df['Status'])  # 0 — развивающаяся, 1 — развитая
    X = df.drop(['Country', 'Year', 'Life expectancy '], axis=1)
    y = df['Life expectancy ']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X.columns.tolist(), scaler, X, y

df, feature_names, scaler, X, y = load_data()

# Переименование признаков на русский
feature_names_ru = {
    'Adult Mortality': 'Смертность взрослых',
    'infant deaths': 'Смертность младенцев',
    'Alcohol': 'Потребление алкоголя',
    'percentage expenditure': 'Расходы на здравоохранение (%)',
    'Hepatitis B': 'Вакцинация от гепатита B (%)',
    'Measles ': 'Заболеваемость корью',
    'BMI': 'Индекс массы тела',
    'under-five deaths': 'Смертность детей до 5 лет',
    'Polio': 'Вакцинация от полиомиелита (%)',
    'Total expenditure': 'Общие расходы на здравоохранение (%)',
    'Diphtheria ': 'Вакцинация от дифтерии (%)',
    'HIV/AIDS': 'Смертность от ВИЧ/СПИДа',
    'GDP': 'ВВП',
    'Population': 'Население',
    'thinness  1-19 years': 'Доля худых (1-19 лет)',
    'thinness 5-9 years': 'Доля худых (5-9 лет)',
    'Income composition of resources': 'Доходы населения',
    'Schooling': 'Среднее количество лет обучения',
    'Status': 'Статус страны (0 - развивающаяся, 1 - развитая)'
}

# === Интерфейс ===
st.title("🌍 Прогноз продолжительности жизни")

st.markdown("""
Это приложение предсказывает ожидаемую продолжительность жизни на основе различных социально-экономических и медицинских показателей. Данные взяты из отчётов Всемирной организации здравоохранения (ВОЗ).

Введите значения интересующих вас параметров и нажмите кнопку, чтобы получить результат.
""")

st.header("🔢 Ввод параметров:")
user_input = {}

col1, col2 = st.columns(2)

for i, col in enumerate(feature_names):
    col_ru = feature_names_ru.get(col, col)
    min_val = round(float(df[col].min()), 2)
    max_val = round(float(df[col].max()), 2)
    avg_val = round(float(df[col].mean()), 2)
    step = (max_val - min_val) / 100

    target_col = col1 if i % 2 == 0 else col2

    if df[col].dtype in ['int64', 'float64']:
        user_input[col] = target_col.slider(
            f"{col_ru}", min_value=min_val, max_value=max_val, value=avg_val, step=round(step, 2))
    else:
        user_input[col] = target_col.selectbox(f"{col_ru}", sorted(df[col].unique()))

st.header("⚙️ Настройка модели")
n_estimators = st.slider("Количество деревьев (n_estimators)", 50, 300, 100, step=50)

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X, y)
prediction = model.predict(input_scaled)[0]

st.header("📈 Результат")
if st.button("🔍 Получить прогноз"):
    st.success(f"Ожидаемая продолжительность жизни: **{prediction:.2f} года**")

st.markdown("---")
st.caption("Разработка: студент НИРС | Данные: WHO | Модель: Gradient Boosting")