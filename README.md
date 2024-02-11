# pandas-matplotlib-seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Задаем зерно для генератора случайных чисел для воспроизводимости
np.random.seed(42)

# Генерируем случайные данные
data = {
    "Дата": pd.date_range(start="2023-01-01", end="2023-04-10", freq='D'),  # Здесь у нас 100 дней (Daily Generate)
    "Тип события": np.random.choice(['Взлом', 'Мошенничество', 'Вирус', 'DDoS'], 100, replace=True),
    "Время реакции (мин)": np.random.randint(5, 60, 100),
    "Уровень угрозы (1-10)": np.random.randint(1, 11, 100)
}

df = pd.DataFrame(data)

print(df)

from scipy.stats import ttest_ind

virus_reaction_time = df[df["Тип события"] == "Вирус"]["Время реакции (мин)"]
ddos_reaction_time = df[df["Тип события"] == "DDoS"]["Время реакции (мин)"]

t_stat, p_value = ttest_ind(virus_reaction_time, ddos_reaction_time)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

#
import statsmodels.api as sm

X = df["Уровень угрозы (1-10)"]
X = sm.add_constant(X)  # добавляем константу для y-пересечения
y = df["Время реакции (мин)"]

model = sm.OLS(y, X).fit()

print(model.summary())
#
# Визуализация результата
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x="Уровень угрозы (1-10)", y="Время реакции (мин)")
plt.title("Регрессионный анализ: Время реакции в зависимости от уровня угрозы")
plt.show()
