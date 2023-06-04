import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# 1. Загрузка котировок форекс из файла
file_path = r"C:\Users\ext17\Downloads\Part5_Materials\Part5_Materials\USDJPYH1MAIN.csv"
data = pd.read_csv(file_path)

# 2. Назначаем цифровой индекс и удаляем колонку datetime
data.drop(columns=['date'], inplace=True)
data.reset_index(inplace=True)
delta=500
perv=8100
vt=perv+delta
tr=vt
chet=tr+delta
# Выбираем нужный срез
data1 = data.loc[perv:vt]
data2 = data.loc[tr:chet]

lines = []

for index, row in data1.iterrows():
    color = 'g' if row['Close'] > row['Open'] else 'r'
    plt.plot([index, index], [row['Open'], row['Close']], color=color, linewidth=2)

    # Проводим на графике прямую линию для каждой котировки
    x1 = row['index']
    y1 = row['Open']
    x2 = row['index'] +1
    y2 = row['Close']

    # Расчет коэффициентов наклона и смещения для синей линии
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    lines.append((slope, intercept, index))

# Словарь для хранения y-координат для каждого x

intersect_points = []
y_dict = {}
y_mean_dict = {}

for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        m1, c1, index1 = lines[i]
        m2, c2, index2 = lines[j]
        if m1 != m2:  # чтобы избежать деления на ноль
            x = (c2 - c1) / (m1 - m2)
            y = m1 * x + c1
            if x >= data1['index'].min() - 1000 and x <= data1['index'].max() + 1000:
                # Добавляем y-координату в словарь для соответствующего x
                x_int = math.floor(x)  # Берем целую часть x
                max_index = max(index1, index2)
                if x_int < max_index:
                    x_int = max_index + (max_index - x_int)
                    y = -y  # Меняем y на -y
                intersect_points.append((x_int, y))
                if x_int in y_dict:
                    y_dict[x_int].append(y)
                else:
                    y_dict[x_int] = [y]

# Отрисовываем точки пересечения
#for point in intersect_points:
    #plt.plot(point[0], point[1], 'ko')

# Вычисляем и рисуем средние y-координаты для каждого x
for x_int in sorted(y_dict.keys()):
    y_list = y_dict[x_int]
    y_mean = np.mean(y_list)
    y_mean_dict[x_int] = y_mean
    plt.plot(x_int, y_mean, 'yo', markersize=1)
# Добавляем котировки из второго среза на график
for index, row in data2.iterrows():
    color = 'g' if row['Close'] > row['Open'] else 'r'
    plt.plot([index, index], [row['Open'], row['Close']], color=color, linewidth=2)

range_data = max(data1[['Open', 'Close']].max().max(), data2[['Open', 'Close']].max().max()) - min(data1[['Open', 'Close']].min().min(), data2[['Open', 'Close']].min().min())
padding = 0.4 * range_data
ax = plt.gca()

# Установка пределов для оси Y
ax.set_ylim(min(data1[['Open', 'Close']].min().min(), data2[['Open', 'Close']].min().min()) - padding, max(data1[['Open', 'Close']].max().max(), data2[['Open', 'Close']].max().max()) + padding)

# Обновление границы x для добавления дополнительных 40 индексов слева и справа
ax.set_xlim(data1['index'].min() - 40, data2['index'].max() + 40)

# Создание DataFrame из словаря y_mean_dict
df_yellow_points = pd.DataFrame(list(y_mean_dict.items()), columns=['x', 'y'])

# Расчет скользящей средней с окном в 10 точек
df_yellow_points['y_rolling_mean'] = df_yellow_points['y'].rolling(window=100).mean()
window = 100

df_yellow_points['x'] = df_yellow_points['x'] - window / 2
# Рисуем скользящую среднюю на графике
plt.plot(df_yellow_points['x'], df_yellow_points['y_rolling_mean'], 'b-', linewidth=1)

plt.show()

df_yellow_points.set_index('x', inplace=True)

# Объединение data1 и data2
data_combined = pd.concat([data1, data2])

# Объединение data_combined и df_yellow_points
data_final = data_combined.join(df_yellow_points, how='outer')

# Запись data_final в файл CSV
data_final.to_csv(r"C:\Users\ext17\OneDrive\Рабочий стол\combined_data.csv")

from sklearn.preprocessing import MinMaxScaler

# Сначала объединим два набора данных
df_combined = pd.concat([data1[['Open', 'Close']], data2[['Open', 'Close']]], axis=0)

# Находим минимальное и максимальное значения
y_min = df_combined.min().min()
y_max = df_combined.max().max()

# Создаем экземпляр MinMaxScaler с найденными минимальными и максимальными значениями
scaler = MinMaxScaler(feature_range=(y_min, y_max))

# Масштабируем y_rolling_mean
df_yellow_points['y_rolling_mean_scaled'] = scaler.fit_transform(df_yellow_points[['y_rolling_mean']])

# Рисуем масштабированную скользящую среднюю на графике
plt.plot(df_yellow_points.index, df_yellow_points['y_rolling_mean_scaled'], 'b-', linewidth=1)


plt.show()

fig, ax1 = plt.subplots()  # Создаем новое окно для рисования

# Рисуем скользящую среднюю на графике (синий график)
ax1.plot(df_yellow_points.index, df_yellow_points['y_rolling_mean'], 'b-', linewidth=1)
ax1.set_xlim(data1['index'].min() - 40, data2['index'].max() + 40)
ax1.set_xlabel('Index')
ax1.set_ylabel('Average Intersection Points', color='b')

ax2 = ax1.twinx()  # Создаем вторую ось Y для отображения котировок

# Рисуем котировки из первого и второго среза данных (зеленый график)
for index, row in data1.iterrows():
    color = 'g' if row['Close'] > row['Open'] else 'r'
    ax2.plot([index, index], [row['Open'], row['Close']], color=color, linewidth=2)
for index, row in data2.iterrows():
    color = 'g' if row['Close'] > row['Open'] else 'r'
    ax2.plot([index, index], [row['Open'], row['Close']], color=color, linewidth=2)
ax2.set_ylabel('Forex Quotes', color='g')

plt.show()

