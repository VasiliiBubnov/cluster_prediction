


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import monotonically_increasing_id
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
# Создание SparkSession
spark = SparkSession.builder \
    .appName("Forex Clustering with PySpark") \
    .getOrCreate()

# Загрузка данных
data= pd.read_csv(r'C:\Users\ext17\Downloads\DAT_ASCII_EURUSD_M1_2020_new.csv')

# Создание новых признаков
"""
data = data.withColumn('Close-Open', data['Close'] - data['Open']) \
    .withColumn('High-Low', data['High'] - data['Low']) \
    .withColumn('High-Close', data['High'] - data['Close']) \
    .withColumn('High-Open', data['High'] - data['Open']) \
    .withColumn('Low-Close', data['Low'] - data['Close']) \
    .withColumn('Low-Open', data['Low'] - data['Open']) \
    .withColumn('index', monotonically_increasing_id())
    """
import pywt

# Разделение временного ряда на подмножества
# Разделение временного ряда на подмножества
subseq_length = 50
close_prices = data["Close"].tolist()
subsequences = [close_prices[i:i + subseq_length] for i in range(len(close_prices) - subseq_length + 1)]

# ...


# Вычисление вейвлет-коэффициентов для каждой подпоследовательности
wavelet_coeffs = []
wavelet_type = 'db1'  # Выберите тип вейвлета (например, Daubechies 'db1')

for subseq in subsequences:
    coeffs = pywt.wavedec(subseq, wavelet_type)
    coeffs_flattened = np.concatenate(coeffs)
    wavelet_coeffs.append(coeffs_flattened)

# Конвертирование вейвлет-коэффициентов в DataFrame и добавление ID
wavelet_coeffs_df = pd.DataFrame(wavelet_coeffs)
wavelet_coeffs_df['index'] = np.arange(len(wavelet_coeffs_df))

# Создание Spark DataFrame из wavelet_coeffs_df
wavelet_coeffs_spark_df = spark.createDataFrame(wavelet_coeffs_df)

# Замена строки assembler на новую строку, которая использует все столбцы, кроме 'index', в качестве входных данных
assembler = VectorAssembler(inputCols=['Close-Open', 'High-Low', 'High-Close', 'High-Open', 'Low-Close', 'Low-Open'], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
pipeline = Pipeline(stages=[assembler, scaler])

# Обучение и применение конвейера преобразования данных
model = pipeline.fit(data)
scaled_data = model.transform(data)

# Кластеризация с использованием KMeans
n_clusters = 30
kmeans = KMeans(featuresCol='scaled_features', k=n_clusters)
kmeans_model = kmeans.fit(scaled_data)
predictions = kmeans_model.transform(scaled_data)

# Создание нового DataFrame с кластерами
data_with_clusters = data.join(predictions.select('index', 'prediction'), predictions['index'] == data['index'])

# Д
# Добавление столбца с метками кластеров
data_with_clusters = data_with_clusters.withColumnRenamed('prediction', 'Cluster')

# Сохранение меток кластеров в текстовый файл
cluster_labels = data_with_clusters.select('Cluster').toPandas()


# Сохранение меток кластеров в текстовый файл в одну строку с одиночным пробелом между ними
with open(r'C:\Users\ext17\Downloads\clustergig222.txt', "w") as f:
    for idx, label in enumerate(cluster_labels['Cluster']):
        if idx != 0:
            f.write(" ")
        f.write(str(label))

# Подсчет и вывод количества сохраненных меток


# Подсчет количества элементов в каждом кластере
pd.set_option('display.max_rows', None)
print(cluster_labels['Cluster'].value_counts())
value_counts = cluster_labels['Cluster'].value_counts()
value_counts.to_csv(r'C:\Users\ext17\Downloads\cluster_countsgig222.txt', index=True, header=['Count'])

# Вычисление средних арифметических моделей свеч для каждого класса
mean_values = data_with_clusters.groupBy('Cluster').mean('Close-Open', 'High-Low', 'High-Close', 'High-Open', 'Low-Close', 'Low-Open')

# Сохранение средних значений в файл

mean_values.toPandas().to_csv(r'C:\Users\ext17\Downloads\meangig222.csv', index=False)

# Отображение графиков
mean_values_pd = mean_values.toPandas()
n_plots = 20
bar_width = 0.15

for plot_counter in range(n_plots):
    cluster_label = plot_counter % n_clusters
    mean_data = mean_values_pd.loc[mean_values_pd['Cluster'] == cluster_label]

    fig, ax = plt.subplots()
    index = np.arange(len(mean_data.columns) - 1)
    rects1 = ax.bar(index, mean_data.iloc[0, 1:], bar_width, label=f'Cluster {cluster_label}')

    ax.set_title(f'Средние значения атрибутов свечей для кластера {cluster_label}')
    ax.set_xticks(index)
    ax.set_xticklabels(['Close-Open', 'High-Low', 'High-Close', 'High-Open', 'Low-Close', 'Low-Open'])
    ax.legend()
 
    plt.show()
    print(f"Количество сохраненных меток: {len(cluster_labels)}")