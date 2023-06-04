from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import monotonically_increasing_id, collect_list, row_number
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Создание SparkSession
spark = SparkSession.builder \
    .appName("Forex Clustering with PySpark") \
    .getOrCreate()

# Загрузка данных
data_2020 = spark.read.csv(r'C:\Users\ext17\Downloads\DAT_ASCII_EURUSD_M1_2020\DAT_ASCII_EURUSD_M1_2020_new.csv', header=True, inferSchema=True)
data_2021 = spark.read.csv(r'C:\Users\ext17\Downloads\DAT_ASCII_EURUSD_M1_2021\DAT_ASCII_EURUSD_M1_2021_new.csv', header=True, inferSchema=True)
data_2022 = spark.read.csv(r'C:\Users\ext17\Downloads\DAT_ASCII_EURUSD_M1_2022\DAT_ASCII_EURUSD_M1_2022_new.csv', header=True, inferSchema=True)

# Объединение датафреймов
data = data_2020.union(data_2021).union(data_2022)

# Создание новых признаков
data = data.withColumn('Close-Open', data['Close'] - data['Open']) \
    .withColumn('index', monotonically_increasing_id())

# Создание окна для вычисления 10-строчных элементов
window = Window.orderBy("index").rowsBetween(0, 9)

# Сбор 10-строчных элементов в список и фильтрация данных, чтобы оставить только каждый десятый элемент
data_10_rows = data.select('index', collect_list("Close-Open").over(window).alias("Close-Open_list")) \
    .filter(row_number().over(window) % 10 == 0)

# Создание конвейера преобразования данных
assembler = VectorAssembler(inputCols=['Close-Open_list'], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
pipeline = Pipeline(stages=[assembler, scaler])

# Обучение и применение конвейера преобразования данных
model = pipeline.fit(data_10_rows)
scaled_data = model.transform(data_10_rows)

# Кластеризация с использованием KMeans
n_clusters = 10
kmeans = KMeans(featuresCol='scaled_features', k=n_clusters)
kmeans_model = kmeans.fit(scaled_data)
predictions = kmeans_model.transform(scaled_data)

# Добавление столбца с метками кластеров
data_with_clusters = data_10_rows.join(predictions.select('index', 'prediction'), predictions['index'] == data_10_rows['index']).withColumnRenamed('prediction', 'Cluster')
cluster_labels = data_with_clusters.select('Cluster').toPandas()
with open(r'C:\Users\ext17\Downloads\clustergig.txt', "w") as f:
    for idx, label in enumerate(cluster_labels['Cluster']):
        if idx != 0:
            f.write(" ")
        f.write(str(label))
pd.set_option('display.max_rows', None)
print(cluster_labels['Cluster'].value_counts())
value_counts = cluster_labels['Cluster'].value_counts()
value_counts.to_csv(r'C:\Users\ext17\Downloads\cluster_countsgig.txt', index=True, header=['Count'])
mean_values = data_with_clusters.groupBy('Cluster').mean('Close-Open_list')
mean_values.toPandas().to_csv(r'C:\Users\ext17\Downloads\meangig.csv', index=False)
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
    ax.set_xticklabels(['Close-Open'])
    ax.legend()
    
    plt.show()
    print(f"Количество сохраненных меток: {len(cluster_labels)}")
