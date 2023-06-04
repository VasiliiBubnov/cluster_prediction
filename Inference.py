

import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer

output_dir = './model_bn_custom/'
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = TFGPT2LMHeadModel.from_pretrained(output_dir)

# Загрузка текста из файла
file_path = "C:\\Users\\ext17\\Downloads\\clusterssss\\cluster223.txt"
with open(file_path, "r") as f:
    text = f.read()

# Разделение текста на номера и создание списка
numbers = text.split()
n = 0

# Создание датафрейма
df = pd.DataFrame(columns=['pred', 'fact'])

# Итерация по тексту и заполнение датафрейма
# ... (оставшаяся часть кода)

# Итерация по тексту и заполнение датафрейма
while n + 50 < len(numbers):
    input_text = ' '.join(numbers[n:n + 50])
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    beam_output = model.generate(input_ids,
                                 max_length=len(input_ids[0]) + 2,  # Изменено на длину входа + 2
                                 num_beams=5,
                                 temperature=0.7,
                                 no_repeat_ngram_size=2,
                                 num_return_sequences=5)
    output_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    pred_number = output_text.split()[0]
    fact_number = numbers[n + 50]

    df = df.append({'pred': pred_number, 'fact': fact_number}, ignore_index=True)
    n += 1

# ... (оставшаяся часть кода)


# Загрузка файла means12.csv
means12 = pd.read_csv("C:\\Users\\ext17\\Downloads\\means12.csv")
df['pred'] = df['pred'].astype('int64')
df['fact'] = df['fact'].astype('int64')  # Преобразуйте тип данных столбца 'fact' в int64

# Получение значений для колонок pred_value и pred_fact

df = df.merge(means12[['Cluster', 'Mean_Close_Open']], left_on='pred', right_on='Cluster', how='left')
df = df.rename(columns={'Mean_Close_Open': 'pred_value'}).drop(columns=['Cluster'])

df = df.merge(means12[['Cluster', 'Mean_Close_Open']], left_on='fact', right_on='Cluster', how='left')
df = df.rename(columns={'Mean_Close_Open': 'pred_fact'}).drop(columns=['Cluster'])

# Сравнение знаков и создание колонки срав
df['срав'] = (np.sign(df['pred_value']) == np.sign(df['pred_fact'])).astype(int)

# Деление pred_fact на pred_value и создание колонки мод
df['мод'] = df['pred_fact'] / df['pred_value']

# Вывод и визуализация
print(df[['срав', 'мод']])
df[['срав', 'мод']].plot()
