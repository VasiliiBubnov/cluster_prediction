

import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import spearmanr, kendalltau

def prediction(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    beam_output = model.generate(input_ids,
                                  max_length=len(input_ids[0]) + 50,
                                  num_beams=5,
                                  temperature=0.7,
                                  no_repeat_ngram_size=2,
                                  num_return_sequences=5)
    return tokenizer.decode(beam_output[0], skip_special_tokens=True)
def prediction1(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    beam_output = model1.generate(input_ids,
                                  max_length=len(input_ids[0]) + 50,
                                  num_beams=5,
                                  temperature=0.7,
                                  no_repeat_ngram_size=2,
                                  num_return_sequences=5)
    return tokenizer.decode(beam_output[0], skip_special_tokens=True)

output_dir = './model_custom777/'
output_dir1 = './model_custom777 rev/'
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = TFGPT2LMHeadModel.from_pretrained(output_dir)
model1 = TFGPT2LMHeadModel.from_pretrained(output_dir1)

file_path = "C:\\Users\\ext17\\Downloads\\clusterssss\\clusterrick777.txt"
with open(file_path, "r") as f:
    text = f.read()

means12 = pd.read_csv("C:\\Users\\ext17\\Downloads\\means7777.csv")

numbers = text.split()
n = 0

df = pd.DataFrame(columns=['pred', 'fact'])
cumsum_df = pd.DataFrame(columns=['cumsum_input', 'cumsum_output'])


correlation_df = pd.DataFrame()



# The rest of the code for cumsum and plotting remains the same
# ... (the same code as above)

iteration = 0
n = 230

all_cumsum_input = []
all_cumsum_output = []



while n + 50 < len(numbers) and iteration < 1:
    input_text = ' '.join(numbers[n:n + 50])
    current_text = input_text

    for i in range(10):
        # Предсказываем пред
        output_text = prediction(current_text)
        last_50_output = output_text.split()[-50:]
        df.loc[iteration, 'pred'] = last_50_output
        iteration += 1  # Увеличиваем итерацию после каждого предсказания пред

        # Предсказываем факт на основе пред
        reversed_last_50_output = list(reversed(last_50_output))
        output_text = prediction1(' '.join(reversed_last_50_output))
        last_50_output = output_text.split()[-50:]
        df.loc[iteration - 1, 'fact'] = last_50_output  # Записываем факт в предыдущую строку, соответствующую пред

        current_text = ' '.join(last_50_output)

    n += 1  # Сдвигаем подмассив на 1 элемент вперед

for iteration in range(10):
    if 'pred' in df.columns and iteration in df.index:
        input_clusters = [int(cluster) for cluster in df.loc[iteration, 'pred']]
    else:
        input_clusters = []
        
    if 'fact' in df.columns and iteration in df.index:
        output_clusters = [int(cluster) for cluster in df.loc[iteration, 'fact']]
    else:
        output_clusters = []

    input_mean_close_open = [means12.loc[means12['Cluster'] == cluster, 'Mean_Close_Open'].values[0] for cluster in input_clusters]
    output_mean_close_open = [means12.loc[means12['Cluster'] == cluster, 'Mean_Close_Open'].values[0] for cluster in output_clusters]

    cumsum_input = np.cumsum(np.array(input_mean_close_open))
    cumsum_output = np.cumsum(np.array(output_mean_close_open))

    all_cumsum_input.append(cumsum_input)
    all_cumsum_output.append(cumsum_output)


# Рисуем все массивы cumsum_input
plt.figure()
for i, cumsum_input in enumerate(all_cumsum_input):
    plt.plot(cumsum_input, label=f'p{i+1}')
plt.xlabel('Element Index')
plt.ylabel('Cumulative Sum')
plt.title('Cumulative Sum of Mean_Close_Open for Input (pred)')
plt.legend()
plt.show()
plt.pause(1)
plt.close()

# Рисуем все массивы cumsum_output
plt.figure()
for i, cumsum_output in enumerate(all_cumsum_output):
    plt.plot(cumsum_output, label=f'f{i+1}')
plt.xlabel('Element Index')
plt.ylabel('Cumulative Sum')
plt.title('Cumulative Sum of Mean_Close_Open for Output (fact)')
plt.legend()
plt.show()
plt.pause(1)
plt.close()
# Вычисляем среднее арифметическое всех строк в колонке пред
average_pred = df['pred'].apply(lambda x: np.mean([int(cluster) for cluster in x])).values

# Вычисляем среднее арифметическое всех строк в колонке факт
average_fact = df['fact'].apply(lambda x: np.mean([int(cluster) for cluster in x])).values


average_input = np.mean(all_cumsum_input, axis=0)

# Вычисляем среднее арифметическое между элементами разных массивов all_cumsum_output
average_output = np.mean(all_cumsum_output, axis=0)

# Рисуем график со средними значениями all_cumsum_input
plt.figure()
plt.plot(average_input, label='Average Input')
plt.xlabel('Element Index')
plt.ylabel('Average Value')
plt.title('Average Value of All Cumulative Sums of Input')
plt.legend()
plt.show()
plt.pause(1)
plt.close()

# Рисуем график со средними значениями all_cumsum_output
plt.figure()
plt.plot(average_output, label='Average Output')
plt.xlabel('Element Index')
plt.ylabel('Average Value')
plt.title('Average Value of All Cumulative Sums of Output')
plt.legend()
plt.show()
plt.pause(1)
plt.close()


