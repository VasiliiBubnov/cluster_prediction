

import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import spearmanr, kendalltau

output_dir = './model_custom777/'
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = TFGPT2LMHeadModel.from_pretrained(output_dir)

file_path = "C:\\Users\\ext17\\Downloads\\clusterssss\\clusterrick777.txt"
with open(file_path, "r") as f:
    text = f.read()

means12 = pd.read_csv("C:\\Users\\ext17\\Downloads\\means7777.csv")

numbers = text.split()
n = 0

df = pd.DataFrame(columns=['input_text', 'output_text'])
cumsum_df = pd.DataFrame(columns=['fact', 'pred'])
results_df = pd.DataFrame(columns=['sred', 'real'])

iteration = 0
correlation_df = pd.DataFrame()
n = 1
cycle_counter = 0

while n + 50 < len(numbers):
    input_text = ' '.join(numbers[n:n + 200])
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    beam_output = model.generate(input_ids,
                                 max_length=len(input_ids[0]) + 150,
                                 num_beams=5,
                                 temperature=0.7,
                                 no_repeat_ngram_size=2,
                                 num_return_sequences=5)
    output_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)

    df = df.append({'input_text': input_text, 'output_text': output_text}, ignore_index=True)

    input_clusters = list(map(int, ' '.join(numbers[n:n + 350]).split()))
    output_clusters = list(map(int, output_text.split()))

    input_mean_close_open = [means12.loc[means12['Cluster'] == cluster, 'Mean_Close_Open'].values[0] for cluster in input_clusters]
    output_mean_close_open = [means12.loc[means12['Cluster'] == cluster, 'Mean_Close_Open'].values[0] for cluster in output_clusters]

    cumsum_input_mean_close_open = np.cumsum(np.array(input_mean_close_open))
    cumsum_output_mean_close_open = np.cumsum(np.array(output_mean_close_open))

    cumsum_df = cumsum_df.append({'fact': cumsum_input_mean_close_open[-150:], 'pred': cumsum_output_mean_close_open[-50:]}, ignore_index=True)
    cumsum_df.to_csv('cumsum_sum.csv', index=False)

    if cycle_counter % 20 == 0 and cycle_counter > 0:
        last_20_preds = cumsum_df['pred'].tail(20).to_numpy()
        sum_values = np.zeros(20)
        for i, row in enumerate(last_20_preds):
            sum_values[i:] += row[:-(i+1)]
        sred = sum_values.sum() / 20
        real = cumsum_df.iloc[-1]['fact'][-1]

        results_df = results_df.append({'sred': sred, 'real': real}, ignore_index=True)
        results_df.to_csv('results.csv', index=False)
        print(f'Sred: {sred}, Real: {real}')

    # Вывод графика на каждой итерации
    plt.plot(cumsum_input_mean_close_open, label='Input')
    plt.plot(cumsum_output_mean_close_open, label='Output')

    plt.xlabel('Element Index')
    plt.ylabel('Cumulative Sum')
    plt.title(f'Cumulative Sum of Mean_Close_Open for Input and Output (n={n})')
    plt.legend()
    plt.show()
    plt.pause(1)
    plt.close()

    n += 50
    cycle_counter += 1


