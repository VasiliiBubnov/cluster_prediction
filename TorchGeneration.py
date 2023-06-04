
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Создание набора данных
class MyDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.tokenizer = tokenizer
        with open(data_file, "r") as f:
            self.data = f.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.data[idx]).ids

# Загрузка токенизатора
tokenizer = GPT2TokenizerFast.from_pretrained(r'C:\Users\ext17\Downloads\token1')

# Загрузка набора данных
train_dataset = MyDataset(r'C:\Users\ext17\Downloads\cluster.txt', tokenizer)

# Конфигурация и создание модели
config = GPT2Config(vocab_size=30000)
model = GPT2LMHeadModel(config)

# Обучение модели
training_args = TrainingArguments(
    report_to=None,
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir=None, # добавьте эту строку для отключения логирования Wandb
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Сохранение модели
trainer.save_model("your_trained_model")

# Загрузка обученной модели и токенизатора
model = GPT2LMHeadModel.from_pretrained("your_trained_model")
tokenizer = GPT2TokenizerFast.from_pretrained("your_trained_model")

# Генерация продолжения текста
input_text = "12345"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
generated_output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

print(f"Input text: {input_text}")
print(f"Generated text: {generated_text}")
