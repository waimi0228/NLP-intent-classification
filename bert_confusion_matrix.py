import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取數據
data = pd.read_csv('identification.csv')  # 修改為你的檔案路徑
df = pd.DataFrame(data)

# 標籤編碼
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
num_labels = len(label_encoder.classes_)

# 訓練集和測試集拆分
X_train, X_test, y_train, y_test = train_test_split(df['question'], df['label'], test_size=0.2, random_state=50)

# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 創建自定義 Dataset 類別
class QuestionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 設定最大長度
MAX_LEN = 100

# 創建訓練和測試 Dataset
train_dataset = QuestionDataset(X_train.values, y_train.values, tokenizer, MAX_LEN)
test_dataset = QuestionDataset(X_test.values, y_test.values, tokenizer, MAX_LEN)

# 加載 BERT 預訓練模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)

# 訓練參數設定
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# 創建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 訓練模型
trainer.train()

# 評估模型
eval_result = trainer.evaluate()
print(f"Evaluation Result: {eval_result}")

# 預測測試集
test_loader = DataLoader(test_dataset, batch_size=32)
predictions = []
true_labels = y_test.values

# 預測所有測試集數據
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(batch_predictions)

# 計算準確率
accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy}')

# 計算混淆矩陣
conf_matrix = confusion_matrix(true_labels, predictions)

# 顯示混淆矩陣
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
