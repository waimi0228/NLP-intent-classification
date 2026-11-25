import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

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

model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=num_labels)

# 訓練參數設定
#微調
training_args = TrainingArguments(
    output_dir='./results',  # 模型保存路徑
    learning_rate=5e-5, # 調整學習率
    num_train_epochs=3,  # 訓練次數
    per_device_train_batch_size=32,  # 每個設備的訓練批次大小
    per_device_eval_batch_size=32,  # 每個設備的評估批次大小
    warmup_steps=500,  # 進行多少步驟的學習率熱身
    weight_decay=0.01,  # 權重衰減
    logging_dir='./logs',  # 日誌路徑
    logging_steps=10,
    evaluation_strategy="epoch"  # 每個 epoch 評估一次
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

# 預測單個問題的函數
def predict_question(question):
    encoding = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    return predicted_label


# 測試預測
test_question = "第一次辦理身分證"
predicted_label = predict_question(test_question)
print   (f'Predicted Label: {predicted_label}')