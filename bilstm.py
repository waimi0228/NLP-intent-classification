import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# 讀取數據
data = pd.read_csv('identification.csv')
df = pd.DataFrame(data)

# 標籤編碼
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 訓練集和測試集拆分
X_train, X_test, y_train, y_test = train_test_split(df['question'], df['label'], test_size=0.2, random_state=42)

# 文本序列化處理
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 設置最大序列長度為 100，並填充序列
max_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# 獲取類別數量
num_classes = len(np.unique(y_train))

# 建立模型
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_length),
    SpatialDropout1D(0.3),  # 空間 Dropout
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),  # 增加 Dense 層的單元數
    Dense(num_classes, activation='softmax')  # 使用 softmax 處理多類別分類
])

# 編譯模型，使用 sparse_categorical_crossentropy 適合對應數字編碼的標籤
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型，並使用 EarlyStopping 回調函數避免過擬合
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train_padded, y_train, epochs=20, validation_data=(X_test_padded, y_test), batch_size=32, callbacks=[early_stopping])

# 評估模型
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# 預測並顯示分類報告
predictions = np.argmax(model.predict(X_test_padded), axis=1)  # 多類別需要使用 argmax 獲取預測類別
print(classification_report(y_test, predictions))

# 預測單個問題的函數
def predict_question(question):
    seq = tokenizer.texts_to_sequences([question])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]  # 使用 argmax 獲取類別
    return label

# 測試預測
test_question = '初領身份證'
predicted_label = predict_question(test_question)
print(f'Predicted Label: {predicted_label}')




#print(df['label'].value_counts())