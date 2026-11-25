# 行政手続き質問の意図分類（BERT / BiLSTM）

チャットボットで市民の質問に答えるため、質問文がどの手続きに関するものかを分類するモデルを作成しました。
事前学習済みの **BERT** と、ゼロから学習する **BiLSTM** の2手法を比較しています。

| 質問（入力） | 意図ラベル（出力） |
|---|---|
| 第一次辦身分證，該去哪辦理 | 初領身分證 |
| 辦初領身分證，需要帶什麼 | 初領身分證 |

## データ
`identification.csv`：身分証に関する質問 104 件、4 クラス（自作）

| ラベル | 件数 |
|---|---|
| 補辦身分證（再発行） | 34 |
| 初領身分證（初回申請） | 32 |
| 身分證遺失（紛失） | 28 |
| 大頭照規格（証明写真の規格） | 10 |

## 手法
| ファイル | 内容 |
|---|---|
| `bert.py` | `bert-base-chinese` を fine-tuning して分類 |
| `bert_confusion_matrix.py` | BERT の学習に加え、混同行列を可視化 |
| `bilstm.py` | Embedding + 2層 BiLSTM（Keras）、EarlyStopping で過学習を抑制 |

train : test = 8 : 2

## 使用技術
Python / PyTorch / Hugging Face Transformers / TensorFlow (Keras) / scikit-learn

## 実行方法
```bash
pip install -r requirements.txt
python bert.py
python bilstm.py
```
