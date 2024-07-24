from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载预训练的BERT模型和分词器
model = AutoModelForSequenceClassification.from_pretrained("../data/bert/bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("../data/bert/bert-base-uncased")

# 对输入语句进行编码
inputs = tokenizer("The weather is really nice today, I'm very happy", return_tensors="pt")
# inputs = tokenizer("The weather is very bad today, and I feel very atmospheric", return_tensors="pt")

# 获取BERT模型的输出
outputs = model(**inputs)

# 获取情感类别
probabilities = torch.softmax(outputs.logits, dim=-1)
predicted_class = torch.argmax(probabilities).item()

# 输出情感类别
if predicted_class == 0:
    print("消极")
elif predicted_class == 1:
    print("中立")
else:
    print("积极")
