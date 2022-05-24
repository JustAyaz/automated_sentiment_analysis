#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import *
from sqlalchemy import *
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

# export FLASK_ENV=development
# export FLASK_DEBUG=1


app = Flask(__name__, static_url_path='/static')
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True

PRE_TRAINED_MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


class GPReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class_names = ['neutral', 'positive', 'negative']


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(outputs["pooler_output"])
        return self.out(output)


model = SentimentClassifier(len(class_names))
model_name = 'classifier3.pt'
path = F"{model_name}"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load(path, map_location=device))
# Or you can move the loaded model into the specific device
model.to(device)


def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    del inputs['token_type_ids']
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted


engine = create_engine('postgresql://postgres:imnotweakbyanymeans@localhost:5434/diploma', echo=True)
meta = MetaData(engine)
models = Table('models', meta, autoload=True)
reviews = Table('reviews', meta, autoload=True)
conn = engine.connect()


@app.route('/')
def hello_world():
    # print(predict("Очень плохо!"))
    return "Hello there!"


# @app.route('/predictfeedback/<fdb>', methods=['GET'])
# def predictfeedback(fdb):
#     if request.method == 'GET':
#         print(predict(fdb))
#         return str(predict(fdb))

@app.route('/leavefdb', methods=['GET', 'POST'])
def message():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        review = request.form.get('review')
        rating = request.form.get('rating1')
        print(rating)
        sent = predict(review)
        if sent == 0:
            sentiment = "neutral"
        elif sent == 1:
            sentiment = "positive"
        else:
            sentiment = "negative"

        # Запись в БД
        res = reviews.insert().values(text=review, score=rating, sentiment=sentiment)
        result = conn.execute(res)

        return render_template('success.html')


@app.route('/admin')
def admin():
    panel = {}

    #Все отзывы
    panel["all_reviews"] = 1781
    panel["all_reviews_data"] = [100,170,380,520,1010,1430,1781]

    #Средний рейтинг
    panel["median_raiting_data"] = [4.5,4.9,4.1,3.5,5.5,4.5,4.7]
    s = select(reviews).where(reviews.c.id > 0)
    result = conn.execute(s)
    count = 0
    summ = 0
    for row in result.fetchall():
        if row["score"]:
            summ += row["score"]
            count +=1
    panel["median_raiting"] = float("{0:.2f}".format(summ/count))

    #Новые отзывы
    panel["new_reviews"] = 420
    panel["new_reviews_data"] = [100,70,210,140,490,420]

    #Анализ оценки
    panel["tonalnost_accyracy"] = []
    panel["tonalnost_f1"] = []
    s = select(models).where(models.c.id > 0)
    result = conn.execute(s)
    for row in result.fetchall():
        if row["accuracy"] and row["f1"]:
            panel["tonalnost_accyracy"].append(row["accuracy"])
            panel["tonalnost_f1"].append(row["f1"])


    #Классификация отзывов - положительные - нейтральные - отрицательные
    panel["percent_reviews"] = [949,496,336]

    return render_template('admin.html', panel = panel)


@app.route('/history')
def history():
    s = select(models).where(models.c.id > 0).order_by(desc("id"))
    result = conn.execute(s)
    history = []
    for row in result.fetchall():
        history.append(row)
    return render_template('history.html', history = history)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    status = 0
    if request.method == 'GET':
        status = 0
        return render_template('upload.html', status = status)
    else:
        print("@@@@@")
        print(request.files['file-input'])
        status = 1
        return render_template('upload.html', status=status)


if __name__ == '__main__':
    app.run(debug=True)
