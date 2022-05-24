from sqlalchemy import *
import tensorflow as tf
import torch
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import f1_score
import torch.nn.functional as F

# export FLASK_ENV=development
# export FLASK_DEBUG=1

MAX_LEN = 160

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
model.to(device)


def to_sentiment(rating):
    rating = int(rating)
    if rating < 3:
        return 2
    elif rating == 4 or rating == 3:
        return 0
    else:
        return 1


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


try:
    engine = create_engine('postgresql://postgres:imnotweakbyanymeans@localhost:5434/diploma', echo=True)
    meta = MetaData(engine)

    models = Table('models', meta, autoload=True)
    reviews = Table('reviews', meta, autoload=True)

    conn = engine.connect()

    s = select(reviews).where(reviews.c.check == None)
    result = conn.execute(s)
    text = []
    rating = []
    ids = []
    for row in result.fetchall():
        if row["text"] and row["score"]:
            text.append(row["text"])
            rating.append(row["score"])
            ids.append(row["id"])
    print(len(text))
    df = pd.DataFrame({'text': text, 'rating': rating})
    # print(df.info())

    df['sentiment'] = df.rating.apply(to_sentiment)
    print(df.head())

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

    BATCH_SIZE = 16

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    bert_model = model

    model = SentimentClassifier(len(class_names))
    model = model.to(device)

    EPOCHS = 8

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )

    test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
    )
    accuracy = test_acc.item()

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )
    f1 = f1_score(y_test, y_pred, average='macro')

    second_query = select(models).where(models.c.id > 0)
    second_query_result = conn.execute(second_query)
    len_for_name = len(result.fetchall())+1
    name = "classifier" + str(len_for_name) + ".pt"
    path = F"/models/{name}"
    torch.save(model.state_dict(), path)

    now = datetime.now()
    third_query = models.insert().values(date=now, name=name, accuracy=accuracy, f1=f1)
    third_query_result = conn.execute(third_query)

    model_id = third_query_result.inserted_primary_key
    fourth_query = reviews.update().where(reviews.c.id.in_(ids)).values(model_id=model_id, checked=True)
    fourth_query_result = conn.execute(fourth_query)

except Exception as e:
    print(e)
