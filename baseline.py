# -*-coding:utf-8-*-
import numpy as np
import time
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import PretrainedLanguageModel, CrossmodalEncoder, PredictModel
from data_loader import load_mosi_pkl, load_mosei_pkl, SentiDataset
from utils import set_random_seed, get_parameter_number, interval_time

start = time.time()

parser = argparse.ArgumentParser(description="some optional arguments")
parser.add_argument("--seed", type=int, default=666, help="random seed")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--num_epoch", type=int, default=200, help="number of epoch")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
parser.add_argument("--pretrained_language_model_name", type=str,
                    choices=["../data/bert/bert-base-uncased", "../data/bert/bert-large-uncased"],
                    default="../data/bert/bert-base-uncased", help="pretrain language model which is used"
                    )
parser.add_argument("--save_model_name", type=str, default="bert-base-uncased", help="save model name")
args = parser.parse_args()

set_random_seed(args.seed)

if "large" in args.pretrained_language_model_name:
    text_dim = 1024
else:
    text_dim = 768
audio_dim = 33
video_dim = 709
embed_dim = 256
fc_dim = 256

if args.dataset == "mosi":
    train_text, train_audio, train_video, train_label = load_mosi_pkl("../data/CMU-MOSI/mosi.pkl", "train")
    valid_text, valid_audio, valid_video, valid_label = load_mosi_pkl("../data/CMU-MOSI/mosi.pkl", "valid")
    test_text, test_audio, test_video, test_label = load_mosi_pkl("../data/CMU-MOSI/mosi.pkl", "test")
elif args.dataset == "mosei":
    train_text, train_audio, train_video, train_label = load_mosei_pkl("../data/CMU-MOSEI/mosei.pkl", "train")
    valid_text, valid_audio, valid_video, valid_label = load_mosei_pkl("../data/CMU-MOSEI/mosei.pkl", "valid")
    test_text, test_audio, test_video, test_label = load_mosei_pkl("../data/CMU-MOSEI/mosei.pkl", "test")
else:
    raise ValueError("The parameter of dataset must be within ['mosi', 'mosei']")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = SentiDataset(train_text, train_audio, train_video, train_label, args.pretrained_language_model_name, device)
valid_data = SentiDataset(valid_text, valid_audio, valid_video, valid_label, args.pretrained_language_model_name, device)
test_data = SentiDataset(test_text, test_audio, test_video, test_label, args.pretrained_language_model_name, device)

train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)

pretrained_language_model = PretrainedLanguageModel(args.pretrained_language_model_name)
crossmodal_encoder = CrossmodalEncoder(text_dim, audio_dim, video_dim, embed_dim)
model = PredictModel(pretrained_language_model, crossmodal_encoder, text_dim, fc_dim)

print("\033[1;35mTotal parameters: {}, Trainable parameters: {}\033[0m".format(*get_parameter_number(model)))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

criterion = torch.nn.L1Loss()
model = model.to(device)


def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in tqdm(iterator):
        input_ids, token_type_ids, attention_mask, audio, vision, label = batch
        input_ids, token_type_ids, attention_mask = input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()

        optimizer.zero_grad()

        output = model(input_ids, token_type_ids, attention_mask, audio, vision)

        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def valid_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(iterator):
            input_ids, token_type_ids, attention_mask, audio, vision, label = batch
            input_ids, token_type_ids, attention_mask = input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()
            output = model(input_ids, token_type_ids, attention_mask, audio, vision)

            loss = criterion(output, label)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test_epoch(model, iterator):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in iterator:
            input_ids, token_type_ids, attention_mask, audio, vision, label = batch
            input_ids, token_type_ids, attention_mask = input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()

            outputs = model(input_ids, token_type_ids, attention_mask, audio, vision)

            logits = outputs.detach().cpu().numpy()
            label_ids = label.detach().cpu().numpy()


            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels

def test_score(model, iterator, use_zero=False):

    preds, y_test = test_epoch(model, iterator)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.round(np.mean(np.absolute(preds - y_test)), decimals=3)
    corr = np.round(np.corrcoef(preds, y_test)[0][1], decimals=3)

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = np.round(f1_score(y_test, preds, average="weighted"), decimals=4)
    acc = np.round(accuracy_score(y_test, preds), decimals=4)

    return acc, f_score, mae, corr

if not os.path.exists("saved_models/prediction/mosi/baseline"):
    os.makedirs("saved_models/prediction/mosi/baseline")
if not os.path.exists("saved_models/prediction/mosei/baseline"):
    os.makedirs("saved_models/prediction/mosei/baseline")

max_valid_loss = 999

for epoch in range(args.num_epoch):

    print("Epoch {}/{}  ".format(epoch+1, args.num_epoch))
    start_time = time.time()

    train_loss = train_epoch(model, train_loader, optimizer, criterion)

    valid_loss = valid_epoch(model, valid_loader, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = interval_time(start_time, end_time)
    print("Epoch: {} | Train Loss: {} | Validation Loss: {} | Time: {}m {}s".format(epoch + 1, train_loss, valid_loss, epoch_mins, epoch_secs))

    if valid_loss < max_valid_loss:
        max_valid_loss = valid_loss
        print('Saving the model ..........')
        torch.save(model, 'saved_models/prediction/{}/baseline/{}_model.pth'.format(args.dataset, args.save_model_name))
        print("Saved the model to saved_models/prediction/{}/baseline/{}_model.pth !".format(args.dataset, args.save_model_name))

    # scheduler.step()

model = torch.load('saved_models/prediction/{}/baseline/{}_model.pth'.format(args.dataset, args.save_model_name))

test_acc, test_f_score, test_mae, test_corr = test_score(model, test_loader, args.pretrained_language_model_name)
print("Accuracy: {}, F1_score: {}, MAE: {}, Corr: {}".format(test_acc, test_f_score, test_mae, test_corr))

print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))
