# -*-coding:utf-8-*-
import os
import time
import argparse
import math
from tqdm import tqdm, trange

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import PretrainModel
from data_loader import load_lexicon, PretrainDataset, load_emovoxceleb_pkl
from utils import set_random_seed, get_parameter_number, interval_time

start = time.time()

parser = argparse.ArgumentParser(description="some optional arguments")
parser.add_argument("--seed", type=int, default=666, help="random seed")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epoch")
parser.add_argument("--pretrained_language_model_name", type=str,
                    choices=["../data/bert/bert-large-uncased", "../data/bert/bert-large-uncased"],
                    default="../data/bert/bert-large-uncased", help="pretrain language model which is used"
                    )
parser.add_argument("--pretrain_model_path", type=str, choices=["bert-base-uncased", "bert-large-uncased"],
                    default="bert-large-uncased", help="pretrain model")
args = parser.parse_args()

set_random_seed(args.seed)

bs = 32
if "large" in args.pretrained_language_model_name:
    text_dim = 1024
else:
    text_dim = 768
audio_dim = 33
video_dim = 709
embed_dim = 256
fc_dim = 256
to_save_epoch = [1, 5, 10, 20, 50, 100, 150, 200]
print("training on : ", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
raw_text, audio_data, video_data = load_emovoxceleb_pkl("../data/emovoxceleb_test1000.pkl")
lexicon = load_lexicon("data/vader_lexicon.txt")

tokenizer_name = args.pretrained_language_model_name

print("Using {} as backbone pretrain language model and training {} epochs".format(args.pretrain_model_path,
                                                                                   args.num_epochs))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = PretrainDataset(raw_text, audio_data, video_data, lexicon, tokenizer_name, device)

dataloader = DataLoader(dataset=data, batch_size=bs, shuffle=True)

model = PretrainModel(args.pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim)

for param in model.pretrained_language_model.parameters():
    param.requires_grad = False

print("\033[1;35mTotal parameters: {}, Trainable parameters: {}\033[0m".format(*get_parameter_number(model)))

optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

criterion = torch.nn.L1Loss()
model = model.to(device)

angle = 1 / 4
s = 1

def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    t_loss = 0
    s_loss = 0
    for batch in iterator:
        input_ids, token_type_ids, attention_mask, audio, vision, label, index = batch
        input_ids, token_type_ids, attention_mask = input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()
        bs = input_ids.size()[0]
        optimizer.zero_grad()
        output, sim_a, sim_v = model(input_ids, token_type_ids, attention_mask, audio, vision, index, bs)
        sim_sum = [(a + b) / 2 for a, b in zip(sim_a, sim_v)]
        sim_sum = [(1 - sim) * s if sim < 0 else torch.abs(torch.cos(math.pi * (sim + angle))) * s for sim in
                   sim_sum]  # 0<=s<=1/2
        t_l = criterion(output, label)
        s_l = sum(sim_sum) / bs
        loss = t_l + 0.5 * s_l
        loss.backward()
        optimizer.step()

        t_loss += t_l
        s_loss += s_l
        epoch_loss = s_loss + s_loss
    return epoch_loss, t_loss, s_loss


if not os.path.exists("saved_models/pretrain/" + args.pretrain_model_path):
    os.makedirs("saved_models/pretrain/" + args.pretrain_model_path)

for epoch in range(args.num_epochs):
    print("Epoch: {}".format(epoch + 1))
    start_time = time.time()
    dataloader = tqdm(dataloader, total=len(dataloader))
    train_loss, t_loss, s_loss = train_epoch(model, dataloader, optim, criterion)

    if epoch + 1 in to_save_epoch:
        print("Saving pretrained model to saved_models/pretrain/{}/model_epoch{}.pth ...".format(
            args.pretrain_model_path,
            epoch + 1)
        )
        torch.save(model, "saved_models/pretrain/{}/model_epoch{}.pth".format(
            args.pretrain_model_path,
            epoch + 1)
                   )
        print("Saved pretrained model to saved_models/pretrain/{}/model_epoch{}.pth !".format(
            args.pretrain_model_path,
            epoch + 1)
        )
    end_time = time.time()
    epoch_mins, epoch_secs = interval_time(start_time, end_time)
    print("Epoch: {} | Train Loss: {} |t_Loss: {}|s_loss {}|| Time: {}m {}s".format(
        epoch + 1, train_loss,
        t_loss,
        s_loss,
        epoch_mins,
        epoch_secs))

print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))
