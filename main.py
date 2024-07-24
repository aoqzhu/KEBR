# -*-coding:utf-8-*-
import numpy as np
import time
import os
import argparse
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import math
from metrics import Metrics
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import PretrainModel,PredictModel
from data_loader import load_mosi_pkl, load_mosei_pkl, SentiDataset
from utils import set_random_seed, get_parameter_number, get_flops, interval_time,file_save

start = time.time()

parser = argparse.ArgumentParser(description="some optional arguments")
parser.add_argument("--seed", type=int, default=666, help="random seed")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei","sims"], default="mosi")
parser.add_argument("--num_epoch", type=int, default=200, help="number of epoch")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
#  200 is 0.05+1
parser.add_argument("--pretrain_model_epoch", type=int, default=200, help="pretrain model epoch")
parser.add_argument("--pretrained_language_model_name", type=str,
                    choices=["../data/bert/bert-base-uncased", "../data/bert/bert-large-uncased"],
                    default="../data/bert/bert-base-uncased", help="pretrain language model which is used"
                    )
parser.add_argument("--pretrain_model_path", type=str, choices=["bert-base-uncased", "bert-large-uncased"],
                    default="bert-base-uncased", help="pretrain model")

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
elif args.dataset == "sims":
    train_text, train_audio, train_video, train_label = load_mosei_pkl("../data/CMU-SIMS/sims.pkl", "train")
    valid_text, valid_audio, valid_video, valid_label = load_mosei_pkl("../data/CMU-SIMS/sims.pkl", "valid")
    test_text, test_audio, test_video, test_label = load_mosei_pkl("../data/CMU-SIMS/sims.pkl", "test")
else:
    raise ValueError("The parameter of dataset must be within ['mosi', 'mosei']")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = SentiDataset(train_text, train_audio, train_video, train_label, args.pretrained_language_model_name,
                          device)
valid_data = SentiDataset(valid_text, valid_audio, valid_video, valid_label, args.pretrained_language_model_name,
                          device)
test_data = SentiDataset(test_text, test_audio, test_video, test_label, args.pretrained_language_model_name, device)

train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

pretrained_model = torch.load(
    "saved_models/pretrain/{}/model_epoch{}.pth".format(args.pretrain_model_path, str(args.pretrain_model_epoch)))

model = PredictModel(pretrained_model.pretrained_language_model, pretrained_model.crossmodal_encoder, text_dim, fc_dim)

for param in model.pretrained_language_model.parameters():
    param.requires_grad = True

print("\033[1;35mTotal parameters: {}, Trainable parameters: {}\033[0m".format(*get_parameter_number(model)))

params_group = [{"params": model.pretrained_language_model.parameters(), "lr":0.000005},
                {"params": model.crossmodal_encoder.parameters(),"lr": 0.000005},
                {"params": model.fc1.parameters()},
                {"params": model.fc2.parameters()}]
optimizer = optim.Adam(params_group, lr=args.learning_rate)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

criterion = torch.nn.L1Loss()
model = model.to(device)

angle = 0.4
s = 1
t_list =[]
a_list =[]
v_list =[]
e_list =[]
def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    t_loss = 0
    s_loss = 0
    sa_loss = 0
    sv_loss = 0
    for batch in iterator:
        input_ids, token_type_ids, attention_mask, audio, vision, label = batch
        input_ids, token_type_ids, attention_mask = input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()
        bs = input_ids.size()[0]
        optimizer.zero_grad()
        output, sim_a, sim_v = model(input_ids, token_type_ids, attention_mask, audio, vision)
        sim_aa = [(1 - sim) * s if sim < 0 else torch.abs(torch.cos(math.pi * (sim + angle))) * s for sim in
                  sim_a]  # 0<=s<=1/2
        sim_vv = [(1 - sim) * s if sim < 0 else torch.abs(torch.cos(math.pi * (sim + angle))) * s for sim in
                  sim_v]
        # sim_sum = [(a + b) / 2 for a, b in zip(sim_a, sim_v)]
        # sim_sum = [(1 - sim) * s if sim < 0 else torch.abs(torch.cos(math.pi * (sim + angle))) * s for sim in
        #            sim_sum]  # 0<=s<=1/2
        sim_sum = [
            torch.abs(torch.cos(math.pi * ((a + b) / 2 + angle) * torch.exp(abs(a - b)*0.5))) if a >= 0 and b >= 0 else 1 + torch.abs(
                a) + torch.abs(b) for a, b in
            zip(sim_a, sim_v)]

        t_l = criterion(output, label)
        s_l = sum(sim_sum) / bs
        loss = t_l + s_l
        s_la = sum(sim_aa) / bs
        s_lv = sum(sim_vv) / bs
        loss.backward()
        optimizer.step()
        loss2 = t_l + s_la + s_lv
        s_loss += s_l.item()
        epoch_loss += loss2.item()
        sa_loss += s_la.item()
        t_loss += t_l.item()

        sv_loss += s_lv.item()
    e_list.append(epoch_loss)
    a_list.append(sa_loss)
    v_list.append(sv_loss)
    t_list.append(t_loss)
    return epoch_loss, t_loss, s_loss

def valid_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    t_loss = 0
    s_loss = 0
    with torch.no_grad():
        for batch in iterator:
            input_ids, token_type_ids, attention_mask, audio, vision, label = batch
            input_ids, token_type_ids, attention_mask = input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()
            bs = input_ids.size()[0]
            output, sim_a, sim_v = model(input_ids, token_type_ids, attention_mask, audio, vision)
            sim_sum = [(a + b) / 2 for a, b in zip(sim_a, sim_v)]
            sim_sum = [1 - sim if sim < 0 else torch.abs(torch.cos(math.pi * (sim + angle))) for sim in sim_sum]
            t_l = criterion(output, label)
            s_l = sum(sim_sum) / bs

            t_loss += t_l
            s_loss += s_l
            epoch_loss =t_loss + s_loss
        return epoch_loss, t_loss, s_loss


def test_epoch(model, iterator):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in iterator:
            input_ids, token_type_ids, attention_mask, audio, vision, label = batch
            input_ids, token_type_ids, attention_mask = input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze()

            # outputs = model(input_ids, token_type_ids, attention_mask, audio, vision)
            outputs, _, _ = model(input_ids, token_type_ids, attention_mask, audio, vision)

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
    metrics = Metrics()
    eval_results = metrics.eval_mosei_regression(y_test, preds)
    print(preds[:10], y_test[:10])
    mae = np.round(np.mean(np.absolute(preds - y_test)), decimals=4)
    corr = np.round(np.corrcoef(preds, y_test)[0][1], decimals=4)
    predsbigz = preds
    y_test_bigz = y_test
    preds = preds >= 0
    y_test = y_test >= 0

    preds_z = predsbigz > 0
    y_test_z = y_test_bigz > 0

    f_score = np.round(f1_score(y_test, preds, average="weighted"), decimals=4)
    acc = np.round(accuracy_score(y_test, preds), decimals=4)

    f_score_z = np.round(f1_score(y_test_z, preds_z, average="weighted"), decimals=4)
    acc_z = np.round(accuracy_score(y_test_z, preds_z), decimals=4)


    return acc, f_score, mae, corr, f_score_z, acc_z,eval_results


if not os.path.exists("saved_models/prediction/mosi/our"):
    os.makedirs("saved_models/prediction/mosi/our")
if not os.path.exists("saved_models/prediction/mosei/our"):
    os.makedirs("saved_models/prediction/mosei/our")
if not os.path.exists("saved_models/prediction/sims/our"):
    os.makedirs("saved_models/prediction/sims/our")

max_valid_loss = 999

for epoch in range(args.num_epoch):
    start_time = time.time()
    train_loader = tqdm(train_loader, total=len(train_loader))
    train_loss, t_l, s_l = train_epoch(model, train_loader, optimizer, criterion)
    valid_loss, vt_l, vs_l = valid_epoch(model, valid_loader, criterion)
    # viz.line([[train_loss, valid_loss]], [epoch], win="train", update="append")
    end_time = time.time()
    epoch_mins, epoch_secs = interval_time(start_time, end_time)
    print("\nEpoch: {} | Train Loss: {} | t_l: {} | s_l: {} |Time: {}m {}s".format(epoch + 1, train_loss, t_l,
                                                                                   s_l,
                                                                                   epoch_mins, epoch_secs))

    print("Epoch: {} | Valid Loss:: {} |vt_l: {} |vs_l: {} ".format(epoch + 1, valid_loss, vt_l, vs_l))
    if valid_loss < max_valid_loss:
        max_valid_loss = valid_loss
        print('Saving the model ...')
        torch.save(model, 'saved_models/prediction/{}/our/{}_model.pth'.format(args.dataset, args.pretrain_model_path))
        print("Saved the model to saved_models/prediction/{}/our/{}_model.pth !".format(args.dataset,
                                                                                        args.pretrain_model_path))
    scheduler.step()

model = torch.load('saved_models/prediction/{}/our/{}_model.pth'.format(args.dataset, args.pretrain_model_path))

test_acc, test_f_score, test_mae, test_corr, f_score_z, acc_z, result= test_score(model, test_loader,
                                                                               args.pretrain_model_path)
log = 'Has0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
      'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s' \
      '------------------------------------------' % (result['Has0_acc_2'],
                                                      result['Has0_F1_score'],
                                                      result['Non0_acc_2'], result['Non0_F1_score'],
                                                      result['Mult_acc_5'],
                                                      result['Mult_acc_7'], result['MAE'],
                                                      result['Corr'])
print(log)
file_save(e_list,t_list,a_list,v_list,'Loss')
print("Accuracy: {}, F1_score: {}, MAE: {}, Corr: {},f_score_z:{},acc_z:{}".format(test_acc, test_f_score, test_mae,
                                                                                       test_corr, f_score_z, acc_z))

print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))
print("train angle==",angle)
