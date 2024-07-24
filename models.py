# -*-coding:utf-8-*- 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
import math
import numpy as np

from torch.nn import Parameter

class PretrainedLanguageModel(nn.Module):
    def __init__(self, pretrained_language_model_name):
        super(PretrainedLanguageModel, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_language_model_name)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask).last_hidden_state
        return output

def cosine_similarity(vec1, vec2):

    cosine_similarities = []


    for i in range(vec1.size(0)):

        a_batch = vec1[i]
        b_batch = vec2[i]
        a_batch_flat = a_batch.flatten()
        b_batch_flat = b_batch.flatten()

        norm_A = torch.norm(a_batch_flat)
        norm_B = torch.norm(b_batch_flat)

        dot_product = torch.dot(a_batch_flat, b_batch_flat)

        cosine_similarity = dot_product / (norm_A * norm_B)
        cosine_similarities.append(cosine_similarity)
    return cosine_similarities

def matrix_alignment(matrix1, matrix2, matrix3):
    # Calculate the number of rows and columns for both matrices
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    rows3, cols3 = matrix3.shape
    max_cols = cols1 if cols1 > cols2 and cols1 > cols3 else (cols2 if cols2 > cols3 else cols3)
    max_rows = rows1 if rows1 > rows2 and rows1 > rows3 else (rows2 if rows2 > rows3 else rows3)
    if(max_cols//cols1-1)>0:
        matrix1 = matrix1.repeat(1, max_cols//cols1)
    if(max_cols//cols2-1)>0:
        matrix2 = matrix2.repeat(1, max_cols // cols2)
    if(max_cols//cols3-1)>0:
        matrix3 = matrix2.repeat(1, max_cols // cols3)
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    rows3, cols3 = matrix3.shape
    max_cols = cols1 if cols1 > cols2 and cols1 > cols3 else (cols2 if cols2 > cols3 else cols3)
    matrix1 = F.pad(matrix1, (0, max_cols - cols1), mode='constant', value=0)
    matrix2 = F.pad(matrix2, (0, max_cols - cols2), mode='constant', value=0)
    matrix3 = F.pad(matrix3, (0, max_cols - cols3), mode='constant', value=0)
    return matrix1, matrix2, matrix3

# ---------------------------------------------------------------------------------------------------------------------
# non-verbal information injection, i.e., multimodal fusion
class CrossModalAttention(nn.Module):
    def __init__(self, modality1_dim, modality2_dim, embed_dim, attn_dropout=0.5):
        super(CrossModalAttention, self).__init__()
        self.modality1_dim = modality1_dim
        self.modality2_dim = modality2_dim
        self.embed_dim = embed_dim
        self.modality1_ln = nn.LayerNorm(self.modality1_dim)
        self.modality2_ln = nn.LayerNorm(self.modality2_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scaling = self.embed_dim ** -0.5
        self.proj_modality1 = nn.Linear(self.modality1_dim, self.embed_dim)
        self.proj_modality2_k = nn.Linear(self.modality2_dim, self.embed_dim)
        self.proj_modality2_v = nn.Linear(self.modality2_dim, self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.modality1_dim)

    def forward(self, modality1, modality2):
        q = self.proj_modality1(self.modality1_ln(modality1))
        k = self.proj_modality2_k(self.modality2_ln(modality2))
        v = self.proj_modality2_v(self.modality2_ln(modality2))
        attention = F.softmax(torch.bmm(q, k.permute(0, 2, 1)) * self.scaling, dim=-1)
        context = torch.bmm(attention, v)
        output = self.proj(context)
        # output = self.attn_dropout(output)
        # output = output + self.modality1_ln(modality1)
        # output = output + modality1
        return output


class CrossmodalEncoderLayer(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, embed_dim, attn_dropout=0.5):
        super(CrossmodalEncoderLayer, self).__init__()
        self.cma_a = CrossModalAttention(text_dim, audio_dim, embed_dim)
        self.cma_v = CrossModalAttention(text_dim, video_dim, embed_dim)
        self.layernorm = nn.LayerNorm(text_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc = nn.Linear(text_dim, text_dim)

    def forward(self, text, audio, video):
        # output = self.cma_a(text, audio) + self.cma_v(text, video) + self.layernorm(text)
        output = self.cma_a(text, audio) + self.cma_v(text, video) + text
        residual = output
        output = self.fc(self.layernorm(output))
        output = self.attn_dropout(output)
        output = output + residual
        return output


class CrossmodalEncoder(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, embed_dim, num_layers=4, attn_dropout=0.5):
        super(CrossmodalEncoder, self).__init__()
        self.encoderlayer = CrossmodalEncoderLayer(text_dim, audio_dim, video_dim, embed_dim, attn_dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        for layer in range(self.num_layers):
            new_layer = self.encoderlayer
            self.layers.append(new_layer)

    def forward(self, text, audio, video):
        output = text
        for layer in self.layers:
            output = layer(output, audio, video)
        return output


class PretrainModelCopy(nn.Module):
    """
    Pretrain model with crossmodal attention 在特征融合走计算相似度
    """

    def __init__(self, pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim, num_layers=4,
                 attn_dropout=0.5, fc_dropout=0.5):
        super(PretrainModelCopy, self).__init__()
        self.pretrained_language_model = PretrainedLanguageModel(pretrained_language_model_name)
        self.crossmodal_encoder = CrossmodalEncoder(text_dim, audio_dim, video_dim, embed_dim, num_layers, attn_dropout)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def feature_extractor(self, input_ids, token_type_ids, attention_mask, audio, video):
        output = self.pretrained_language_model(input_ids, token_type_ids, attention_mask)
        output = self.crossmodal_encoder(output, audio, video)
        # output = torch.mean(output, dim=1)
        return output

    def fitter(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def cosine_similarity(self, vec1, vec2):
        cosine_similarities = []

        for i in range(vec1.size(0)):
            a_batch = vec1[i]
            b_batch = vec2[i]
            a_batch_flat = a_batch.flatten()
            b_batch_flat = b_batch.flatten()
            norm_A = torch.norm(a_batch_flat)
            norm_B = torch.norm(b_batch_flat)

            dot_product = torch.dot(a_batch_flat, b_batch_flat)

            cosine_similarity = dot_product / (norm_A * norm_B)
            cosine_similarities.append(cosine_similarity)
        return cosine_similarities

    """
    A multimodel fusion model that combines the output of a pre-trained language model and two multimodal models.

    """

    def __init__(self, pretrained_language_model_name, text_dim, audio_dim, video_dim, embed_dim, fc_dim, num_layers=4,
                 attn_dropout=0.5, fc_dropout=0.5):
        super(PretrainModel, self).__init__()
        self.pretrained_language_model = PretrainedLanguageModel(pretrained_language_model_name)
        self.crossmodal_encoder = CrossmodalEncoder(text_dim, audio_dim, video_dim, embed_dim, num_layers, attn_dropout)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def feature_extractor(self, input_ids, token_type_ids, attention_mask, audio, video):
        output = self.pretrained_language_model(input_ids, token_type_ids, attention_mask)
        output = self.crossmodal_encoder(output, audio, video)
        return output

    def fitter(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, token_type_ids, attention_mask, audio, video, index, bs):
        output = self.feature_extractor(input_ids, token_type_ids, attention_mask, audio, video)
        output_n = output
        output = self.fitter(output[torch.tensor(range(bs)), index, :])

        audio = torch.flatten(audio, 1)
        video = torch.flatten(video, 1)
        output_n = torch.flatten(output_n, 1)

        audio = F.normalize(audio, dim=1, out=None)
        video = F.normalize(video, dim=1, out=None)
        output_n = F.normalize(output_n, dim=1, out=None)

        output_n, audio, video = matrix_alignment(output_n, audio, video)

        sim_a = cosine_similarity(audio, output_n)
        sim_v = cosine_similarity(video, output_n)

        return output, sim_a, sim_v

class LanguageModelClassifier(nn.Module):
    def __init__(self, pretrained_language_model, text_dim, fc_dim, fc_dropout=0.5):
        super(LanguageModelClassifier, self).__init__()
        self.pretrained_language_model = pretrained_language_model
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.pretrained_language_model(input_ids, token_type_ids, attention_mask)
        output = torch.mean(output, dim=1)
        # output = output[:, 0, :]
        output = self.predict(output)
        return output


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        label_tmp = label
        label_tmp = label_tmp.to(torch.int64)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label_tmp.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output


# ---------------------------------------------------------------------------------------------------------------------
class PredictModel(nn.Module):
    def __init__(self, pretrained_language_model, crossmodal_encoder, text_dim, fc_dim, fc_dropout=0.5):
        super(PredictModel, self).__init__()
        self.pretrained_language_model = pretrained_language_model
        self.crossmodal_encoder = crossmodal_encoder
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(text_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, input_ids, token_type_ids, attention_mask, audio, video):
        output = self.pretrained_language_model(input_ids, token_type_ids, attention_mask)
        output = self.crossmodal_encoder(output, audio, video)
        output_n = output
        output = torch.mean(output, dim=1)

        output_n = torch.flatten(output_n, 1)
        audio = torch.flatten(audio, 1)
        video = torch.flatten(video, 1)

        audio = F.normalize(audio, dim=1, out=None)
        video = F.normalize(video, dim=1, out=None)
        output_n = F.normalize(output_n, dim=1, out=None)
        # weight = F.normalize(torch.transpose(self.fc2.weight, 0, 1))
        output_n, audio, video = matrix_alignment(output_n, audio, video)

        sim_a = cosine_similarity(audio, output_n)
        sim_v = cosine_similarity(video, output_n)
        # output = output[:,0,:]
        output = self.predict(output)

        return output, sim_a, sim_v



