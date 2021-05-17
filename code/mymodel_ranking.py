import os
import sys
import numpy as np

from transformers import BertTokenizer, BertConfig, BertModel
import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss

import torchsnooper

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Bert_MyRanker_Model(nn.Module):
    def __init__(self, num_cands, config):
        super(Bert_MyRanker_Model,self).__init__()
        self.num_cands=num_cands #[0，62+1)
        self.num_labels = 2
        self.bert_model = BertModel.from_pretrained('./bert-base-uncased', config=config)
        self.dropout = nn.Dropout(p = 0.1)
        self.attention_dense = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(0.2, True),
            nn.Linear(768, 1),
            nn.Tanh()
        )
        self.nil_classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(0.2, True),
            nn.Linear(768, 2),
            nn.Tanh()
        )#这里需要初始化这个，要不然在classification时load参数找不到这个参数


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None, # 传入范围0~100, 数值型。但是实际上只有100个候选实体，所以下面算得的attention_argmax_index范围是0~99
    ):
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,

        )
        # 取63个候选的output vector
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output) # [63, 768]
        # attention层：输入维度[63,768]，输出维度[1,63]
        attention_logits = self.attention_dense(pooled_output)
        attention_logits = torch.reshape(attention_logits, [1,self.num_cands])
        attention_probs = nn.Softmax(dim=-1)(attention_logits)
        # 计算概率最大的候选实体的index，既为top1 vector对应的下标
        attention_argmax_index = torch.argmax(attention_probs) # 类型：数值， 不是数组

        ranking_label = torch.tensor([labels]).to(device)

        criterion = MarginRankingLoss(margin=1.0)
        rank_y = torch.tensor([1.0], device = device)
        # max_index = 0
        # #get top 99 cand index which embedding is most similar to pooled_output[0] 
        # index_cosinsimscore_dict = {}
        # index_cosinsimscore_dict[0] = 1
        # for i in range(1, self.num_cands):
        #     index_cosinsimscore_dict[i] = torch.cosine_similarity(pooled_output[0], pooled_output[i], dim=0) #dim不确定取0还是1
        # # print(pooled_output[0].shape)
        # reversesorted_index_cosinsimscore_dict = sorted(index_cosinsimscore_dict.items(),key = lambda x:x[1],reverse = True)
        # # print('\n----------')


        instance_mean_loss = []
        pos_score = self.attention_dense(pooled_output[0])
        for i in range(1, self.num_cands): #只和index[1,99]前99个负例比较
            neg_score = self.attention_dense(pooled_output[i])
            # neg_index = reversesorted_index_cosinsimscore_dict[i][0]
            # # print('-----neg_index: ' + str(neg_index))
            # neg_score = self.attention_dense(pooled_output[neg_index])
            margin_loss = criterion(pos_score, neg_score, rank_y).unsqueeze(0)
            instance_mean_loss.append(margin_loss)
        instance_mean_loss = torch.cat(instance_mean_loss)
        loss = torch.mean(instance_mean_loss)

        attention_argmax_score = self.attention_dense(pooled_output[attention_argmax_index])
        return loss, attention_argmax_index, attention_argmax_score

