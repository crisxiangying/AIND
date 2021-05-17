import os
import sys
import numpy as np

from transformers import BertTokenizer, BertConfig, BertModel
import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss

import torchsnooper

# config = BertConfig.from_json_file('./bert-base-uncased/config.json')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Bert_MyRanker_Model(nn.Module):
    def __init__(self, num_cands, config):
        super(Bert_MyRanker_Model,self).__init__()
        self.num_cands=num_cands #[0，62+1)
        self.num_labels = 2
        self.bert_model = BertModel.from_pretrained('./bert-base-uncased', config=config)
        # self.bert_model.to(device)
        self.dropout = nn.Dropout(p = 0.1)
        self.attention_dense = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(0.2, True),
            nn.Linear(768, 1),
            nn.Tanh()
        )
        # self.attention_dense = nn.Linear(768, 1).cuda() #将embedding映射到分数值，learn2rank
        self.nil_classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(0.2, True),
            nn.Linear(768, 2),
            nn.Tanh()
        )
        # self.nil_classifier = nn.Linear(768, 2).cuda() #这里需要初始化这个，要不然在classification时load参数找不到这个参数


        # self.init_weights()

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
        # print('pooled_output shape:--------')
        # print(pooled_output.shape)
        pooled_output = self.dropout(pooled_output) # [63, 768]
        # attention层：加全连接，输入维度[63,768]，输出维度[1,63]
        attention_logits = self.attention_dense(pooled_output)
        # print('attention_logits shape:--------')
        # print(attention_logits.shape)
        attention_logits = torch.reshape(attention_logits, [1,self.num_cands])
        # print('resize attention_logits shape:--------')
        # print(attention_logits.shape)
        # print('attention logits:--------------')
        # print(attention_logits)
        attention_probs = nn.Softmax(dim=-1)(attention_logits)
        # 计算概率最大的候选实体的index，既为top1 vector对应的下标
        attention_argmax_index = torch.argmax(attention_probs) # 类型：数值， 不是数组

        return pooled_output, attention_logits, attention_argmax_index
