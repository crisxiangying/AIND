import os
import sys
import numpy as np

from transformers import BertTokenizer, BertConfig, BertModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torchsnooper

# config = BertConfig.from_json_file('./bert-base-uncased/config.json')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class Bert_MyClassifier_Model(nn.Module):
    def __init__(self, num_cands,config):
        super(Bert_MyClassifier_Model,self).__init__()
        self.num_cands=num_cands #[0，62+1]
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
        self.nil_classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(0.2, True),
            nn.Linear(768, 2),
            nn.Tanh()
        )

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
    ):
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,

        )
        # 取63个候选的output vector
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output) # [num_cands, 768]
        # attention层：输入维度[num_cands,768]，输出维度[1,num_cands]
        attention_logits = self.attention_dense(pooled_output)
        attention_logits = torch.reshape(attention_logits, [1,self.num_cands])
        attention_probs = nn.Softmax(dim=-1)(attention_logits)
        # 计算概率最大的候选实体的index，既为top1 vector对应的下标
        attention_argmax_index = torch.argmax(attention_probs) # 类型：数值， 不是数组

        # 定二分类的groundtruth label
        nil_label = torch.tensor([0])
        if attention_argmax_index == labels:  # 若attention层得到的top1与label一致
            nil_label = torch.tensor([1])
        nil_label = nil_label.to(device)

        # 取top1 vector
        predict_vector = pooled_output[attention_argmax_index]
        # 再加一个全连接，输入维度[1,768]，输出维度[1,2]
        nil_logits = self.nil_classifier(predict_vector)
        nil_logits = torch.reshape(nil_logits, [1,2])
        nil_probs = nn.Softmax(dim=-1)(nil_logits)
        predict = torch.argmax(nil_probs) # 二分类的预测label
        # 注：二分类，
        # 0: top1 != groundtruth(groundtruth!=nil && top1!=groundtruth  ||  ground==nil)，
        # 1: top1 == groundtruth(groundtruth!=nil)

        if labels is not None:
            nil_loss_fct = CrossEntropyLoss()
            nil_loss = nil_loss_fct(nil_logits.view(-1, self.num_labels), nil_label.view(-1))

        return nil_loss, attention_argmax_index, nil_label, predict, predict_vector

