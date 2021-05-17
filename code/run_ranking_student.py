import torch
import torch.nn as nn
from transformers import AdamW
from transformers import BertTokenizer, BertConfig, BertModel
import loaddata, file_utils, ranking_student_model
from torch.nn import MarginRankingLoss, KLDivLoss

import torchsnooper
import os
import json
import numpy
import argparse

def build_dataset(tfrecord, tfrecord_idx, num_cands):
    data_samples = []
    data_iterator = loaddata.get_example_list(num_cands, tfrecord, tfrecord_idx, '')
    for i, batch in enumerate(data_iterator):
        for b in batch:
            input_ids = torch.tensor((b["input_ids"].numpy())[0])
            token_type_ids = torch.tensor((b["segment_ids"].numpy())[0])
            attention_mask = torch.tensor((b["input_mask"].numpy())[0])
            label_id = torch.tensor((b["label_id"].numpy())[0][0])

            record = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label_id": label_id
            }
            data_samples.append(record)
    return data_samples

def get_embedding_and_scores(embedding_model, src_or_tgt_data, K, device):
    embedding_model.train()

    input_ids, token_type_ids, attention_mask, label_id = src_or_tgt_data['input_ids'][:K, :].long().to(device), src_or_tgt_data['token_type_ids'][:K, :].long().to(device), src_or_tgt_data['attention_mask'][:K, :].long().to(device), src_or_tgt_data['label_id'].long().to(device)
    pooled_output, attention_logits, attention_argmax_index = embedding_model(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask)   

    return pooled_output, attention_logits, attention_argmax_index


def main():
    parser = argparse.ArgumentParser(description='generate tfrecord idx')
    parser.add_argument('--mode', type=str,  
                        help='train or test')
    parser.add_argument('--num_labels', type=int, default=2, 
                        help='label number of classification')
    parser.add_argument('--num_cands', type=int, default=99, 
                        help='create_training_data_tensorflowclassifier传回的数据集，只有前k个cands，不包含添加的nil')
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help='model learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, 
                        help='model weight decay')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='number of run epochs')
    parser.add_argument('--load_model_path', type=str,  
                        help='load model')
    parser.add_argument('--src_tfrecord', type=str,  
                        help='load src domain data')
    parser.add_argument('--src_tfrecord_idx', type=str,  
                        help='load src domain data')
    parser.add_argument('--tgt_tfrecord', type=str,  
                        help='load tgt domain data')
    parser.add_argument('--tgt_tfrecord_idx', type=str,  
                        help='load tgt domain data')
    parser.add_argument('--test_tfrecord', type=str,  
                        help='load test domain data')
    parser.add_argument('--test_tfrecord_idx', type=str,  
                        help='load test domain data')
    parser.add_argument('--save_model_path_prefix', type=str,  
                        help='save model')
    parser.add_argument('--read_fakelabel_file', type=str,  
                        help='save model')
    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # 模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
    config = BertConfig.from_json_file('./bert-base-uncased/config.json')
    model = ranking_student_model.Bert_MyRanker_Model(args.num_cands, config)
    model.to(device)

    # 定义优化器和损失函数
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # train
    if args.mode == 'train':
        model_dict = model.state_dict()
        # ---------------------load trained parameters------------------
        # basebert_pretrained_dict = torch.load("/home/cxy/bert/pytorch_transformers/ranking_bertmodel_tripletsloss_sample99_299cands/newbert99cands_bert_model_notniltraindata_3epoches.pth")
        basebert_pretrained_dict = torch.load(args.load_model_path)
        basebert_pretrained_dict = {k: v for k, v in basebert_pretrained_dict.items()}
        renamed_dict = {}
        for key in basebert_pretrained_dict.keys():
            newkey = key.replace('bert', 'bert_model')
            renamed_dict[newkey] = basebert_pretrained_dict[key]
        #--------------------update parameters
        renamed_dict = {k: v for k, v in renamed_dict.items() if k in model_dict}
        model_dict.update(renamed_dict)
        model.load_state_dict(model_dict)
    
        # 因为标伪标签的测试数据K=299，get_embedding_and_scores时还是只取前99参与训练
        src_data = build_dataset(args.src_tfrecord, args.src_tfrecord_idx, 299) 
        tgt_data = build_dataset(args.tgt_tfrecord, args.tgt_tfrecord_idx, 299)

        # 将tgt的label index 替换为教师模型预测的fake label index
        # 读伪标签存储文件
        lines = file_utils.readFromFileByLine(args.read_fakelabel_file)
        for i in range(0, len(tgt_data)):
            cur_patent_id_fakelabel = json.loads(lines[i])
            tgt_data[i]['label_id'] = torch.tensor(int(cur_patent_id_fakelabel['label_entity_id'])).long().to(device)
        while len(tgt_data) < len(src_data):
            tgt_data.extend(tgt_data)
        tgt_data = tgt_data[: len(src_data)]

        assert len(src_data) == len(tgt_data)

        # 定义两个损失函数
        el_criterion = nn.MarginRankingLoss(margin=1.0)
        kl_criterion = nn.KLDivLoss()
        rank_y = torch.tensor([1.0], device = device)
        for i in range(args.epochs): # epochs轮数
            epoch_loss = 0
            epoch_acc = 0
            for index in range(0, len(src_data)):
                optimizer.zero_grad()
                src_pooled_output, src_attention_logits, src_attention_argmax_index = get_embedding_and_scores(model, src_data[index], args.num_cands, device)
                tgt_pooled_output, tgt_attention_logits, tgt_attention_argmax_index = get_embedding_and_scores(model, tgt_data[index], args.num_cands, device)
                # 计算 src domain EL loss
                src_ranking_loss = []
                pos_score = src_attention_logits[0][0].reshape(1)
                for item in range(1, args.num_cands): #只和index[1,99]前99个负例比较
                    neg_score = src_attention_logits[0][item].reshape(1)
                    margin_loss = el_criterion(pos_score, neg_score, rank_y).unsqueeze(0)
                    src_ranking_loss.append(margin_loss)
                src_ranking_loss = torch.cat(src_ranking_loss)
                src_ranking_loss = torch.mean(src_ranking_loss) 
                # 计算tgt domain kl loss
                # 构造target当前sample的label的 one-hot vector
                target_index = tgt_data[index]['label_id'].to(device)
                target = [0 for i in range(args.num_cands)]
                target.insert(target_index, 1) #在label index下标处增加元素1
                target = numpy.array(target[: args.num_cands]).astype(float)
                target = torch.from_numpy(target).float().to(device)
                # predict 分布 vector
                tgt_attention_probs = nn.Softmax(dim=-1)(tgt_attention_logits).reshape(args.num_cands)
                # predict index
                tgt_attention_argmax_index = tgt_attention_argmax_index.reshape(1)
                # kl loss
                kl_loss = kl_criterion(tgt_attention_probs.log(), target)
                total_loss = src_ranking_loss + kl_loss

                epoch_loss += total_loss.item()
                acc = (tgt_attention_argmax_index == torch.tensor(0).reshape(1).to(device)).item()
                epoch_acc += acc
                total_loss.backward()
                optimizer.step()
                if index % 200 == 0:
                    print(tgt_attention_argmax_index)
                    print(target)
                    print(src_ranking_loss)
                    print(kl_loss)
                    print("current loss:", epoch_loss / (index + 1))
        
            print("train loss: ", epoch_loss/len(src_data), "\t", "train acc:", epoch_acc/len(tgt_data))
            cc = i+1
            torch.save(model.state_dict(), args.save_model_path_prefix + str(i+1) + 'epochs.pth')
            
    

if __name__ == "__main__":
    main()


