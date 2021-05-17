import torch
import torch.nn as nn
from transformers import AdamW
from transformers import BertTokenizer, BertConfig, BertModel
import mymodel_ranking, loaddata, file_utils

import torchsnooper
import os
import json
import argparse

def train_input_iterator(num_cands, model, dali_iterator, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    for i, batch in enumerate(dali_iterator):
        for b in batch:
            count += 1
            input_ids = torch.tensor((b["input_ids"].numpy())[0])
            token_type_ids = torch.tensor((b["segment_ids"].numpy())[0])
            attention_mask = torch.tensor((b["input_mask"].numpy())[0])
            label_id = torch.tensor((b["label_id"].numpy())[0][0])

            # 需要 LongTensor
            input_ids, token_type_ids, attention_mask, label_id = input_ids.long(), token_type_ids.long(), attention_mask.long(), label_id.long()

            # 梯度清零
            optimizer.zero_grad()
            # 迁移到GPU
            input_ids, token_type_ids, attention_mask, label_id = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), label_id.to(device)
            loss, attention_argmax_index, attention_argmax_score = model(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, labels=label_id)

            # 计算cands ranking acc
            acc = ((torch.tensor([attention_argmax_index]).to(device)) == (torch.tensor([label_id]).to(device))).item()

            # 反向传播
            loss.backward()
            optimizer.step()
            # epoch 中的 loss 和 acc 累加
            epoch_loss += loss.item()
            epoch_acc += acc
            if i % 200 == 0:
                print(attention_argmax_index)
                print("current loss:", epoch_loss / (i + 1), "\t", "current acc:", epoch_acc / ((i + 1)))
    return epoch_loss / count, epoch_acc / count


def predict_input_iterator(num_cands, model, dali_iterator, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    total_count = 0
    label63_count = 0
    with torch.no_grad():
        for i, batch in enumerate(dali_iterator):
            for b in batch:
                count += 1
                input_ids = torch.tensor((b["input_ids"].numpy())[0])
                token_type_ids = torch.tensor((b["segment_ids"].numpy())[0])
                attention_mask = torch.tensor((b["input_mask"].numpy())[0])
                label_id = torch.tensor((b["label_id"].numpy())[0][0])
                mention_guid = (b["mention_guid"].numpy())[0][0]  # 数值

                # 需要 LongTensor
                input_ids, token_type_ids, attention_mask, label_id = input_ids.long(), token_type_ids.long(), attention_mask.long(), label_id.long()
                # 迁移到GPU
                input_ids, token_type_ids, attention_mask, label_id = input_ids.to(device), token_type_ids.to(
                    device), attention_mask.to(device), label_id.to(device)

                loss, attention_argmax_index, attention_argmax_score = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                                 attention_mask=attention_mask, labels=label_id)
                
                # # # write all predicted samples:
                # write_line = {}
                # write_line['mention_id'] = str(mention_guid)
                # write_line['label_entity_id'] = str(attention_argmax_index.cpu().numpy()) # fake label
                # file_utils.writeFile("/home/cxy/bert/pytorch_transformers/ranking_linkedin_pretrain/onesteppretrain_ranking3epochs_linkedin_fakelabel.json", json.dumps(write_line))
                
                # # # write predicted wrong cases:
                if str(attention_argmax_index.cpu().numpy())!= '0':
                    line = str(mention_guid) + '\n' + str(label_id.cpu().numpy()) + '\n' + str(attention_argmax_index.cpu().numpy())
                    # print(str(attention_argmax_index.cpu().numpy()))
                    file_utils.writeFile("/home/cxy/bert/pytorch_transformers/patent_teacher_student_model_result/predict_withoutpretrain_log_studentmodel_finetune_343.json", line)
                    total_count += 1

                # # # write predicting results(including right and wrong) according to label type(nil/0)
                # line = str(mention_guid) + '\n' + str(label_id.cpu().numpy()) + '\n' + str(attention_argmax_index.cpu().numpy()) + '\n' + str(attention_argmax_score.cpu().numpy())
                
                # # # using for threshold baseline result analysis: 
                # # case label nil
                # file_utils.writeFile("/home/cxy/bert/pytorch_transformers/ranking_bertmodel_tripletsloss_sample99_299cands/threshold_result_testsamples/newbert99cands_ranking_3epoches_nil.json", line)
                # # end

                # # case label 0
                # if str(attention_argmax_index.cpu().numpy())!= '0':
                #     total_count += 1
                #     file_utils.writeFile("/home/cxy/bert/pytorch_transformers/ranking_bertmodel_tripletsloss_sample99_299cands/threshold_result_testsamples/newbert99cands_ranking_3epoches_notnil_predictwrong.json", line)
                # else:
                #     file_utils.writeFile("/home/cxy/bert/pytorch_transformers/ranking_bertmodel_tripletsloss_sample99_299cands/threshold_result_testsamples/newbert99cands_ranking_3epoches_notnil_predictright.json", line)
                # # end
               
    print('all wrong cases that predicted index != 0: ' + str(total_count))


def main():
    parser = argparse.ArgumentParser(description='generate tfrecord idx')
    parser.add_argument('--mode', type=str,  
                        help='train or test')
    parser.add_argument('--num_labels', type=int, default=2, 
                        help='label number of classification')
    parser.add_argument('--num_cands', type=int, default=299, 
                        help='create_training_data_tensorflowclassifier传回的数据集，只有前k个cands，不包含添加的nil')
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help='model learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, 
                        help='model weight decay')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='number of run epochs')
    parser.add_argument('--load_model_path', type=str,  
                        help='load model')
    parser.add_argument('--data_tfrecord', type=str,  
                        help='load data')
    parser.add_argument('--data_tfrecord_idx', type=str,  
                        help='load data')
    parser.add_argument('--save_model_path_prefix', type=str,  
                        help='save model')
    args = parser.parse_args()

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # 模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
    config = BertConfig.from_json_file('./bert-base-uncased/config.json')
    model = mymodel_ranking.Bert_MyRanker_Model(args.num_cands, config)
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
        # basebert_pretrained_dict = torch.load("/home/cxy/bert/zeshel_master/pretrain_wiki2company/BERT_companylm_expandpretraindata/pytorch_model.bin")
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
        
        # # 加载数据集
        # train_tfrecord = "/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords/train_notnil_99cands.tfrecord"
        # train_tfrecord_idx = "/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords/train_notnil_99cands.idx"
        
        # # model.load_state_dict(torch.load('/home/cxy/bert/pytorch_transformers/new_model_notniltraindata_rankingloss_3epoches.pth'))
        for i in range(args.epochs): # epochs轮数
            # iterator读数据 必须循坏内每轮都重新读iterator，因为一轮之后iterator迭代到最后为空了
            train_iterator = loaddata.get_example_list(args.num_cands, args.data_tfrecord, args.data_tfrecord_idx, '')
            train_loss, train_acc = train_input_iterator(args.num_cands, model, train_iterator, optimizer, device)
            # 输出当前轮次训练结果
            print("train loss: ", train_loss, "\t", "train acc:", train_acc)
            torch.save(model.state_dict(), args.save_model_path_prefix + str(i+1) + 'epochs.pth')
      


    # predict
    if args.mode == 'test':
        # test_tfrecord = "/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords/patent_notnil_299cands.tfrecord"
        # test_tfrecord_idx = "/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords/patent_notnil_299cands.idx"
        test_iterator = loaddata.get_example_list(args.num_cands, args.data_tfrecord, args.data_tfrecord_idx, '')
        #
        # 加载保存的模型
        # model.load_state_dict(torch.load("/home/cxy/bert/pytorch_transformers/teacher_student_model_result/withoutpretrain_log_studentmodel_4epoches.pth"))
        model.load_state_dict(torch.load(args.load_model_path))
        # 预测
        predict_input_iterator(args.num_cands, model, test_iterator, device)


if __name__ == "__main__":
    main()


