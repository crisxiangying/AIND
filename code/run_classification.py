import os
import torch
import torch.nn as nn
from transformers import AdamW
from transformers import BertTokenizer, BertConfig, BertModel
import mymodel_classification, loaddata, file_utils

import torchsnooper
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
            loss, attention_argmax_index, nil_label, predict, predict_vector = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                             attention_mask=attention_mask, labels=label_id)
            # 计算nil分类acc
            acc = (predict == nil_label).item()

            # 反向传播
            loss.backward()
            optimizer.step()
            # epoch 中的 loss 和 acc 累加
            epoch_loss += loss.item()
            epoch_acc += acc
            if i % 200 == 0:
                print("current loss:", epoch_loss / (i + 1), "\t", "current acc:", epoch_acc / ((i + 1)))

    return epoch_loss / count, epoch_acc / count


def predict_input_iterator(num_cands, model, dali_iterator, device, save_predictresult_path_prefix):
    model.eval()
    count = 0
    total_count = 0
    fn_count = 0
    fp_label0_count = 0
    fp_labelnil_count = 0
    with torch.no_grad():
        for i, batch in enumerate(dali_iterator):
            for b in batch:
                count += 1
                input_ids = torch.tensor((b["input_ids"].numpy())[0])
                token_type_ids = torch.tensor((b["segment_ids"].numpy())[0])
                attention_mask = torch.tensor((b["input_mask"].numpy())[0])
                label_id = torch.tensor((b["label_id"].numpy())[0][0]) # tensor(数值)
                mention_guid = (b["mention_guid"].numpy())[0][0] # 数值

                # 需要 LongTensor
                input_ids, token_type_ids, attention_mask, label_id = input_ids.long(), token_type_ids.long(), attention_mask.long(), label_id.long()
                # 迁移到GPU
                input_ids, token_type_ids, attention_mask, label_id = input_ids.to(device), token_type_ids.to(
                    device), attention_mask.to(device), label_id.to(device)

                loss, attention_argmax_index, nil_label, predict, predict_vector = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                                 attention_mask=attention_mask, labels=label_id)

                line = str(mention_guid) + '\n' + str(label_id.cpu().numpy()) + '\n' + str(attention_argmax_index.cpu().numpy())

                # # case tag fake label for test dataset (teacher-student model for domain transfer)
                # write_line = {}
                # write_line['mention_id'] = str(mention_guid)
                # write_line['nil_label'] = str(nil_label.cpu().numpy())
                # write_line['predict_ranking_index'] = str(attention_argmax_index.cpu().numpy())
                # write_line['predict_decision_label'] = str(predict.cpu().numpy())
                # file_utils.writeFile("/home/cxy/bert/pytorch_transformers/teacher_student_classification_model_result/predict_pretrain_3epoches_patent_notnil_299cands.json", json.dumps(write_line))

                # case predict on test dataset (normal classification model)
                if str(label_id.cpu().numpy()) == '0' and str(predict.cpu().numpy()) == '0':
                    total_count += 1
                    fn_count += 1
                    if str(attention_argmax_index.cpu().numpy()) != '0':
                        file_utils.writeFile(save_predictresult_path_prefix + "/fn_predictnot0.json", line)
                    else:
                        file_utils.writeFile(save_predictresult_path_prefix + "/fn_predict0.json", line)
                    print('fn ===== label index: ' + str(label_id.cpu().numpy()) + '  predict id: ' + str(
                        attention_argmax_index.cpu().numpy()) +
                          '  label nil: ' + str((nil_label.cpu().numpy())[0]) + '  predict nil: ' + str(
                        predict.cpu().numpy()))
                if str(label_id.cpu().numpy()) == '0' and str(attention_argmax_index.cpu().numpy()) != '0' and str(predict.cpu().numpy()) == '1':
                    total_count += 1
                    fp_label0_count += 1
                    file_utils.writeFile(save_predictresult_path_prefix + "/fp_label0.json", line)
                    print('fp_label0 ===== label index: ' + str(label_id.cpu().numpy()) + '  predict id: ' + str(
                        attention_argmax_index.cpu().numpy()) +
                          '  label nil: ' + str((nil_label.cpu().numpy())[0]) + '  predict nil: ' + str(
                        predict.cpu().numpy()))
                if str(label_id.cpu().numpy()) == str(num_cands) and str(predict.cpu().numpy()) == '1':
                    total_count += 1
                    fp_labelnil_count += 1
                    file_utils.writeFile(save_predictresult_path_prefix + "/fp_labelnil.json", line)
                    print('fp_labelnil ===== label index: ' + str(label_id.cpu().numpy()) + '  predict id: ' + str(attention_argmax_index.cpu().numpy()) +
                          '  label nil: ' + str((nil_label.cpu().numpy())[0]) + '  predict nil: ' + str(predict.cpu().numpy()))
                if str(label_id.cpu().numpy()) == str(num_cands) and str(predict.cpu().numpy()) == '0':
                    file_utils.writeFile(save_predictresult_path_prefix + "/tn.json", line)
    print('test data: ' + str(count))
    print('total wrong: ' +str(total_count))
    print('fn wrong: ' + str(fn_count))
    print('fp_label0_count wrong: ' + str(fp_label0_count))
    print('fp_labelnil_count wrong: ' + str(fp_labelnil_count))


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
    parser.add_argument('--save_predictresult_path_prefix', type=str,  
                        help='save predict result')
    args = parser.parse_args()
    

    # 模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
    config = BertConfig.from_json_file('./bert-base-uncased/config.json')
    model = mymodel_classification.Bert_MyClassifier_Model(args.num_cands, config)
    model.to(device)

    # 定义优化器和损失函数
    # Prepare optimizer and schedule (linear warmup and decay)
    # model.load_state_dict(torch.load("/home/cxy/bert/pytorch_transformers/newbert99cands_bert_model_notniltraindata_3epoches.pth"))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    # train ------------------------------------------------------------------------------    
    if args.mode == 'train':
        # 冻结BERT所有层参数
        # 固定前面的参数，只训练nil_classifier的参数（nil_classifier.weight, nil_classifier.bias）
        model_dict = model.state_dict()      
        # pretrained_dict = torch.load('/home/cxy/bert/pytorch_transformers/ranking_tripletsloss_100cands/model_traindata_nilloss_3_3epochs.pth')
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        model.load_state_dict(torch.load(args.load_model_path))
        
        for k, v in model.named_parameters():
            if k != 'nil_classifier.0.weight' and k != 'nil_classifier.0.bias' and k != 'nil_classifier.2.weight' and k != 'nil_classifier.2.bias':
                v.requires_grad=False

        # 指定可更新参数，传给optimizer
        optim_param = [{'params': model.nil_classifier.parameters()}]
        optimizer = torch.optim.Adam(optim_param, lr = args.learning_rate, betas = (0.9, 0.999), weight_decay = 1e-5)

        for i in range(args.epochs): # epochs轮数
            # iterator读数据 必须循坏内每轮都重新读iterator，因为一轮之后iterator迭代到最后为空了
            train_iterator = loaddata.get_example_list(args.num_cands, args.data_tfrecord, args.data_tfrecord_idx, '')
            train_loss, train_acc = train_input_iterator(args.num_cands, model, train_iterator, optimizer, device)
            # 输出当前轮次训练结果
            print("train loss: ", train_loss, "\t", "train acc:", train_acc)
            # 每轮都保存当前训练出来的模型
            torch.save(model.state_dict(), args.save_model_path_prefix + str(i+1) + 'epochs.pth')
        
    # predict -------------------------------------------------------------------------------------------
    if args.mode == 'test':
        test_iterator = loaddata.get_example_list(args.num_cands, args.data_tfrecord, args.data_tfrecord_idx, '')
        #
        # 加载保存的模型
        model.load_state_dict(torch.load(args.load_model_path))
        # model.load_state_dict(torch.load("/home/cxy/bert/pytorch_transformers/teacher_student_model_result/newmodel_classifier_withoutpretrain_log_train99cands_3epochs_2epochs.pth"))#, map_location={'cuda:0':'cuda:1'}))
        # 预测
        predict_input_iterator(args.num_cands, model, test_iterator, device, args.save_predictresult_path_prefix)


if __name__ == "__main__":
    main()


