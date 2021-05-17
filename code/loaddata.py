import os.path
import json
import argparse

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.tfrecord as tfrec

import torch
from subprocess import call

import file_utils


BATCH_SIZE = 1  # batch size per GPU
ITERATIONS = 32
IMAGE_SIZE = 3
NUM_CANDS = 299 #99 #构造的候选集的大小，即真实候选实体的大小，不为nil index加一 case polybert:100(1mention+99cands)
MAX_SEQ_LENGTH = 64 # case polybert:32

class TFRecordPipeline(Pipeline):
    def __init__(self, num_cands, batch_size, num_threads, device_id, tfrecord_path, tfrecord_idx_path):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.TFRecordReader(path = tfrecord_path,
                                        index_path = tfrecord_idx_path,
                                        features = {"input_ids" : tfrec.FixedLenFeature([num_cands, MAX_SEQ_LENGTH], tfrec.int64, -1),
                                                    "segment_ids": tfrec.FixedLenFeature([num_cands, MAX_SEQ_LENGTH], tfrec.int64, -1),
                                                    "input_mask": tfrec.FixedLenFeature([num_cands, MAX_SEQ_LENGTH], tfrec.int64, -1),
                                                    "label_id": tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                                    "mention_guid": tfrec.FixedLenFeature([1], tfrec.int64,  -1)
                                        })

    def define_graph(self):
        inputs = self.input(name="Reader")
        input_ids = inputs["input_ids"]
        segment_ids = inputs['segment_ids']
        input_mask = inputs['input_mask']
        label_id = inputs["label_id"]
        mention_guid = inputs["mention_guid"]
        return (input_ids, segment_ids, input_mask, label_id, mention_guid)

def get_example_list(num_cands, tfrecord, tfrecord_idx, tfile):
    label_range = (0, 999)
    pipes = [TFRecordPipeline(num_cands=num_cands, batch_size=BATCH_SIZE, num_threads=2, device_id=1, tfrecord_path = tfrecord, tfrecord_idx_path = tfrecord_idx)]
    pipes[0].build()
    dali_iter = DALIGenericIterator(pipes, ['input_ids', 'segment_ids', 'input_mask', 'label_id', 'mention_guid'], pipes[0].epoch_size("Reader"))
    return dali_iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate tfrecord idx')
    parser.add_argument('--tfrecord', type=str, metavar='N',
                        help='use which tfrecord data to build tfrecord_idx')
    parser.add_argument('--tfrecord_idx', type=str, metavar='N',
                        help='target tfrecord_idx path')
    args = parser.parse_args()

    tfrecord = "/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords/linkedin_notnil_299cands.tfrecord"
    tfrecord_idx = "/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords/linkedin_notnil_299cands.idx"

    # # 准备tfrecord_idx文件
    tfrecord2idx_script = "tfrecord2idx"
    if not os.path.isfile(tfrecord_idx):
        call([tfrecord2idx_script, tfrecord, tfrecord_idx])


    # 读取tfrecord数据，可以写进文件
    dali_iterator = get_example_list(NUM_CANDS, tfrecord, tfrecord_idx, "/home/cxy/bert/pytorch_transformers/try.json")
    count = 0
    for i, batch in enumerate(dali_iterator):
        for b in batch:
            count += 1
            input_ids = torch.tensor((b["input_ids"].numpy())[0])
            token_type_ids = torch.tensor((b["segment_ids"].numpy())[0])
            attention_mask = torch.tensor((b["input_mask"].numpy())[0])
            label_id = torch.tensor((b["label_id"].numpy())[0][0])
            print(input_ids.size())
            # print(token_type_ids.size())
            # print(attention_mask.size())
            print(label_id)
    print(count)