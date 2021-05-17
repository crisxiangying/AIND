# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create training data TF examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import random
import tensorflow as tf
from bertbase import tokenization

import utils

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("nil_id", "99999999",  #具体值待改
                    "Id of NIL entity")

flags.DEFINE_string("entities_file", None,
                    "Path to entities json file.")

flags.DEFINE_string("mentions_file", None,
                    "Path to mentions json file.")

flags.DEFINE_string("candidates_file", None,
                    "Path to candidates file.")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "is_training", True,
    "Training data")

flags.DEFINE_bool(
    "split_by_domain", False,
    "Split output TFRecords by domain.")

flags.DEFINE_integer("max_seq_length", 64, "Maximum sequence length.")

# create训练数据代码中，num_cands需设为64 /300，即最后一个元素留给nil。
# 但是最后返回的是[0,len(num_cands)),即前63/299，最后一个没有返回写入tfrecord。所以classifier代码读进的只有前63/299个有对应候选实体的。
# 所以create时63+1的主要意义是：把nil的数据babel标为63
flags.DEFINE_integer("num_cands", 300, "Number of entity candidates.")  # 64 / 300 / 100

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")


class TrainingInstance(object):
    """A single set of features of data."""

    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 mention_id,
                 mention_guid,
                 cand_guids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.mention_id = mention_id
        self.mention_guid = mention_guid
        self.cand_guids = cand_guids

    def __str__(self):
        s = ""
        s += "input_ids: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.input_ids[:FLAGS.max_seq_length]]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids[:FLAGS.max_seq_length]]))
        s += "input_mask: %s\n" % (" ".join([str(x) for x in self.input_mask[:FLAGS.max_seq_length]]))
        s += "mention_id: %s\n" % (" ".join([str(x) for x in self.mention_id[:FLAGS.max_seq_length]]))
        s += "label_id: %d\n" % self.label_id
        s += "mention_guid: %d\n" % self.mention_guid
        s += "cand_guids: %d\n" % self.cand_guids
        s += "\n"
        return s


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    num_cands, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    # print('len of instances: ' +str(len(instances)))

    for (inst_index, instance) in enumerate(instances):

        input_ids = instance.input_ids
        input_mask = instance.input_mask
        segment_ids = instance.segment_ids
        mention_id = instance.mention_id
        label_id = instance.label_id
        mention_guid = instance.mention_guid
        cand_guids = instance.cand_guids
        # label_id一定都是0或者63
        assert len(input_ids) == max_seq_length * (num_cands-1) ## new change
        assert len(input_mask) == max_seq_length * (num_cands-1) ## new change
        assert len(segment_ids) == max_seq_length * (num_cands-1) ## new change
        assert len(mention_id) == max_seq_length * (num_cands-1) ## new change

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["mention_id"] = create_int_feature(mention_id)
        features["label_id"] = create_int_feature([label_id])
        features["mention_guid"] = create_int_feature([mention_guid])
        features["cand_guids"] = create_int_feature(cand_guids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens[:FLAGS.max_seq_length]]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values[:FLAGS.max_seq_length]])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


# 已改
# 读原始的数据文件，调用create_instances_from_document()生成training instances， 写进TFrecord
def create_training_instances(entity_files, mentions_files, tokenizer, max_seq_length,
                              rng, is_training=True):
    """Create `TrainingInstance`s from raw text."""

    entities = {}
    for input_file in entity_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                line = line.strip()
                if not line:
                    break
                line = json.loads(line)
                entities[line['entity_id']] = line  #key:entity_id, value:line对应的jso
    mentions = []
    for input_file in mentions_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                line = line.strip()
                if not line:
                    break
                line = json.loads(line)
                mentions.append(line)

    candidates = {}
    with tf.gfile.GFile(FLAGS.candidates_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            line = line.strip()
            if not line:
                break
            line = json.loads(line)
            candidates[line['mention_id']] = line['candidates']

    vocab_words = list(tokenizer.vocab.keys())

    instances = []

    for i, mention in enumerate(mentions): #对mentions中的每一个mention

        
        # 一个instance 是一个训练的输入，包括该mention的所有candidates编号、groundtruth candi编号
        instance = create_instances_from_document(
            mention, entities, candidates, tokenizer, max_seq_length,
            vocab_words, rng, is_training=is_training)

        if instance:
            instances.append(instance)
        # print(str(i))
        if i > 0 and i % 100 == 0:
            tf.logging.info("Instance: %d" % i)

    if is_training:
        rng.shuffle(instances)

    # print('return create_training_instances()')
    return instances


# 未改
def get_context_tokens(context_tokens, start_index, end_index, max_tokens, tokenizer):
    start_pos = start_index - max_tokens
    if start_pos < 0:
        start_pos = 0
    prefix = ' '.join(context_tokens[start_pos: start_index])
    suffix = ' '.join(context_tokens[end_index + 1: end_index + max_tokens + 1])
    prefix = tokenizer.tokenize(prefix)
    suffix = tokenizer.tokenize(suffix)
    mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index + 1]))

    assert len(mention) < max_tokens

    remaining_tokens = max_tokens - len(mention)
    half_remaining_tokens = int(math.ceil(1.0 * remaining_tokens / 2))

    mention_context = []

    if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
        prefix_len = half_remaining_tokens
    elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
        prefix_len = remaining_tokens - len(suffix)
    elif len(prefix) < half_remaining_tokens:
        prefix_len = len(prefix)

    if prefix_len > len(prefix):
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]

    mention_context = prefix + mention + suffix
    mention_start = len(prefix)
    mention_end = mention_start + len(mention) - 1
    mention_context = mention_context[:max_tokens]

    assert mention_start <= max_tokens
    assert mention_end <= max_tokens

    return mention_context, mention_start, mention_end


def pad_sequence(tokens, max_len):
    assert len(tokens) <= max_len
    return tokens + [0] * (max_len - len(tokens))

# 对于传入的一个mention string，生成它的instance，包括该mention的所有candidates编号、groundtruth candi编号
def create_instances_from_document(
        mention, all_entities, candidates, tokenizer, max_seq_length,
        vocab_words, rng, is_training=True):
    # all_entities应该是 entity-id：entity-text
    """Creates `TrainingInstance`s for a single document."""

    # Account for [CLS], [SEP], [SEP]  token_a 和 token_b 实际可以占的长度
    max_num_tokens = max_seq_length - 3

    # context_document_id = mention['context_document_id'] #该mention所在的document的id
    label_entity_id = mention['label_entity_id']

    # mention
    mention_text_tokenized = tokenizer.tokenize(mention['text'])

    mention_id_in_dataset = mention['mention_id']
    assert mention_id_in_dataset in candidates
    cand_entity_ids = candidates[mention_id_in_dataset]
    # 现在这种条件处理有点粗暴，会损失一些数据，但是如果一个样本的cands就只有一个label是不是也没啥训练学习的意义
    # if not cand_entity_ids or len(cand_entity_ids)==0 or (len(cand_entity_ids)==1 and cand_entity_ids[0]==label_entity_id):
    #     # print("filter id : ",mention_id_in_dataset)
    #     return None

    

    if not cand_entity_ids or len(cand_entity_ids)==0:
        # print("filter id : ",mention_id_in_dataset)
        return None

    # 候选实体    print('mention text get..')
    num_cands = FLAGS.num_cands

    label_id = None
    #对于非training：
    if not is_training and label_entity_id == FLAGS.nil_id:
        cand_entity_ids = [cand for cand in cand_entity_ids if cand != label_entity_id]
        while len(cand_entity_ids) < num_cands:
            cand_entity_ids.extend(cand_entity_ids)
        cand_entity_ids = cand_entity_ids[:num_cands]
        cand_entity_ids[num_cands-1] = FLAGS.nil_id
        label_id = num_cands - 1
        print(len(cand_entity_ids))
        print(num_cands)
        assert len(cand_entity_ids) == num_cands
        ### 对测试集，把通过了cands generation的mentions记下来，baseline也用这些做，
        ### 不在整个notnil mention数据集上做
        # utils.writeFile("/home/cxy/bert/data/wiki/expandNIL_putcasback/testdata/normalized_nontnil_paper_mentions.json", json.dumps(mention))
    if not is_training and label_entity_id != FLAGS.nil_id and label_entity_id not in cand_entity_ids: #非NIL且candidate中没有groundtruth时，该instance返回None
        # print("filter id : ", mention_id_in_dataset)
        return None
    if not is_training and label_entity_id != FLAGS.nil_id and label_entity_id in cand_entity_ids:
        ###
        if len(cand_entity_ids)==1 and cand_entity_ids[0]==label_entity_id:   # 若cands只有一个元素，就是label，则用label填满所有cands
            while len(cand_entity_ids) < num_cands:
                cand_entity_ids.extend(cand_entity_ids)
        else:   #否则，用所有非label的cands填满cands
            cand_entity_ids = [cand for cand in cand_entity_ids if cand != label_entity_id]
        ###
            while len(cand_entity_ids) < num_cands:
                cand_entity_ids.extend(cand_entity_ids)
        cand_entity_ids.insert(0, label_entity_id)  # cands的index=0处替换为label
        cand_entity_ids = cand_entity_ids[:num_cands]
        cand_entity_ids[num_cands - 1] = FLAGS.nil_id
        label_id = None
        # for i, entity in enumerate(cand_entity_ids):
        #     if entity == label_entity_id:
        #         # assert label_id == None
        #         label_id = i
        label_id = 0
        # assert label_id == 0
        assert len(cand_entity_ids) == num_cands
        ### 对测试集，把通过了cands generation的mentions记下来，baseline也用这些做，
        ### 不在整个notnil mention数据集上做
        # utils.writeFile("/home/cxy/bert/data/wiki/expandNIL_putcasback/testdata/normalized_nontnil_paper_mentions.json", json.dumps(mention))

    # 对于training：
    if is_training and label_entity_id == FLAGS.nil_id:
        print(mention_id_in_dataset)
        cand_entity_ids = [cand for cand in cand_entity_ids if cand != label_entity_id]
        while len(cand_entity_ids) < num_cands:
            cand_entity_ids.extend(cand_entity_ids)
        cand_entity_ids = cand_entity_ids[:num_cands] # 先截断
        cand_entity_ids[num_cands - 1] = FLAGS.nil_id # 再把NIL赋给最后一个元素
        label_id = num_cands - 1
        assert len(cand_entity_ids) == num_cands
        print(label_id)
    if is_training and label_entity_id != FLAGS.nil_id:
        ###
        if len(cand_entity_ids) == 1 and cand_entity_ids[0] == label_entity_id:  # 若cands只有一个元素，就是label，则用label填满所有cands
            while len(cand_entity_ids) < num_cands:
                cand_entity_ids.extend(cand_entity_ids)
        else:  # 否则，用所有非label的cands填满cands
            cand_entity_ids = [cand for cand in cand_entity_ids if cand != label_entity_id]
            ###
            while len(cand_entity_ids) < num_cands:
                cand_entity_ids.extend(cand_entity_ids)
        cand_entity_ids.insert(0, label_entity_id)  # cands的index=0处替换为label
        cand_entity_ids = cand_entity_ids[:num_cands]
        cand_entity_ids[num_cands - 1] = FLAGS.nil_id
        label_id = None
        # for i, entity in enumerate(cand_entity_ids):
        #     if entity == label_entity_id:
        #         # assert label_id == None
        #         label_id = i
        label_id = 0
        # assert label_id == 0
        assert len(cand_entity_ids) == num_cands


    instance_tokens = []
    instance_input_ids = []
    instance_segment_ids = []
    instance_input_mask = []
    instance_mention_id = []

    # for cand_entity_id in cand_entity_ids:
    for item in range (0,len(cand_entity_ids)-1): ## new change [0, 64-1)
        cand_entity_id = cand_entity_ids[item] ## new change
        cand_entity = all_entities[cand_entity_id]['text']
        cand_entity_truncate = ' '.join(cand_entity.split())
        cand_entity = tokenizer.tokenize(cand_entity_truncate)
        # cand 的长度比较短，取全
        tokens_b = cand_entity
        # mention 的长度若非常长，截取为总token可占长度-cand的长度
        mention_text_tokenized = mention_text_tokenized[:max_num_tokens - len(tokens_b)]
        tokens_a = mention_text_tokenized

        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将该pair的tokens映射到id数字上
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)  # 该pair tokens 的 segment id
        input_mask = [1] * len(input_ids)  # 全初始化为1
        mention_id = [0] * len(input_ids)  # bert input embedding中的，不是原始数据集标记的mention的index，全初始化为0

        # Update these indices to take [CLS] into account
        new_mention_start = 1
        # print('mention_text_tokenized length: ' + str(len(mention_text_tokenized))) #
        new_mention_end = len(mention_text_tokenized) + 1
        # print(mention_text_tokenized) #
        # print(tokens[new_mention_start: new_mention_end + 1]) #

        assert tokens[new_mention_start: new_mention_end] == mention_text_tokenized
        for t in range(new_mention_start, new_mention_end+1):
            mention_id[t] = 1  # 标记mention的位置

        assert len(input_ids) <= max_seq_length #检查拼起来的input 长度是否在最大长度限制范围内

        tokens = tokens + ['<pad>'] * (max_seq_length - len(tokens))  # 将tokens长度补全
        instance_tokens.extend(tokens)  # 将该候选pair的token加入该instance
        instance_input_ids.extend(pad_sequence(input_ids, max_seq_length))
        instance_segment_ids.extend(pad_sequence(segment_ids, max_seq_length))
        instance_input_mask.extend(pad_sequence(input_mask, max_seq_length))
        instance_mention_id.extend(pad_sequence(mention_id, max_seq_length))

        # print('cand entity : ' + str(cand_entity_id) + 'added..')
        print('len of instance tokens: ' + str(len(instance_tokens))) ## new change

    cand_entity_guids = []
    for i in range(0,num_cands-1):
        cand_entity_guids.append(int(cand_entity_ids[i])) ###新改 三行 try

    instance = TrainingInstance(
        tokens=instance_tokens,
        input_ids=instance_input_ids,
        input_mask=instance_input_mask,
        segment_ids=instance_segment_ids,
        label_id=label_id,
        mention_id=instance_mention_id,
        mention_guid=int(mention['mention_id']), #数据集中标记的mention index
        cand_guids=cand_entity_guids)  #该mention对应的cands id list

    # print('return create_instances_from_document()')

    return instance


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    entities_file = []
    for input_pattern in FLAGS.entities_file.split(","):
        entities_file.extend(tf.gfile.Glob(input_pattern))
    mentions_files = []
    for input_pattern in FLAGS.mentions_file.split(","):
        mentions_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in entities_file:
        tf.logging.info("  %s", input_file)
    for input_file in mentions_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        entities_file, mentions_files, tokenizer, FLAGS.max_seq_length,
        rng, is_training=FLAGS.is_training)

    tf.logging.info("*** Writing to output files ***")
    tf.logging.info("  %s", FLAGS.output_file)

    if FLAGS.split_by_domain:
        print('split by domain !!!')
        for corpus in instances:
            output_file = "%s/%s.tfrecord" % (FLAGS.output_file, corpus)
            write_instance_to_example_files(instances[corpus], tokenizer, FLAGS.max_seq_length,
                                            FLAGS.num_cands, [output_file])
    else:
        print('do not split by domain ^-^')
        write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                        FLAGS.num_cands, [FLAGS.output_file])


if __name__ == "__main__":
    flags.mark_flag_as_required("entities_file")
    flags.mark_flag_as_required("mentions_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
