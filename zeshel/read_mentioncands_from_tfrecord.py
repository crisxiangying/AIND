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
"""Entity Linking training/eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import bertbase
from bertbase import optimization
from bertbase import tokenization
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "num_cands", 299,
    "Number of candidates. ")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_string("eval_domain", 'nil', "Eval domain.")

flags.DEFINE_string("output_predict_file", "/home/cxy/bert/pytorch_transformers/result_linkedin_test_train99cands/tfrecord_testsamples_nil_299cands_mention_cands.txt", "Test output file.")

flags.DEFINE_string("output_outputlayer_file", '/home/cxy/bert/data/wiki/expandNIL_putcasback/tfrecord_outputlayer.txt', "Test output file.")

flags.DEFINE_string("logits_file", '/home/cxy/bert/data/wiki/expandNIL_putcasback/tfrecord_logits.txt', "Test output file.")
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "num_train_examples", 13664,
    "Number of training examples.")


def file_based_input_fn_builder(input_file, num_cands, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {

        "input_ids": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
        "mention_id": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
        "label_id": tf.FixedLenFeature([], tf.int64),
        "mention_guid": tf.FixedLenFeature([], tf.int64),
        "cand_guids": tf.FixedLenFeature([num_cands,1], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d
    return input_fn


def create_zeshel_model(bert_config, is_training, input_ids, input_mask,
     segment_ids, mention_ids, labels, use_one_hot_embeddings, nil_embedding_arr):
  """Creates a classification model."""

  num_labels = input_ids.shape[1].value
  print('num_labels: ' + str(num_labels))
  seq_len = input_ids.shape[-1].value
  print('seq_len: ' + str(seq_len))

  input_ids = tf.reshape(input_ids, [-1, seq_len])
  segment_ids = tf.reshape(segment_ids, [-1, seq_len])
  input_mask = tf.reshape(input_mask, [-1, seq_len])
  mention_ids = tf.reshape(mention_ids, [-1, seq_len])
  nil_embedding = tf.convert_to_tensor(nil_embedding_arr, dtype = tf.float32)
  nil_embedding = tf.reshape(nil_embedding, [1, 768])

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      mention_ids=mention_ids,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)


  output_layer = model.get_pooled_output()
  return_output_layer = output_layer  ###新改 回299个，不要拼成300
  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [1], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    ### new change
    output_layer = tf.reshape(output_layer, [FLAGS.train_batch_size, num_labels, -1]) ###[8, 63, 768]

    # 计算logits
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, num_labels])  ########### 这里维度需要+1，留给NIL空位
    # 概率 用softmax计算
    probabilities = tf.nn.softmax(logits, axis=-1)
    return_output_layer = tf.reshape(return_output_layer, [299,768])  ###########新改
    print('return output_layer shape: ' + str(return_output_layer.shape) + str(return_output_layer.__class__))
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits, probabilities, return_output_layer)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, nil_embedding):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_id"]
    mention_ids = features["mention_id"]
    mention_guid = features["mention_guid"]
    cand_guids = features["cand_guids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities, output_layer) = create_zeshel_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, mention_ids,
        label_ids, use_one_hot_embeddings, nil_embedding) ###这个函数输入参数有nil_embedding

    print('return from model output_layer shape: ' + str(output_layer.shape) + str(output_layer.__class__))

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = bertbase.modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions)
        loss = tf.metrics.mean(values=per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
                       # "logits": logits,
                       # "probabilities": probabilities,
                       # "label_ids": label_ids,
                       # "mention_ids": mention_ids,
                       "mention_guid": mention_guid,  ####写 cand guids writer1
                       # "output_layer": output_layer,
                       "cand_guids": cand_guids  ####写 cand guids writer1
              },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = bertbase.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  if FLAGS.do_train:
    num_train_examples = FLAGS.num_train_examples
    num_train_steps = int(
        num_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      nil_embedding = nil_embedding)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_predict:
    # choose cand file
    predict_file = os.path.join(FLAGS.data_dir, "linkedin_nil_299cands.tfrecord") #填需要取cands序号的文件
    print(os.path.exists(predict_file))

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        num_cands=FLAGS.num_cands,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)
    result = estimator.predict(input_fn=predict_input_fn)
    print('********* finish inputing ***********')
    output_predict_file = FLAGS.output_predict_file
    output_outputlayer_file = FLAGS.output_outputlayer_file ###新改
    logits_file = FLAGS.logits_file
    print(os.path.exists(output_outputlayer_file))
    writer1 = tf.gfile.GFile(output_predict_file, "w")###新改
    writer2 = tf.gfile.GFile(output_outputlayer_file, "w")###新改
    writer3 = tf.gfile.GFile(logits_file, "w")  ###新改
    num_written_lines = 0
    tf.logging.info("***** write mentionguid candguid follow input order*****")
    for (i, prediction) in enumerate(result):
      num_written_lines += 1
      mention_guid = prediction['mention_guid']  ####写mention guid writer1
      cand_guids = prediction['cand_guids']  ####写 cand guids writer1
      output_line1 = str(mention_guid) + '\n'  + str(cand_guids) + '\n'
      writer1.write(output_line1)



if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()





