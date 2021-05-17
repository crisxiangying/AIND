BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12
EXPTS_DIR=/home/cxy/bert/data/wiki
TFRecords=/home/cxy/bert/data/wiki/TFRecords/mentions
USE_TPU=false
TPU_NAME=None

EXP_NAME=BERT_fntn
INIT=$BERT_BASE_DIR/bert_model.ckpt


export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12
export EXPTS_DIR=/home/cxy/bert/data/wiki/author_graph
export TFRecords=/home/cxy/bert/data/wiki/author_graph/TFRecords
export USE_TPU=false
export TPU_NAME=None
export EXP_NAME=BERT_fntn_notnil_train
INIT=/home/cxy/bert/data/wiki/expandNIL/BERT_fntn_notnil_train/model.ckpt-34160


export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12
export EXPTS_DIR=/home/cxy/bert/data/wiki/expandNIL_putcasback
export TFRecords=/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords
export USE_TPU=false
export TPU_NAME=None
export EXP_NAME=BERT_fntn_notnil_train_removetestdata_5epochs
INIT=$BERT_BASE_DIR/bert_model.ckpt

python run_classifier_v2_trainmodel_notnil.py \
  --do_train=true \
  --do_eval=false \
  --do_predict=false \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --max_seq_length=64 \
  --train_batch_size=2 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --num_cands=63 \
  --save_checkpoints_steps=1000 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME


export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12
export EXPTS_DIR=/home/cxy/bert/data/wiki/expandNIL_putcasback
export TFRecords=/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords
export USE_TPU=false
export TPU_NAME=None
export EXP_NAME=BERT_fntn_all_train_after_notnil_train
export INIT=/home/cxy/bert/data/wiki/expandNIL_putcasback/BERT_fntn_notnil_train_removetestdata_5epochs/model.ckpt-34497

python run_classifier_v2_trainmodel_notnil.py \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --max_seq_length=64 \
  --train_batch_size=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --num_cands=299 \
  --save_checkpoints_steps=1000 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME


export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12
export EXPTS_DIR=/home/cxy/bert/data/wiki
export TFRecords=/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords
export USE_TPU=false
export TPU_NAME=None
export EXP_NAME=expandNIL_putcasback
export INIT=/home/cxy/bert/data/wiki/expandNIL_putcasback/BERT_fntn_notnil_train_removetestdata_5epochs/model.ckpt-34497

python run_classifier_graph.py \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --max_seq_length=64 \
  --train_batch_size=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --num_cands=299 \
  --save_checkpoints_steps=1000 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME


python run_classifier_v2_predictmodel.py \
  --do_train=true \
  --do_eval=false \
  --do_predict=false \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --max_seq_length=64 \
  --train_batch_size=2 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --num_cands=63 \
  --save_checkpoints_steps=1000 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME



EXPTS_DIR=/home/cxy/bert/data/wiki/shaozhoudata
TFRecords=/home/cxy/bert/data/wiki/shaozhoudata/TFRecords
USE_TPU=false
TPU_NAME=None
EXP_NAME=/home/cxy/bert/data/wiki/BERT_fntn
INIT=/home/cxy/bert/data/wiki/BERT_fntn/model.ckpt-34160


export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12
export EXPTS_DIR=/home/cxy/bert/data/wiki/expandNIL
export TFRecords=/home/cxy/bert/data/wiki/expandNIL/TFRecords
export USE_TPU=false
export TPU_NAME=None
export EXP_NAME=/home/cxy/bert/data/wiki/author_graph/BERT_fntn_notnil_train_cls_compression
export INIT=$BERT_BASE_DIR/bert_model.ckpt

python run_classifier_v2_compression_trainmodel.py \
  --do_train=true \
  --do_eval=false \
  --do_predict=false \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --max_seq_length=64 \
  --train_batch_size=2 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --num_cands=63 \
  --save_checkpoints_steps=1000 \
  --output_dir=$EXP_NAME \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME

python run_classifier_v2_get_output_embedding.py \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --max_seq_length=64 \
  --train_batch_size=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --num_cands=63 \
  --save_checkpoints_steps=1000 \
  --output_dir=$EXP_NAME \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME