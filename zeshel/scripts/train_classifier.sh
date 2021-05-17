BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12
EXPTS_DIR=/home/cxy/bert/data/wiki
TFRecords=/home/cxy/bert/data/wiki/TFRecords/mentions
USE_TPU=false
TPU_NAME=None

EXP_NAME=BERT_fntn
INIT=$BERT_BASE_DIR/bert_model.ckpt


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

