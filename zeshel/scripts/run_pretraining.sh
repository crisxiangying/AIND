export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12
export EXPTS_DIR=/home/cxy/bert/zeshel_master/pretrain_linkedin
export INPUT_FILE=/home/cxy/bert/data/zeshel/pretrain_linkedin/paper_pretraindata.tfrecord
# export INIT=$BERT_BASE_DIR/bert_model.ckpt
export INIT=/home/cxy/bert/zeshel_master/pretrain_linkedin/BERT_linkedinlm/model.ckpt-10000
export EXP_NAME=BERT_linkedlm_paperlm_twosteps
export USE_TPU=False
TPU_NAME=tpu0

split=val
domain='coronation_street'

# WB -> Src+Tgt
EXP_NAME=BERT_srctgtlm
INIT=$BERT_BASE_DIR/bert_model.ckpt
INPUT_FILE=$TFRecords/train/*,$TFRecords/$split/${domain}.tfrecord

python run_pretraining.py \
  --input_file=$INPUT_FILE \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --train_batch_size=2 \
  --max_seq_length=64 \
  --num_train_steps=10000 \
  --num_warmup_steps=500 \
  --save_checkpoints_steps=2000 \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
  --learning_rate=2e-5
# 上面的max_seq_length 要与 create_pretraining_data.sh 里的设置保持一致

# WB -> Src+Tgt -> Tgt
EXP_NAME=BERT_srctgtlm_tgtlm_${domain}
INIT=$EXPTS_DIR/BERT_srctgtlm/model.ckpt-10000
INPUT_FILE=$TFRecords/$split/${domain}.tfrecord

python run_pretraining.py \
  --input_file=$INPUT_FILE \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --train_batch_size=256 \
  --max_seq_length=256 \
  --num_train_steps=10000 \
  --num_warmup_steps=500 \
  --save_checkpoints_steps=2000 \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
  --learning_rate=2e-5
