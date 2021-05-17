export DATA_DIR=/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords
export SAVE_MODEL_PREFIX_DIR=/home/cxy/bert/pytorch_transformers

# train
nohup python run_ranking_student.py \
--mode=train \
--num_labels=2 \
--num_cands=99 \
--learning_rate=1e-5 \
--weight_decay=1e-2 \
--epochs=4 \
--load_model_path=$SAVE_MODEL_PREFIX_DIR/ranking_linkedin_pretrain/linkedinlm10000epoches_bert99cands_bert_model_notniltraindata_3epochs.pth \
--src_tfrecord=$DATA_DIR/train_notnil_299cands.tfrecord \
--src_tfrecord_idx=$DATA_DIR/train_notnil_299cands.idx \
--tgt_tfrecord=$DATA_DIR/linkedin_notnil_299cands.tfrecord \
--tgt_tfrecord_idx=$DATA_DIR/linkedin_notnil_299cands.idx \
--save_model_path_prefix=$SAVE_MODEL_PREFIX_DIR/linkedin_teacher_student_model_result/onesteppretrain_teacher3_log_studentmodel_ \
--read_fakelabel_file=$SAVE_MODEL_PREFIX_DIR/ranking_linkedin_pretrain/onesteppretrain_ranking3epochs_linkedin_fakelabel.json >> linkedin.out &


# test
# 直接调用 run_ranking.py 的 test mode 来得到学生模型在 target 上的预测结果
python run_ranking.py \
--mode=test \
--num_labels=2 \
--num_cands=299 \
--learning_rate=1e-5 \
--weight_decay=1e-2 \
--load_model_path=$SAVE_MODEL_PREFIX_DIR/ranking_company_pretrain/expandpretrain10000epoches_bert99cands_bert_model_notniltraindata_3epoches.pth \
--data_tfrecord=$DATA_DIR/patent_notnil_299cands.tfrecord \
--data_tfrecord_idx=$DATA_DIR/patent_notnil_299cands.idx \
