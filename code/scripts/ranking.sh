export DATA_DIR=/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords
export SAVE_MODEL_PREFIX_DIR=/home/cxy/bert/pytorch_transformers

# train
nohup python run_ranking.py \
--mode=train \
--num_labels=2 \
--num_cands=99 \
--learning_rate=1e-5 \
--weight_decay=1e-2 \
--epochs=4 \
--load_model_path=$SAVE_MODEL_PREFIX_DIR/linkedin_teacher_student_model_result/onesteppretrain_teacher3_log_studentmodel_3epochs.pth \
--data_tfrecord=$DATA_DIR/train_notnil_99cands.tfrecord \
--data_tfrecord_idx=$DATA_DIR/train_notnil_99cands.idx \
--save_model_path_prefix=$SAVE_MODEL_PREFIX_DIR/linkedin_teacher_student_model_result/onepretrain_srcranking3_student3_srcranking >> onepretrain_srcranking3_student3_srcranking.out &
#--load_model_path=/home/cxy/bert/zeshel_master/pretrain_linkedin/BERT_linkedinlm/pytorch_model.bin \


# test
python run_ranking.py \
--mode=test \
--num_labels=2 \
--num_cands=299 \
--learning_rate=1e-5 \
--weight_decay=1e-2 \
--epochs=1 \
--load_model_path=$SAVE_MODEL_PREFIX_DIR/linkedin_teacher_student_model_result/onepretrain_srcranking3_student3_srcranking1epochs.pth \
--data_tfrecord=$DATA_DIR/linkedin_notnil_299cands.tfrecord \
--data_tfrecord_idx=$DATA_DIR/linkedin_notnil_299cands.idx

nohup python run_ranking.py \
--mode=train \
--num_labels=2 \
--num_cands=99 \
--learning_rate=1e-5 \
--weight_decay=1e-2 \
--epochs=4 \
--load_model_path=$SAVE_MODEL_PREFIX_DIR/patent_teacher_student_model_result/withoutpretrain_log_studentmodel_4epoches.pth \
--data_tfrecord=$DATA_DIR/train_notnil_99cands.tfrecord \
--data_tfrecord_idx=$DATA_DIR/train_notnil_99cands.idx \
--save_model_path_prefix=$SAVE_MODEL_PREFIX_DIR/patent_teacher_student_model_result/predict_withoutpretrain_log_studentmodel_4epoches >> withoutpretrain_log_studentmodel_4epoches.out &

python run_ranking.py \
--mode=test \
--num_labels=2 \
--num_cands=299 \
--learning_rate=1e-5 \
--weight_decay=1e-2 \
--epochs=1 \
--load_model_path=$SAVE_MODEL_PREFIX_DIR/patent_teacher_student_model_result/withoutpretrain_log_studentmodel_finetune_341.pth \
--data_tfrecord=$DATA_DIR/patent_notnil_299cands.tfrecord \
--data_tfrecord_idx=$DATA_DIR/patent_notnil_299cands.idx
