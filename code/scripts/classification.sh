export DATA_DIR=/home/cxy/bert/data/wiki/expandNIL_putcasback/TFRecords
export SAVE_MODEL_PREFIX_DIR=/home/cxy/bert/pytorch_transformers

# train
# K = 99
nohup python run_classification.py \
--mode=train \
--num_labels=2 \
--num_cands=99 \
--learning_rate=1e-5 \
--weight_decay=1e-2 \
--epochs=3 \
--load_model_path=$SAVE_MODEL_PREFIX_DIR/linkedin_teacher_student_model_result/onesteppretrain_teacher3_log_studentmodel_3epochs.pth \
--data_tfrecord=$DATA_DIR/train_all_100cands.tfrecord \
--data_tfrecord_idx=$DATA_DIR/train_all_100cands.idx \
--save_model_path_prefix=$SAVE_MODEL_PREFIX_DIR/linkedin_teacher_student_model_result/onsteppretrain_classifier_ranking3epochs_student3epochs_ >> onepretraintrainclassi_ranking3_student3.out &

# test
# K = 299
# tag pseudo labels for target domain (patent / linkedin)
python run_classification.py \
--mode=test \
--num_labels=2 \
--num_cands=299 \
--learning_rate=1e-5 \
--weight_decay=1e-2 \
--epochs=1 \
--load_model_path=$SAVE_MODEL_PREFIX_DIR/linkedin_teacher_student_model_result/onsteppretrain_classifier_ranking3epochs_student3epochs_1epochs.pth \
--data_tfrecord=$DATA_DIR/linkedin_nil_299cands.tfrecord \
--data_tfrecord_idx=$DATA_DIR/linkedin_nil_299cands.idx \
--save_predictresult_path_prefix=$SAVE_MODEL_PREFIX_DIR/linkedin_teacher_student_model_result/onesteppretrain_rankingteacherstudent33_classification1
