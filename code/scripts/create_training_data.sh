export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12

export ZESHEL_DATA=/home/cxy/bert/data/wiki/author_graph
export MENTIONS=$ZESHEL_DATA
export ENTITIES=$ZESHEL_DATA/entity.json
export CANDIDATES=$ZESHEL_DATA/author_graph/candidates.json
export OUTPUT_DIR=$ZESHEL_DATA/TFRecords




 
mkdir -p $OUTPUT_DIR/{train,val,test}


export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12

export ZESHEL_DATA=/home/cxy/bert/data
export MENTIONS=$ZESHEL_DATA/zeshel/linkedin/linkedin_nil_mentions.json
export ENTITIES=$ZESHEL_DATA/wiki/expandNIL_putcasback/entity.json
export CANDIDATES=$ZESHEL_DATA/zeshel/linkedin/linkedin_mentions_cands.json
export OUTPUT_DIR=$ZESHEL_DATA/wiki/expandNIL_putcasback/TFRecords

python create_training_data_tensorflowclassifier.py \
  --entities_file=$ENTITIES \
  --mentions_file=$MENTIONS \
	--candidates_file=$CANDIDATES \
  --output_file=$OUTPUT_DIR/linkedin_nil_299cands.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --is_training=False \
  --random_seed=12345 &





export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12

export ZESHEL_DATA=/home/cxy/bert/data/wiki/expandNIL_putcasback
export MENTIONS=$ZESHEL_DATA/testdata/patent_notnil_mentions.json
export ENTITIES=$ZESHEL_DATA/entity.json
export CANDIDATES=$ZESHEL_DATA/testdata/patent_notnil_mentions_cands.json
export OUTPUT_DIR=$ZESHEL_DATA/TFRecords


export BERT_BASE_DIR=/home/cxy/uncased_L-12_H-768_A-12

export ZESHEL_DATA=/home/cxy/bert/data/wiki/expandNIL_putcasback
export MENTIONS=$ZESHEL_DATA/traindata/train_mentions_notnil.json
export ENTITIES=$ZESHEL_DATA/entity.json
export CANDIDATES=$ZESHEL_DATA/traindata/train_candidates.json
export OUTPUT_DIR=$ZESHEL_DATA/TFRecords



python create_training_data_polybert.py \
  --entities_file=$ENTITIES \
  --mentions_file=$MENTIONS \
	--candidates_file=$CANDIDATES \
  --output_file=$OUTPUT_DIR/train_notnil_polybert.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=32 \
  --is_training=True \
  --random_seed=12345 &



export ZESHEL_DATA=/home/cxy/bert/data/wiki/author_graph
export MENTIONS=$ZESHEL_DATA/1try.json
export ENTITIES=/home/cxy/bert/data/wiki/expandNIL/new-entity.json
export CANDIDATES=$ZESHEL_DATA/candidates_removesomeentity.json
export OUTPUT_DIR=$ZESHEL_DATA/TFRecords

python create_training_data.py \
  --entities_file=$ENTITIES \
  --mentions_file=$MENTIONS \
	--candidates_file=$CANDIDATES \
  --output_file=$OUTPUT_DIR/1try.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --is_training=False \
  --random_seed=12345 &


//change_label_index
python create_training_data_change_label_index.py \
  --entities_file=$ENTITIES \
  --mentions_file=$MENTIONS/test.json \
	--candidates_file=$CANDIDATES \
  --output_file=$OUTPUT_DIR/newtest.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --is_training=False \
  --random_seed=12345

# split="val"

# python create_training_data.py \
#   --documents_file=$val_documents \
#   --mentions_file=$MENTIONS/${split}.json \
# 	--tfidf_candidates_file=$TFIDF_CANDIDATES/${split}.json \
#   --output_file=$OUTPUT_DIR/val \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=256 \
#   --is_training=False \
# 	--split_by_domain=True \
#   --random_seed=12345 &

# split="test"

# python create_training_data.py \
#   --documents_file=$test_documents \
#   --mentions_file=$MENTIONS/${split}.json \
#  	--tfidf_candidates_file=$TFIDF_CANDIDATES/${split}.json \
#   --output_file=$OUTPUT_DIR/test \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=256 \
#   --is_training=False \
# 	--split_by_domain=True \
#   --random_seed=12345 &
