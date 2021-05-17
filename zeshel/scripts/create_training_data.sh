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

python create_training_data.py \
  --entities_file=$ENTITIES \
  --mentions_file=$MENTIONS \
	--candidates_file=$CANDIDATES \
  --output_file=$OUTPUT_DIR/linkedin_nil_299cands.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --is_training=False \
  --random_seed=12345 &

