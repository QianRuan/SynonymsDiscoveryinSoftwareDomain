export BERT_PROGRAM=./bert-master
export BERT_BASE_DIR=./models/pretrained-bert/bert-base-cased
export OUTPUT_DATA=./trainData/data_for_bert/allS_tf_examples.tfrecord
export INPUT_DATA=./trainData/all/allS.txt

python $BERT_PROGRAM/create_pretraining_data.py \
  --input_file=$INPUT_DATA \
  --output_file=$OUTPUT_DATA \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=128 \
  --max_predictions_per_seq=2 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5