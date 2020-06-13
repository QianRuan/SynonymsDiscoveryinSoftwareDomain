export BERT_PROGRAM=./bert-master
export BERT_BASE_DIR=./models/pretrained-bert/bert-base-cased
export INPUT_DATA=./trainData/data_for_bert/github_tf_examples.tfrecord
export OUTPUT=./models/bert/Cbert_github_ts200

python $BERT_PROGRAM/run_pretraining.py \
  --input_file=$INPUT_DATA \
  --output_dir=$OUTPUT \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=10 \
  --max_seq_length=128 \
  --max_predictions_per_seq=2 \
  --num_train_steps=200 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5