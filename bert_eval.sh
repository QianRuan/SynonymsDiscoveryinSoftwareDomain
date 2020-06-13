#before evaluating the domain specific bert models, copy vocab.txt and bert_config.json from the pretrained model
#python main.py --modelName bert_github_ts200 --modelPath models/bert/bert_github_ts200/model.ckpt-200 --eval_syns_CV_bert 4
#python main.py --modelName Cbert_github_ts200 --modelPath models/bert/Cbert_github_ts200/model.ckpt-200 --eval_syns_CV_bert 4
#python main.py --modelName bert_github_ts200 --modelPath models/bert/bert_github_ts200/model.ckpt-200 --eval_sim_bert 
python main.py --modelName Cbert_github_ts200 --modelPath models/bert/Cbert_github_ts200/model.ckpt-200 --eval_sim_bert 