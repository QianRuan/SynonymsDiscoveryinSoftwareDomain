# Synonyms-Discovery-in-Software-Domain
Learning domain-specific word embeddings for synonym discovery in software domain.<br>
## Setup: 
create anaconda env according to utils/requirments.txt<br>
## Data and pretrained models:
download or perpare your own training and evaluation datasets, save them in the folders trainData, evalData<br>
download pretrained models and save in the folder models<br>
## Training and evaluation: FastText models
modify the hyperparameters in utils/constants.py before training<br>
#### approach 1: training WE from scratch on the domain-specific training corpus
python main.py --train_ft_all  --eval_sim  --eval_syns_CV 4 <br> 
<br>
--eval_sim: evaluation method 1, synonyms discovery; <br>
--eval_syns_CV 4 :evaluation method 2, synonym pairs perdiction, do cross-validation with 4 folds<br> 
#### approach 2: domain adaption based on the pretrained model
python main.py --c_train_ft_all  --eval_sim  --eval_syns_CV 4
## Training and evaluation: BERT models
download and extract google's bert project: https://github.com/google-research/bert<br>
modify the paths in the .sh files before running experiments<br>
#### perpare training data for BERT
bash bert_perpare_data.sh<br>
#### approach 1: training WE from scratch on the domain-specific training corpus
bash bert_train.sh<br>
#### approach 2: domain adaption based on the pretrained model
bash bert_c_train.sh<br>
#### evaluation

bash bert_eval.sh or <br>
python main.py --modelName Cbert_github_ts200 --modelPath models/bert/Cbert_github_ts200/model.ckpt-200 --eval_sim_bert --eval_syns_CV_bert 4<br>
<br>
before evaluation, copy the files vocab.txt and bert_config.json from the pretrained bert model to the model's folder<br>
--modelName: name the model; <br>
--modelPath :path to the last checkpoint of the bert model<br> 
## Experiment results
evaluation metrics are saved in evaluation_results.xlxs<br> 
only best models are saved in models/<br> 




