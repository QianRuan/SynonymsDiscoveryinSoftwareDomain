import argparse
import pandas as pd
import time
import sys
from utils.utils import generate_domain_vocab
from utils.utils import generate_syns_dictionary
from utils.utils import load_bin_model
from utils.utils import load_ft_model
from utils.utils import check_curr_best_models_save_and_delete
from utils.utils import copy_eval_results_and_check_best_models
from utils.constants import CORPUS_DICT
from utils.constants import SAMPLE_SYNS_PAIRS
from utils.constants import PL_SYNS_PAIRS
from utils.constants import TRAINFILE
from utils.constants import VALFILE
from utils.constants import TESTFILE
from utils.constants import EXCELFILE
from utils.constants import SHEETNAME1
from utils.constants import SHEETNAME2
from utils.constants import SHEETNAME3
from utils.constants import ROOT_DIR
from utils.constants import FT_CORPUS
from utils.constants import FT_WINDOW
from utils.constants import FT_SG
from utils.constants import FT_SIZE
from utils.constants import FT_MIN_COUNT
from utils.constants import FT_EPOCHS
from evaluation import evaluate_sim_at_k
from evaluation import evaluate_sim_at_k_PL
from evaluation import evaluate_sim_at_k_bert
from evaluation import evaluate_syns_prediction
from evaluation import evaluate_syns_prediction_CV
from evaluation import evaluate_syns_prediction_CV_bert
from train import train_fasttext
from train import c_train_fasttext






parser = argparse.ArgumentParser(description='Training domain-specific word embeddings')
#load existing models
parser.add_argument('--modelName', dest='modelName',
                    help='model name')
parser.add_argument('--modelPath', dest='modelPath',
                    help='model path')
parser.add_argument('--load_bin', dest='load_bin', action='store_true',
                    help='load existing binary models')
parser.add_argument('--load_ft', dest='load_ft', action='store_true',
                    help='load existing fasttext models')

#training 
parser.add_argument('--train_ft', dest='train_ft', action='store_true',
                    help='train fasttext models from scratch')
parser.add_argument('--train_ft_all', dest='train_ft_all', action='store_true',
                    help='train fasttext models from scratch')
parser.add_argument('--corpusName', dest='corpusName', 
                    help='training dataset, see constant.py CORPUS_DICT')
parser.add_argument('--ft_window', dest='ft_window', type=int,
                    help='model hyperparameter')
parser.add_argument('--ft_size', dest='ft_size', type=int,
                    help='word embedding dimension size')
parser.add_argument('--ft_sg', dest='ft_sg', type=int,
                    help='if the skipgram model is used')
parser.add_argument('--ft_epochs', dest='ft_epochs', type=int,
                    help='training epochs')
# parser.add_argument('--ft_min_n', dest='ft_min_n', type=int,
#                     help='if the skipgram model is used')
# parser.add_argument('--ft_max_n', dest='ft_max_n', type=int,
#                     help='training dataset')
# parser.add_argument('--ft_negative', dest='ft_negative', type=int,
#                     help='training dataset')
# parser.add_argument('--ft_word_ngrams', dest='ft_word_ngrams', type=int,
#                     help='training dataset')
parser.add_argument('--ft_min_count', dest='ft_min_count', type=int,
                    help='ignore words with frequency less than the value')


parser.add_argument('--c_train_ft', dest='c_train_ft', action='store_true',
                    help='continue train the pre-trained fasttext model on the training data')
parser.add_argument('--c_train_ft_all', dest='c_train_ft_all', action='store_true',
                    help='train fasttext models from scratch')



#evaluation
parser.add_argument('--eval_sim', dest='eval_sim', action='store_true',
                    help='evaluate similarity at top k')
parser.add_argument('--eval_sim_bert', dest='eval_sim_bert', action='store_true',
                    help='evaluate similarity at top k')
parser.add_argument('--eval_sim_PL', dest='eval_sim_PL', action='store_true',
                    help='evaluate similarity at top k on programming language datasets')
parser.add_argument('--eval_syns', dest='eval_syns', action='store_true',
                    help='evaluate synonym prediction')
parser.add_argument('--eval_syns_CV', dest='eval_syns_CV', type=int,
                    help='evaluate synonym prediction with cross validation')
parser.add_argument('--eval_syns_CV_bert', dest='eval_syns_CV_bert', type=int,
                    help='evaluate synonym prediction with cross validation')
parser.add_argument('--copy_results_check_best', dest='copy_results_check_best', action='store_true',
                    help='copy eval results and color the best')


def evaluate(model,modelName,modelPath,args):
    
    if args.eval_sim_bert:
            sample_domain_vocab = generate_domain_vocab(SAMPLE_SYNS_PAIRS, ROOT_DIR)
            sample_syns_dict = generate_syns_dictionary(sample_domain_vocab, SAMPLE_SYNS_PAIRS, ROOT_DIR)
            excelfile = ROOT_DIR + '/' + EXCELFILE
            evaluate_sim_at_k_bert(model, modelName, modelPath, sample_domain_vocab, sample_syns_dict, excelfile, SHEETNAME1)
            copy_eval_results_and_check_best_models()
    
    if args.eval_sim_PL:
        
        
        if model is not None:
            print ('doing eval_sim_PL')
        
            sample_domain_vocab = generate_domain_vocab(PL_SYNS_PAIRS, ROOT_DIR)
            print(len(sample_domain_vocab))
            sample_syns_dict = generate_syns_dictionary(sample_domain_vocab, PL_SYNS_PAIRS, ROOT_DIR)
            excelfile = ROOT_DIR + '/' + EXCELFILE
            
            evaluate_sim_at_k_PL(model, modelName, modelPath, sample_domain_vocab, sample_syns_dict, excelfile, SHEETNAME2)
            copy_eval_results_and_check_best_models()
            
    if args.eval_sim:
        
        if model is not None:
        
            sample_domain_vocab = generate_domain_vocab(SAMPLE_SYNS_PAIRS, ROOT_DIR)
            sample_syns_dict = generate_syns_dictionary(sample_domain_vocab, SAMPLE_SYNS_PAIRS, ROOT_DIR)
            excelfile = ROOT_DIR + '/' + EXCELFILE
            
            evaluate_sim_at_k(model, modelName, modelPath, sample_domain_vocab, sample_syns_dict, excelfile, SHEETNAME1)
            copy_eval_results_and_check_best_models()
    if args.eval_syns:
        
        if model is not None:
        
            TRAIN=pd.read_csv(ROOT_DIR+'/'+TRAINFILE, encoding='utf-8')
            VAL=pd.read_csv(ROOT_DIR+'/'+VALFILE, encoding='utf-8')
            TEST=pd.read_csv(ROOT_DIR+'/'+TESTFILE, encoding='utf-8')
            excelfile = ROOT_DIR + '/' + EXCELFILE
            
            evaluate_syns_prediction(model, modelName, modelPath, TRAIN, VAL, TEST, excelfile, SHEETNAME2)
            copy_eval_results_and_check_best_models()
        
    if args.eval_syns_CV is not None:
        
        if model is not None:
        
            TRAIN=pd.read_csv(ROOT_DIR+'/'+TRAINFILE, encoding='utf-8')
            VAL=pd.read_csv(ROOT_DIR+'/'+VALFILE, encoding='utf-8')
            TEST=pd.read_csv(ROOT_DIR+'/'+TESTFILE, encoding='utf-8')
            excelfile = ROOT_DIR + '/' + EXCELFILE
            
            evaluate_syns_prediction_CV(model, modelName, modelPath, TRAIN, VAL, TEST, excelfile, SHEETNAME3,args.eval_syns_CV)
            copy_eval_results_and_check_best_models()
            
    if args.eval_syns_CV_bert is not None:
        
        TRAIN=pd.read_csv(ROOT_DIR+'/'+TRAINFILE, encoding='utf-8')
        VAL=pd.read_csv(ROOT_DIR+'/'+VALFILE, encoding='utf-8')
        TEST=pd.read_csv(ROOT_DIR+'/'+TESTFILE, encoding='utf-8')
        excelfile = ROOT_DIR + '/' + EXCELFILE
        
        evaluate_syns_prediction_CV_bert(model, modelName, modelPath, TRAIN, VAL, TEST, excelfile, SHEETNAME3,args.eval_syns_CV_bert)
        copy_eval_results_and_check_best_models()     

    
    
    
 
def main():
    
    start=time.time()
    
    args = parser.parse_args()
    
    if args.copy_results_check_best:
        copy_eval_results_and_check_best_models()
        
     
    if args.modelName is not None:
        
        if args.modelPath is not None:
            
            modelName = args.modelName
            modelPath = ROOT_DIR + '/' + args.modelPath
            
    
            #load models
            if args.load_bin:
                
                model=load_bin_model(modelPath)
                
                evaluate(model,modelName,modelPath,args)
                
                
            if args.load_ft:
                
                model=load_ft_model(modelPath)
                
                evaluate(model,modelName,modelPath,args)
                
        if args.eval_sim_bert:
            
            evaluate(None,modelName,modelPath,args)
            
        if args.eval_syns_CV_bert:
            
            evaluate(None,modelName,modelPath,args)
            
        
            
        #else:
            #sys.exit('--modelPath required!')
            
            
    if args.train_ft :
        
        if (args.corpusName and args.ft_window and args.ft_size and args.ft_sg and args.ft_epochs and args.ft_min_count) is not None:
            
            #train with specific parameters
            corpusName=args.corpusName
            corpusPath=ROOT_DIR+'/'+CORPUS_DICT[corpusName]
            window=args.ft_window
            size=args.ft_size
            sg=args.ft_sg
            epochs=args.ft_epochs
            min_count=args.ft_min_count
            
            
            model, modelName, modelPath=train_fasttext(corpusPath,corpusName,window,size,sg,epochs,min_count)
            
            if model is not None:
                
                evaluate(model,modelName,modelPath,args)
                check_curr_best_models_save_and_delete(model,modelName,modelPath)
                print("DONE! total duration: ",int(time.time()-start))
                print('#########################################################################')
            
            
            
        else:
            sys.exit('Training parameters required, or switch to training all mode!')
                                    
                                    
                                    
            
    if args.c_train_ft :
        
        if (args.corpusName and args.ft_window and args.ft_size and args.ft_sg and args.ft_epochs and args.ft_min_count) is not None:
        
            corpusName=args.corpusName
            corpusPath=ROOT_DIR+'/'+CORPUS_DICT[corpusName]
            window=args.ft_window
            size=args.ft_size
            sg=args.ft_sg
            epochs=args.ft_epochs
            min_count=args.ft_min_count
            
            
            model, modelName, modelPath=c_train_fasttext(corpusPath,corpusName,window,size,sg,epochs,min_count)
           
            if model is not None:
               
               evaluate(model,modelName,modelPath,args)
               check_curr_best_models_save_and_delete(model,modelName,modelPath)
               print("DONE! total duration: ",int(time.time()-start))
               print('#########################################################################')
            
        else:
            sys.exit('Training parameters required, or switch to training all mode!')
            
            
    if args.train_ft_all :
                  
        #check that no other parameters given    
        if (args.corpusName and args.ft_window and args.ft_size and args.ft_sg and args.ft_epochs and args.ft_min_count) is None:
                    
                    #train with parameters saved in constants.py
                    for corpusName in FT_CORPUS:
                        
                        corpusPath=ROOT_DIR+'/'+CORPUS_DICT[corpusName]
                        
                        for window in FT_WINDOW:
                            
                            for size in FT_SIZE:
                                
                                for sg in FT_SG:
                                    
                                    for min_count in FT_MIN_COUNT:
                                        
                                        for epochs in FT_EPOCHS:
                                            
                                            model, modelName, modelPath=train_fasttext(corpusPath,corpusName,window,size,sg,epochs,min_count)
                                            
                                            if model is not None:
                
                                                evaluate(model,modelName,modelPath,args)
                                                check_curr_best_models_save_and_delete(model,modelName,modelPath)
                                                print("DONE! total duration: ",int(time.time()-start))
                                                print('#########################################################################')
        else:
            sys.exit('TRAINING ALL MODE: Non training parameters needed!')
            
    if args.c_train_ft_all :
                 
       #check that no other parameters given    
       if (args.corpusName and args.ft_window and args.ft_size and args.ft_sg and args.ft_epochs and args.ft_min_count) is None:
                   
                   #train with parameters saved in constants.py
                   for corpusName in FT_CORPUS:
                       
                       corpusPath=ROOT_DIR+'/'+CORPUS_DICT[corpusName]
                       
                       for window in FT_WINDOW:
                           
                           for size in FT_SIZE:
                               
                               for sg in FT_SG:
                                   
                                   for min_count in FT_MIN_COUNT:
                                       
                                       for epochs in FT_EPOCHS:
                                           
                                           model, modelName, modelPath=c_train_fasttext(corpusPath,corpusName,window,size,sg,epochs,min_count)
                                           
                                           if model is not None:
               
                                               evaluate(model,modelName,modelPath,args)
                                               check_curr_best_models_save_and_delete(model,modelName,modelPath)
                                               print("DONE! total duration: ",int(time.time()-start))
                                               print('#########################################################################')
       else:
           sys.exit('TRAINING ALL MODE: Non training parameters needed!')    
    
    
        

        
        

if __name__ == '__main__':
    main()


