from builtins import print
import pandas as pd
import os
import time
from gensim.models.fasttext import FastText 
from gensim.test.utils import datapath
from gensim.utils import tokenize
import smart_open
from utils.constants import ROOT_DIR
from utils.constants import PT_FT_PATH
from utils.utils import load_bin_model
from utils.utils import create_dir





class tokenizedCorpus(object):
    
    def __iter__(self):      
        path = datapath(self.corpusPath)
        with smart_open.open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                #if (line and line.strip()):
                yield list(tokenize(line))
    def __call__(self, corpusPath):
        self.corpusPath = corpusPath
        return self

def train_fasttext(corpusPath,corpusName,window,size,sg,epochs,min_count):
    total_start=time.time()
   
    modelName='ft_'+corpusName+'_w'+str(window)+'_d' +str(size)+'_sg' +str(sg)+'_ep' +str(epochs)+'_minC'+str(min_count)
    model = FastText(size=size, window=window, min_count=min_count,sg=sg)
    
    modelPath = ROOT_DIR+'/'+"models/fasttext/"+modelName+"/"+modelName+".model"
    
    #check if the model  already exists
    dirPath = '/'.join(modelPath.split('/')[:-1])
    if os.path.exists(dirPath):
        print("The model already exists! ",modelName)
        return None,None,None
        
    #if not, train
    else:  
        print('#########################################################################')
        print("Ready to train the model ",modelName)
        df_cols=['modelName','total_duration','reading_corpus_duration',
                 'building_vocab_duration','training_duration','corpus_name','total_examples',
                 'window','size','sg','min_n','max_n','negative','word_ngrams','epochs','min_count','vocab_size']
                
        rows=[]
        result = {'modelName':modelName,'total_duration': None,'reading_corpus_duration': None,
                  'building_vocab_duration': None,'training_duration': None,'corpus_name': corpusName,'total_examples': None, 
                  'window':window,'size':size,'sg':sg,'min_n':3,'max_n':6,'negative':5,
                  'word_ngrams':1,'epochs':epochs,'min_count':None,'vocab_size':None}
        
        print("Reading corpus..."+corpusPath)
        startR=time.time()
        t=tokenizedCorpus()
        tCorpus = t(corpusPath)  
        result['reading_corpus_duration'] = int(time.time()-startR)
        
        print("Building vocab...")
        startB = time.time()
        model.build_vocab(sentences=tCorpus) 
        result['building_vocab_duration'] = int(time.time()-startB)
        print("Building done. building_vocab_duration=",result['building_vocab_duration'])
        
        total_examples = model.corpus_count
        result['total_examples'] = total_examples
        print("Total examples: "+str(total_examples))
        
        #train
        print("Training model...: "+modelName)
        startT = time.time()
        model.train(sentences=tCorpus, total_examples=total_examples, epochs=epochs)   
        result['training_duration'] = int(time.time()-startT) 
        print("Training done.")
        
        result['total_duration'] = int(time.time()-total_start) 
        print("Fasttext training time total: "+str(int(time.time()-total_start)))
        
        result['min_count']=model.vocabulary.min_count
        result['vocab_size']=len(model.wv.vectors_vocab)
              
       
        
        
        
        #save model info
        rows.append(result)
        df = pd.DataFrame(rows, columns = df_cols)
        
        #save to csv under the model's directory
        create_dir(modelPath)
        
        df.to_csv(os.path.dirname(modelPath)+'/model_info.csv', mode = 'a', index = False)
    
    
    return model,modelName,modelPath

def c_train_fasttext(corpusPath,corpusName,window,size,sg,epochs,min_count):
    total_start=time.time()
   
    modelName='Cft_'+corpusName+'_w'+str(window)+'_d' +str(size)+'_sg' +str(sg)+'_ep' +str(epochs)+'_minC'+str(min_count)
    
    modelPath = ROOT_DIR+'/'+"models/fasttext/"+modelName+"/"+modelName+".model"
    
    #check if the model  already exists
    dirPath = '/'.join(modelPath.split('/')[:-1])
    
    if os.path.exists(dirPath):
        print("The model already exists!",modelName)
        return None,None,None
        
    #if not, train
    else:  
        print('#########################################################################')
        print("Ready to train the model ",modelName)
        #load the pretrained fasttext model
        model=load_bin_model(ROOT_DIR+'/'+PT_FT_PATH)
        #model =  load_facebook_model(ROOT_DIR+'/'+PT_FT_PATH)
        #model = FastText(size=size, window=window, min_count=min_count,sg=sg)
        
        
            
        df_cols=['modelName','total_duration','reading_corpus_duration',
                 'building_vocab_duration','training_duration','corpus_name','total_examples',
                 'window','size','sg','min_n','max_n','negative','word_ngrams','epochs','min_count','vocab_size']
                
        rows=[]
        result = {'modelName':modelName,'total_duration': None,'reading_corpus_duration': None,
                  'building_vocab_duration': None,'training_duration': None,'corpus_name': corpusName,'total_examples': None, 
                  'window':window,'size':size,'sg':sg,'min_n':3,'max_n':6,'negative':5,
                  'word_ngrams':1,'epochs':epochs,'min_count':None,'vocab_size':None}
        
        print("Reading corpus..."+corpusPath)
        startR=time.time()
        t=tokenizedCorpus()
        tCorpus = t(corpusPath)  
        result['reading_corpus_duration'] = int(time.time()-startR)
        
        print("Building vocab...")
        startB = time.time()
        model.build_vocab(sentences=tCorpus,update=True) 
        result['building_vocab_duration'] = int(time.time()-startB)
        print("Building done. building_vocab_duration=",result['building_vocab_duration'])
        
        total_examples = model.corpus_count
        result['total_examples'] = total_examples
        print("Total examples: "+str(total_examples))
        
        #train
        print("Training model...: "+modelName)
        startT = time.time()
        #model.train(sentences=tCorpus, total_examples=total_examples, epochs=epochs)
        model.train(sentences=tCorpus, total_examples=total_examples,window=window,size=size,sg=sg,epochs=epochs,min_count=min_count)
        result['training_duration'] = int(time.time()-startT) 
        print("Training done.")
        
        result['total_duration'] = int(time.time()-total_start) 
        print("Fasttext training time total: "+str(int(time.time()-total_start)))
        
        result['min_count']=model.vocabulary.min_count
        result['vocab_size']=len(model.wv.vectors_vocab)
              
       
        
    
        
        #save model info
        rows.append(result)
        df = pd.DataFrame(rows, columns = df_cols)
        
        #save to csv under the model's directory
        create_dir(modelPath)
        
        df.to_csv(os.path.dirname(modelPath)+'/model_info.csv', mode = 'a', index = False)
    
    
        return model,modelName,modelPath


       