from utils.utils import append_eval_results_to_excel
from utils.utils import compute_oov

import numpy as np
import time
import pandas as pd
# import os
from pyjarowinkler import distance
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import make_classification

###########Evaluation 1: similarity: find top k similar terms###########
def find_top_k_cand(model, entity, candidates, k):
    
    cand_sim_dict={}
    
    for candidate in candidates:            
        #compute similarity
        sim_score = model.wv.similarity(entity,candidate)
        cand_sim_dict.update({candidate:sim_score})
            
    list_sim=list(cand_sim_dict.values())
   
        
        
    #sort the list by similarity and get top k    
    top_k_idx = np.argsort(list_sim)[-k:]   
    top_k_idx_re = top_k_idx[::-1] 
       
    #find top k candidates
    top_k_cand = []
    for i in top_k_idx_re:        
        top_k_cand.append((list(cand_sim_dict.items())[i][0]))
            
    #print ("top",k,top_k_cand)
    return top_k_cand

def compute_entity_k_precision(true_syns, top_k_cand, k):
    #precision=tp/tp+fp 
    
    #count true positive
    tp = 0
    for cand in top_k_cand:
        if cand in true_syns:
            tp+=1
    
    #tp+fp=k
    precisionAtk=round(tp/k,4)*100
    
    
    return precisionAtk

def compute_entity_k_recall(true_syns, top_k_cand):
    #recall=tp/tp+fn
    
    #count true positive
    tp = 0
    for cand in top_k_cand:
        if cand in true_syns:
            tp+=1
            
    #count false negative
    fn = 0
    for syn in true_syns:
        if not (syn in top_k_cand):
            fn+=1
            
    recallAtk=round(tp/(tp+fn),4)*100
       
    return recallAtk  

def compute_entity_k_f1(precision, recall):
    #f1=2*((precision*recall)/(precision+recall))
    
    if (precision+recall)==0:
        f1=0
    else:
        f1 = round(2*((precision*recall)/(precision+recall)),2)
    
    return f1

def evaluate_sim_at_k(model, modelName, modelPath, domain_vocab, syns_dict, excelfile, sheetname):
    start=time.time()
   
    df_cols=['modelName','duration','oov%',
             'precision@k=1','recall@k=1','f1@k=1',
             'precision@k=3','recall@k=3','f1@k=3',
             'precision@k=6','recall@k=6','f1@k=6',]
    rows=[]
    result = {'modelName':modelName,'duration': None,'oov%':None,
              'precision@k=1': None, 'recall@k=1': None, 'f1@k=1': None, 
              'precision@k=3': None, 'recall@k=3': None, 'f1@k=3': None, 
              'precision@k=6': None, 'recall@k=6': None, 'f1@k=6': None }
    
    #compute oov%
    oov = compute_oov(domain_vocab, model, modelName)
    result['oov%']=oov
    
    #list to save overall metrics
    overall_precisionAtk1=[]
    overall_precisionAtk3=[]
    overall_precisionAtk6=[]
    
    overall_recallAtk1=[]
    overall_recallAtk3=[]
    overall_recallAtk6=[]
    
    overall_f1Atk1=[]
    overall_f1Atk3=[]
    overall_f1Atk6=[]
    
    
    for entity in domain_vocab:
        #get true synonyms of the query entity
        true_syns=syns_dict[entity]  
        
        #take all the rest vocab as candidates
        candidates = domain_vocab.copy()           
        candidates.remove(entity)
        
        #find top k candidates
        top_6_cand = find_top_k_cand(model, entity, candidates, 6)
        top_3_cand = top_6_cand[:3]
        top_1_cand = top_6_cand[:1]
        
        #compute precision
        entity_precisionAtk1 = compute_entity_k_precision(true_syns, top_1_cand, 1)
        entity_precisionAtk3 = compute_entity_k_precision(true_syns, top_3_cand, 3)
        entity_precisionAtk6 = compute_entity_k_precision(true_syns, top_6_cand, 6)
        
        #compute recall
        entity_recallAtk1 = compute_entity_k_recall(true_syns, top_1_cand)
        entity_recallAtk3 = compute_entity_k_recall(true_syns, top_3_cand)
        entity_recallAtk6 = compute_entity_k_recall(true_syns, top_6_cand)
        
        #compute f1_score
        entity_f1Atk1 = compute_entity_k_f1(entity_precisionAtk1, entity_recallAtk1)
        entity_f1Atk3 = compute_entity_k_f1(entity_precisionAtk3, entity_recallAtk3)
        entity_f1Atk6 = compute_entity_k_f1(entity_precisionAtk6, entity_recallAtk6)
        
        
        #append to overall list
        overall_precisionAtk1.append(entity_precisionAtk1)
        overall_precisionAtk3.append(entity_precisionAtk3)
        overall_precisionAtk6.append(entity_precisionAtk6)
        
        overall_recallAtk1.append(entity_recallAtk1)
        overall_recallAtk3.append(entity_recallAtk3)
        overall_recallAtk6.append(entity_recallAtk6)
        
        overall_f1Atk1.append(entity_f1Atk1)
        overall_f1Atk3.append(entity_f1Atk3)
        overall_f1Atk6.append(entity_f1Atk6)
        
        
    #compute overall avg. metrics
    result['precision@k=1'] = round (sum(overall_precisionAtk1) / len (overall_precisionAtk1), 2)
    result['precision@k=3'] = round (sum(overall_precisionAtk3) / len (overall_precisionAtk3), 2)
    result['precision@k=6'] = round (sum(overall_precisionAtk6) / len (overall_precisionAtk6), 2)
    
    result['recall@k=1'] = round (sum(overall_recallAtk1) / len (overall_recallAtk1), 2)
    result['recall@k=3'] = round (sum(overall_recallAtk3) / len (overall_recallAtk3), 2)
    result['recall@k=6'] = round (sum(overall_recallAtk6) / len (overall_recallAtk6), 2)
    
    result['f1@k=1'] = round (sum(overall_f1Atk1) / len (overall_f1Atk1), 2)
    result['f1@k=3'] = round (sum(overall_f1Atk3) / len (overall_f1Atk3), 2)
    result['f1@k=6'] = round (sum(overall_f1Atk6) / len (overall_f1Atk6), 2)
    
    result['duration'] = round(time.time()-start,0)
    
    #save to dataframe
    rows.append(result)
    df = pd.DataFrame(rows, columns = df_cols)
    #save to csv under the model's directory
    dirPath = '/'.join(modelPath.split('/')[:-1])
    df.to_csv(dirPath + '/'+sheetname+'.csv', mode = 'a', index = False)
    #save to excel
    append_eval_results_to_excel(excelfile, sheetname, df)
    
    print(df)
    
    
    return df


###########Evaluation 2: synsonym prediction (pairs classification)###########
def construct_lexical_matching_features(a,b):
    #split terms into list of words
    wordsA=a.split()
    wordsB=b.split()

    #m1: the number of the common words shared by a and b
    m1 = len(set(wordsA) & set(wordsB))

    #m2: m1/length of a * length of b
    m2 = m1/(len(wordsA)*len(wordsB))

    #m3: if a and b only differ by an antonym prefix, true=1, for example 'like'&'dislike'
    antonym_prefix=['anti','dis','il','im','in','ir','non','un']
    m3=0
    for pre in antonym_prefix:
        if (pre+a) == b:
            m3=1
    for pre in antonym_prefix:
        if (pre+b) == a:
            m3=1
    

    #m4: if all the uppercase characters from a and b match each other, true=1. For example:'USA' and 'United States of America'
    upperCharinA=''
    upperCharinB=''
    m4=0
    for c in a:
        if c.isupper():
            upperCharinA=upperCharinA+c
    for c in b:
        if c.isupper():
            upperCharinB=upperCharinB+c
    if (upperCharinA!='') and (upperCharinA==upperCharinB):
        m4=1
   

    #m5: if all the first characters in each word from a and b match each other, true=1. For example:'hs' and hierarchical softmax'
    m5=0
    firstCharA=''
    firstCharB=''
    for w in wordsA:
        firstCharA=firstCharA+w[0]
    for w in wordsB:
        firstCharB=firstCharB+w[0]
    if (firstCharA==b):
        m5=1
    if (firstCharB==a):
        m5=1
    if (firstCharA==firstCharB):
        m5=1
    

    #m6: if one term is the subsequence of another term, true=1
    m6=0
    if a in b:
        m6=1
    if b in a:
        m6=1
     

    lexical_matching_features=np.array([m1,m2,m3,m4,m5,m6])
    return lexical_matching_features

def construct_sim_features(a,b,model):
    
    #cosine similarity
    cos_sim = model.wv.similarity(a,b)

    #Jaro-Winkler similarity
    jw_sim = distance.get_jaro_distance(a, b, winkler=True, scaling=0.1)

    sim_features=np.array([cos_sim,jw_sim])
    
    return sim_features

def construct_pair_features(a,b,model):
    
    #word vectors
    vectorA = model.wv[a]
    vectorB = model.wv[b]
    
    #lexical matching features
    lexical_matching_features = construct_lexical_matching_features(a,b)
    
    
    #similarity features, including cosine similarity and Jaro-Winkler similarity
    sim_features = construct_sim_features(a, b, model)
    
    
    features=np.concatenate((vectorA, vectorB, lexical_matching_features, sim_features), axis=None)
    
    return features

def convert_dataframe_to_X_and_y(df,model):
    listA = df['entity'].to_list()
    listB = df['syn'].to_list()
    labels = df['class'].to_list()
    y = np.array(labels)
    
    feature_dim=8+model.wv.vector_size*2
    X = np.zeros([len(labels),feature_dim], dtype=float)
    
    for i in range(len(labels)):
        a=listA[i]
        b=listB[i]
        if (type(a)==str) and (type(b)==str):
            features=construct_pair_features(listA[i],listB[i],model)
            X[i]=features     
    return X, y    

def evaluate_syns_prediction(model,modelName,modelPath,train,val,test,excelfile,sheetname):
    
    start=time.time()
   
    df_cols=['modelName','duration','C',
             'precision','recall','f1']
            
    rows=[]
    result = {'modelName':modelName,'duration': None,'C': None,
              'precision': None, 'recall': None, 'f1': None}
    
    #perpare datasets
    
    X_train,y_train=convert_dataframe_to_X_and_y(train,model)
    
    X_val,y_val=convert_dataframe_to_X_and_y(val,model)
    
    X_test,y_test=convert_dataframe_to_X_and_y(test,model)
    
    
    #train classifiers and validate with f1 score
    Cs=[0.01, 0.1, 0.5, 1, 4, 16, 64, 256]
    iteration=1000
    dual=False
    models = [LinearSVC(C=Cs[0],max_iter=iteration,dual=dual),
              LinearSVC(C=Cs[1],max_iter=iteration,dual=dual),
              LinearSVC(C=Cs[2],max_iter=iteration,dual=dual),
              LinearSVC(C=Cs[3],max_iter=iteration,dual=dual),
              LinearSVC(C=Cs[4],max_iter=iteration,dual=dual),
              LinearSVC(C=Cs[5],max_iter=iteration,dual=dual),
              LinearSVC(C=Cs[6],max_iter=iteration,dual=dual),
              LinearSVC(C=Cs[7],max_iter=iteration,dual=dual)]

    val_f1_scores=[]

    for clf in models:
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_val)
        val_f1=f1_score(y_val, y_pred, average='macro')    
        val_f1_scores.append(val_f1)

    #find best model    
    idx=val_f1_scores.index(max(val_f1_scores)) 

    best_model=models[idx]
    result['C']=Cs[idx]

    #train the final model and test
    best_model.fit(X_train, y_train)
    y_pred=best_model.predict(X_test)
    test_f1=f1_score(y_test, y_pred, average='macro')  
    test_recall=recall_score(y_test, y_pred, average='macro',zero_division=1)
    test_precision=precision_score(y_test, y_pred, average='macro')

    

        
        
    #compute overall avg. metrics
    result['precision'] = round (test_precision, 4) * 100
    
    result['recall'] = round (test_recall, 4)  * 100
     
    result['f1'] = round (test_f1, 4) * 100
    
    result['duration'] = round(time.time()-start,0)
    
    #save to dataframe
    rows.append(result)
    df = pd.DataFrame(rows, columns = df_cols)
    
    #save to csv under the model's directory
    dirPath = '/'.join(modelPath.split('/')[:-1])
    df.to_csv(dirPath + '/'+sheetname+'.csv', mode = 'a', index = False)
    
    #save to excel
    append_eval_results_to_excel(excelfile, sheetname, df)
    
    print(df)
    
    
    return df
       

