ROOT_DIR = 'D:/Shi/RQ/DSA'
MODEL_DIRS=[ROOT_DIR+'/models/fasttext']

CORPUS_DICT={'github':'trainData/github/githubNoEmptyLine.txt',
            'githubCodeCompressed':'trainData/github/githubCodeCompressed.txt',
            'githubAllCompressed':'trainData/github/githubAllCompressed.txt',
             'sofS':'trainData/stackoverflow/Posts10-cleaned.txt',
             'sofM':'trainData/stackoverflow/Posts20-cleaned.txt',
             'sofXL':'trainData/stackoverflow/Posts-cleaned.txt',
             'allS':'trainData/all/allS.txt',
             'allM':'trainData/all/allM.txt',
             'allXL':'trainData/all/allXL.txt'    
            }

SYNS_PAIRS = ['evalData/synsPairsNL.txt','evalData/synsPairsCodeTrans.txt','evalData/synsPairsFunc2Code.txt']
SAMPLE_SYNS_PAIRS = ['evalData/eval_sim_at_k/sample0.2_synsPairsNL.txt', 
                   'evalData/eval_sim_at_k/sample0.2_synsPairsCodeTrans.txt', 
                   'evalData/eval_sim_at_k/sample0.2_synsPairsFunc2Code.txt']

PL_SYNS_PAIRS = ['evalData/synsPairsCodeTrans.txt','evalData/synsPairsFunc2Code.txt']

TRAINFILE = 'evalData/eval_syns_prediction/syns_TRAIN.csv'
VALFILE = 'evalData/eval_syns_prediction/syns_VAL.csv'
TESTFILE = 'evalData/eval_syns_prediction/syns_TEST.csv'

EXCELFILE = 'evaluation_results.xlsx'
SHEETNAME1 = 'eval_sim_at_k'
SHEETNAME2 = 'eval_sim_PL'
SHEETNAME3 = 'eval_syns_CV'

PT_FT_PATH = 'models/pretrained-fasttext/cc.en.300.bin'

#hyperparameters-fatstext
FT_CORPUS=['github']#,'allS','allM'
FT_WINDOW=[5]#,5,10
FT_SG=[1]
FT_SIZE=[768]#,100,768
FT_MIN_COUNT=[5]#,1,10
FT_EPOCHS=[5]#,8,12,1



