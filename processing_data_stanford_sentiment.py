import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle

lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

def init_process(fin,fout):
    out_file = open(fout,'a')
    with open(fin,'r',buffering=200000,encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == 0:
                    initial_polarity = [0,1]
                elif initial_polarity == 1:
                    initial_polarity = [1,0]
                tweet = line.split(',')[-1]    
                outline = str(initial_polarity) + ':::' + tweet
                out_file.write(outline)
        except Exception as e:
            print(str(e))
    out_file.close()   

init_process('trainingandtestdata/training.1600000.processed.noemoticon.csv','train_set.csv')
init_process('trainingandtestdata/testdata.manual.2009.06.14.csv','test_set.csv')    
                    
def create_lexicon(fout):
    lexicon = []
    with open(fout,'r',buffering=100000,encoding='latin-1') as f:
        try:
            content = ''
            counter = 1
            for line in f:
                counter += 1
                if (counter/2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' '+tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(words+lexicon))
                    print(counter,len(lexicon))
        except Exception as e:
            print(str(e))
            
    with open('stanford_senti_lexicon.pickle','wb') as f:
        pickle.dump(lexicon,f)
        
create_lexicon('train_set.csv')

def convert_to_vec(fin,fout,lexicon_pickle):
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)
        
    outfile = open(fout,'a')
    with open(fin,buffering=20000,encoding='latin-1') as f:
        for line in f:
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(i) for i in words]
            features = np.zeros(len(lexicon))
            for word in words:
                if word.lower() in lexicon:
                    index = lexicon.index(word.lower())
                    features[index] += 1
            outline = str(label)+':::'+str(list(features))+'\n'
            outfile.write(outline)
            
convert_to_vec('test_set.csv','processed_test_set.csv','stanford_senti_lexicon.pickle')            

def shuffle_data(fin):
    df = pd.read_csv(fin,error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('shuffled_train_set',index=False)
    
shuffle_data('train_set.csv')


    
                    
    
        
        
            
                
            