import nltk
import numpy as np
import random
from collections import Counter
from nltk.tokenize import word_tokenize
import pickle
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos,neg):
    lexicon = []
    with open(pos,'r') as f:
        contents = f.readlines()
        for line in contents[:hm_lines]:
            tokens = word_tokenize(line)
            lexicon+=list(tokens)
            
    with open(neg,'r') as f:
        contents = f.readlines()
        for line in contents[:hm_lines]:
            tokens = word_tokenize(line)
            lexicon+=list(tokens)
            
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    count = Counter(lexicon)
    l2 = []
    for c in count:
        if 1000 > count[c] >50:
            l2.append(c)
    print(l2)
    return l2

def sample_handling(sample,lexicon,classification):
    featureset = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            words = word_tokenize(l.lower())
            words = [lemmatizer.lemmatize(i) for i in words]
            features = np.zeros(len(lexicon))
            for w in words:
                if w.lower() in lexicon:
                    index = lexicon.index(w.lower())
                    features[index]+=1
            featureset.append([features,classification])
    return featureset

def create_featuresets_and_labels(pos,neg,test_size=0.2):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos,lexicon,[1,0])
    features += sample_handling(neg,lexicon,[0,1])
    features = np.array(features)
    random.shuffle(features)
    test_length = int(0.2*len(features))
    test_x = list(features[:,0][:test_length])
    train_x = list(features[:,0][test_length:])
    test_y = list(features[:,1][:test_length])
    train_y = list(features[:,1][test_length:])
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_featuresets_and_labels('pos.txt','neg.txt')
    with open('sentiment_sets.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
    
        
                 
        
            
        