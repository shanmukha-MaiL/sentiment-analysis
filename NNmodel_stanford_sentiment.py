import tensorflow as tf
import numpy as np
import nltk,pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nodes_hl_1 = nodes_hl_2 = 500
n_classes = 2
batch_size = 32
tot_batches = int(1600000/batch_size)
tot_epochs = 10
x = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([len(x[0]),nodes_hl_1])),
                      'biases' :tf.Variable(tf.random_normal([nodes_hl_1])) }
    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([nodes_hl_1,nodes_hl_2])),
                      'biases' :tf.Variable(tf.random_normal([nodes_hl_2])) }
    output_layer = {'weights':tf.Variable(tf.random_normal([nodes_hl_2,n_classes])),
                     'biases' :tf.Variable(tf.random_normal([n_classes])) }
    
    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    output = tf.add(tf.matmul(l2,output_layer['weights']),output_layer['biases'])
    return output

#saver = tf.train.Saver(['batches_run','epoch_loss'])
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = neural_network_model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2]) + 1
            print('STARTING EPOCH: ',epoch)
        except:
            epoch = 1
        
        while epoch<=tot_epochs:
#            if epoch!=1:
#                saver.restore(sess,'model_stanford_sentiment.ckpt')
            epoch_loss = 1
            with open('stanford_senti_lexicon.pickle','rb') as f:
                lexicon = pickle.load(f)
                
            with open('shuffled_train_set','r') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(i) for i in words]
                    feature = np.zeros(len(lexicon))
                    for word in words:
                        if word.lower() in lexicon:
                            index = lexicon.index(word.lower())
                            feature[index] += 1
                    line_x = list(feature)
                    line_y = list(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x)>=batch_size:
                        i,c = sess.run([optimizer,loss],feed_dict={x:batch_x,y:batch_y})
                        batches_run += 1
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        print('batches ran: ',batches_run,'/',batch_size,' of epoch: ',epoch,' batch_loss: ',c)
#            saver.save(sess,'model_stanford_sentiment.ckpt')
            print('epoch ',epoch,'/',tot_epochs,' completed.Epoch loss: ',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(epoch,'\n')
            epoch += 1
            
train_neural_network(x)

def create_test_features_and_labels(fin):
    featureset = []
    labels = []
    with open(fin,'r',buffering=20000,encoding='latin-1') as f:
        try:
            counter = 1
            for line in f:
                features = list(eval(line.split(':::')[1]))
                label = list(eval(line.split(':::')[0]))
                featureset.append(features)
                labels.append(label)
                counter += 1
        except :
            pass
            
    featureset = np.array(featureset)
    labels = np.array(labels)
    print('Tested samples: ',counter)
    return featureset,labels

def test_neural_network(x):
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        test_x,test_y = create_test_features_and_labels('processed_test_set.csv')
        print('Accuracy: ',accuracy.eval({x:test_x,y:test_y}))
        
test_neural_network(x)        
        
            
            
                    
            
    
