import tensorflow as tf
from create_sentiment_featuresets import create_featuresets_and_labels

nodes_hl_1 = nodes_hl_2 = nodes_hl_3 = 1500
n_classes = 2
batch_size = 100
tot_epochs = 10
x = tf.placeholder('float')
y = tf.placeholder('float')

train_x,train_y,test_x,test_y = create_featuresets_and_labels('pos.txt','neg.txt')

def neural_network_model(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),nodes_hl_1])),
                      'biases' :tf.Variable(tf.random_normal([nodes_hl_1])) }
    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([nodes_hl_1,nodes_hl_2])),
                      'biases' :tf.Variable(tf.random_normal([nodes_hl_2])) }
    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([nodes_hl_2,nodes_hl_3])),
                      'biases' :tf.Variable(tf.random_normal([nodes_hl_3])) }
    output_layer = {'weights':tf.Variable(tf.random_normal([nodes_hl_3,n_classes])),
                      'biases' :tf.Variable(tf.random_normal([n_classes])) }
    
    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(tot_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                epoch_x,epoch_y = train_x[start:end],train_y[start:end]
                j,c = sess.run([optimizer,loss],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss+=c
                i += batch_size
            print('Epoch ',epoch,' completed.Total loss = ',epoch_loss)
            correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy: ',accuracy.eval({x:test_x,y:test_y}))
train_neural_network(x)            
            
    