from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
class NeuralNetwork(object):
    def __init__(self,learning_rate):
        self.train_features , self.train_target ,self.test_features,self.test_targets = preprocessing()
        self.learning_rate = learning_rate
        self.no_of_features = len(self.train_features.columns)
        self.no_of_labels = len(self.train_target.columns)
        self.no_of_epochs = 100
    def train_keras(self):
         model = Sequential()
         model.add(Dense(units=64,activation='sigmoid',input_dim=self.no_of_features))
         model.add(Dropout(0.2))
         model.add(Dense(units=128,activation='sigmoid'))
         model.add(Dropout(0.2))
         model.add(Dense(units=3))
         model.compile(optimizer='rmsprop',
                       loss='mean_squared_error',
                       metrics=['accuracy'])
         model.fit(self.train_features, self.train_target, epochs=self.no_of_epochs)
         accuracy = model.evaluate(self.test_features,self.test_targets,verbose=0)
         print ("Accuracy for keras model is" + str(accuracy[1]))

    def train_tensor(self):
        features_count = self.no_of_features
        labels_count = self.no_of_labels
        features = tf.placeholder("float")
        labels = tf.placeholder("float")
        layer1_weight_shape = (features_count,64)
        layer2_weight_shape = (64,128)
        layer3_weight_shape = (128,labels_count)
        layer1_bias_shape = (64)
        layer2_bias_shape = (128)
        layer3_bias_shape = (labels_count)
        weights_shape_matrix = [
            layer1_weight_shape,layer2_weight_shape,layer3_weight_shape
        ]
        bias_shape_matrix = [
            layer1_bias_shape,layer2_bias_shape,layer3_bias_shape
        ]


        weights_layer0 = tf.Variable(tf.truncated_normal((weights_shape_matrix[0])))
        biases_layer0 = tf.Variable(tf.zeros(bias_shape_matrix[0]))
        output_layer0 = tf.matmul(features, weights_layer0) + biases_layer0
        output_layer0 = tf.layers.dropout(output_layer0,rate=0.5)
        output_layer0 = tf.sigmoid(output_layer0)
        weights_layer1 = tf.Variable(tf.truncated_normal((weights_shape_matrix[1])))
        biases_layer1 = tf.Variable(tf.zeros(bias_shape_matrix[1]))
        output_layer1 = tf.matmul(output_layer0, weights_layer1) + biases_layer1
        output_layer1 = tf.layers.dropout(output_layer1, rate=0.5)
        output_layer1 = tf.sigmoid(output_layer1)
        weights_layer2 = tf.Variable(tf.truncated_normal((weights_shape_matrix[2])))
        biases_layer2 = tf.Variable(tf.zeros(bias_shape_matrix[2]))
        logits = tf.matmul(output_layer1, weights_layer2) + biases_layer2
        prediction = tf.nn.softmax(logits)
        # Cross entropy
        cross_entropy = -tf.reduce_sum(tf.sqrt((labels - prediction)*(labels - prediction)), reduction_indices=1)
        # Training loss
        loss  = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)
        # Create an operation that initializes all variables
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            session.run(loss,feed_dict={features:self.train_features,labels:self.train_target})
          #  biases_data = session.run(biases)
            is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
            # Calculate the accuracy of the predictions
            accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
            #Print the accuracy of the model
            accuracy = session.run(accuracy,feed_dict={labels:self.train_target,features:self.train_features})
            print ("Accuracy is "+str((accuracy)))


         



