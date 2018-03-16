from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
class NeuralNetwork(object):
    def __init__(self,learning_rate):
        self.train_features , self.train_target ,self.test_features,self.test_targets = preprocessing()
        self.learning_rate = learning_rate
        self.no_of_features = len(self.train_features.columns)
        self.no_of_labels = len(self.train_target.colums)

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
         model.fit(self.train_features, self.train_target, epochs=20)
         accuracy = model.evaluate(self.test_features,self.test_targets,verbose=0)
         print ("Accuracy for keras model is" + str(accuracy[1]))

    def train_tensor(self):
        features_count = self.no_of_features
        labels_count = self.no_of_labels
        features = tf.placeholder(tf.float32)
        labels = tf.placeholder(tf.float32)
        weights = tf.Variable(tf.truncated_normal((features_count,labels_count)))
        biases = tf.Variable(tf.zeros(labels_count))

        logits = tf.matmul(features, weights) + biases

        prediction = tf.nn.softmax(logits)

        # Cross entropy
        cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

        # Training loss
        loss = tf.reduce_mean(cross_entropy)

        # Create an operation that initializes all variables
        init = tf.global_variables_initializer()

        # Test Cases
        with tf.Session() as session:
            session.run(init)
            session.run(loss)
            biases_data = session.run(biases)
            is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
            # Calculate the accuracy of the predictions
            accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))




         



