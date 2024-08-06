import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, initializers


class User2Audio_MultiFeatureSpammerDetection(keras.Model):

    def __init__(self, out_size, self_size, embedding_size, h_dim=64):
        super(User2Audio_MultiFeatureSpammerDetection, self).__init__()
        self.self_model = tf.keras.layers.SimpleRNN(h_dim, activation='tanh', dropout=0.5,
                                                  input_shape=(self_size, embedding_size),
                                                  return_sequences=False)

        self.participant_cnn1 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', input_shape=(1500, 2, 1),
                                                       padding='same')
        self.participant_pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.participant_cnn2 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu')
        self.participant_flatten = tf.keras.layers.Flatten()
        self.participant_dense = tf.keras.layers.Dense(h_dim, name="Audio layer",
                                                       kernel_initializer=initializers.GlorotUniform())

        self.fc1 = tf.keras.layers.Dense(512, name="Predict Normal layer",
                                         kernel_initializer=initializers.GlorotUniform())
        self.fc1_1 = tf.keras.layers.Dense(128, name="Predict Normal layer_1",
                                           kernel_initializer=initializers.GlorotUniform())
        self.fc2 = tf.keras.layers.Dense(out_size, name="Predict layer",
                                         kernel_initializer=initializers.GlorotUniform())
        self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None):
        (self_matrix, participant_matrix) = inputs
        self_matrix = self.self_model(self_matrix)

        participant_matrix = tf.expand_dims(input=participant_matrix, axis=-1)
        participant_matrix = self.participant_cnn1(participant_matrix)
        participant_matrix = self.participant_pool(participant_matrix)
        # participant_matrix = self.participant_cnn2(participant_matrix)
        participant_matrix = self.participant_flatten(participant_matrix)
        participant_matrix = self.participant_dense(participant_matrix)
        participant_matrix = self.relu(participant_matrix)
        participant_matrix = self.dropout(participant_matrix)
        # participant_matrix = self.participant_flatten(participant_matrix)

        multi_feature = tf.concat(values=[self_matrix, participant_matrix], axis=-1)
        out = self.relu(multi_feature)
        out = self.dropout(out)
        out = self.fc1_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
