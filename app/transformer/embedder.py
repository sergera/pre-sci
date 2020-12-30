
# fix randomness as best as possible
seed_value = 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)

from tensorflow import keras
from tensorflow.keras import layers

from keras import backend as K

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

class LearningRateReducerCb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.8
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

class Embedder():
    def __init__(self, categorical_feature, cardinality, target):
        self.sample_size = len(categorical_feature)
        self.categorical_feature = categorical_feature
        self.cardinality = cardinality
        self.target = target
        # if target['is_continuous']:
        if False:
            self.loss_function = rmse
            self.optimizer = "rmsprop"
            # self.loss_function = "mse"
            # self.optimizer = "adam"
        elif True:
            #target is binary
            self.loss_function = tf.nn.softmax_cross_entropy_with_logits
            self.optimizer = "adam"
        else:
            self.loss_function = "categorical_crossentropy"
            self.optimizer = "adam"

        self.model = keras.Sequential()
        self.embedding_size = min(50, self.cardinality+1/2)

        self.create_model()
        self.fit()
        
    def create_model(self):
        # self.model.add(layers.InputLayer(input_shape=(1,),name='input_' + '_'.join(self.categorical_feature.name)))
        self.model.add(
            layers.Embedding(
                input_dim=self.cardinality,
                output_dim=int(self.embedding_size), 
                input_length=1,
                name="embedding",
            )
        )
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(50, activation=tf.nn.relu6))
        self.model.add(layers.Dense(15, activation=tf.nn.relu6))
        self.model.add(layers.Dense(self.embedding_size, activation=tf.nn.relu6))
        self.model.compile(
            loss=self.loss_function, 
            optimizer=self.optimizer, 
            metrics=["accuracy"]
        )

    def fit(self):
        self.model.fit(
            x=self.categorical_feature.astype(np.float64), 
            y=self.target.to_numpy().astype(np.float64),
            epochs=50, 
            batch_size=4,
            callbacks=[LearningRateReducerCb()]
        )

    def predict(self, cat_feature):
        prediction = self.model.predict(x=cat_feature.astype(np.float64))
        return prediction