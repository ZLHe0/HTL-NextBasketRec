"""
Utility functions for building matrix factorization models
"""
import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append(os.path.join(os.getcwd(), '..'))

from data_frame import DataFrame
from tf_base_model import TFBaseModel

### Loading and preprocessing the data for Matrix Factorization (MF)
class DataReader(object):

    # Initialize input
    def __init__(self, data_dir):
        data_cols = ['i', 'j', 'V_ij']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = df.train_test_split(train_size=0.9)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))

        self.num_users = df['i'].max() + 1
        self.num_products = df['j'].max() + 1

    # Generate training batch 
    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000
        )

    # Generate validation batch
    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000
        )

    # Batch generator
    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        return df.batch_generator(batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)

### MF model and its training (a sub-class of TFBaseModel)
class nnmf(TFBaseModel):

    # Initialize the (assumed) rank of the matrix
    def __init__(self, rank=25, **kwargs):
        self.rank = rank
        super(nnmf, self).__init__(**kwargs)

    # Calculate the loss
    def calculate_loss(self):
        self.i = tf.placeholder(dtype=tf.int32, shape=[None])
        self.j = tf.placeholder(dtype=tf.int32, shape=[None])
        self.V_ij = tf.placeholder(dtype=tf.float32, shape=[None])

        self.W = tf.Variable(tf.truncated_normal([self.reader.num_users, self.rank]))
        self.H = tf.Variable(tf.truncated_normal([self.reader.num_products, self.rank]))
        W_bias = tf.Variable(tf.truncated_normal([self.reader.num_users]))
        H_bias = tf.Variable(tf.truncated_normal([self.reader.num_products]))

        global_mean = tf.Variable(0.0)
        w_i = tf.gather(self.W, self.i)
        h_j = tf.gather(self.H, self.j)

        w_bias = tf.gather(W_bias, self.i)
        h_bias = tf.gather(H_bias, self.j)
        interaction = tf.reduce_sum(w_i * h_j, reduction_indices=1)
        preds = global_mean + w_bias + h_bias + interaction

        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, self.V_ij)))

        self.parameter_tensors = {
            'user_embeddings': self.W,
            'product_embeddings': self.H
        }

        return rmse