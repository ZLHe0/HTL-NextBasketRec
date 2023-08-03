"""
Utility functions for rnn training.
"""

from collections import Counter
import tensorflow as tf # Make sure to use 1.x version
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from data_frame import DataFrame
from tf_utils import lstm_layer, time_distributed_dense_layer, dense_layer, sequence_log_loss, wavenet, log_loss
from tf_base_model import TFBaseModel

### Preprocessing utilities
# ensure that all sequences have the same length
def pad_1d(array, max_len):
    array = list(array)[:max_len]
    length = len(array)
    padded = array + [0]*(max_len - len(array))
    return padded, length

# mapping each unique word in the product names to a unique integer
def make_word_idx(product_names):
    words = [word for name in product_names for word in name.split()]
    word_counts = Counter(words)

    max_id = 1
    word_idx = {}
    for word, count in word_counts.items():
        if count < 10:
            word_idx[word] = 0
        else:
            word_idx[word] = max_id
            max_id += 1

    return word_idx

# text encoding
def encode_text(text, word_idx):
    return ' '.join([str(word_idx[i]) for i in text.split()]) if text else '0'



### Loading and preprocessing the data for RNN
class DataReader(object):

    def __init__(self, data_dir):
        # Columns that will be loaded
        # If extra columns are added, then extra placeholder is also needed
        data_cols = [
            'user_id',
            'product_id',
            'aisle_id',
            'department_id',
            'is_ordered_history',
            'index_in_order_history',
            'order_dow_history',
            'order_hour_history',
            'days_since_prior_order_history',
            'order_size_history',
            'reorder_size_history',
            'order_number_history',
            'history_length',
            'product_name',
            'product_name_length',
            'eval_set',
            'label'
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]
        self.full_df = DataFrame(columns=data_cols, data=data)

        print(self.full_df.shapes())
        print("loaded data")

        # Split the data into training and validation sets
        self.train_df, self.val_df = self.full_df.train_test_split(train_size=0.9)
        # Output each set's information
        print('train size', len(self.train_df))
        print('validation size', len(self.val_df))
        print('test size', len(self.full_df))

    # Generator for training batches    
    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    # Generator for validation batches
    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    # Generator for test (full data) batches 
    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.full_df,
            shuffle=False,
            num_epochs=1,
            is_test=True
        )

    # Batch generator for rnn
    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        batch_gen = df.batch_generator(batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)
        # Apply a shift to the features and variables.
        for batch in batch_gen:
            batch['order_dow_history'] = np.roll(batch['order_dow_history'], -1, axis=1)
            batch['order_hour_history'] = np.roll(batch['order_hour_history'], -1, axis=1)
            batch['days_since_prior_order_history'] = np.roll(batch['days_since_prior_order_history'], -1, axis=1)
            batch['order_number_history'] = np.roll(batch['order_number_history'], -1, axis=1)
            batch['next_is_ordered'] = np.roll(batch['is_ordered_history'], -1, axis=1)
            batch['is_none'] = batch['product_id'] == 0
            if not is_test:
                batch['history_length'] = batch['history_length'] - 1
            yield batch


# RNN model and its training and prediction (a sub-class of TFBaseModel)
class rnn(TFBaseModel):

    # Initialize the model with the given parameters
    def __init__(self, lstm_size=300, **kwargs):
        # The size of the LSTM unit output
        self.lstm_size = lstm_size
        super(rnn, self).__init__(**kwargs)

    # Create inputs for the model
    def get_input_sequences(self):
        # Define the nice-shaped placeholders for the input variable
        self.user_id = tf.placeholder(tf.int32, [None])
        self.product_id = tf.placeholder(tf.int32, [None])
        self.aisle_id = tf.placeholder(tf.int32, [None])
        self.department_id = tf.placeholder(tf.int32, [None])
        self.is_none = tf.placeholder(tf.int32, [None])
        self.history_length = tf.placeholder(tf.int32, [None])

        self.is_ordered_history = tf.placeholder(tf.int32, [None, 100])
        self.index_in_order_history = tf.placeholder(tf.int32, [None, 100])
        self.order_dow_history = tf.placeholder(tf.int32, [None, 100])
        self.order_hour_history = tf.placeholder(tf.int32, [None, 100])
        self.days_since_prior_order_history = tf.placeholder(tf.int32, [None, 100])
        self.order_size_history = tf.placeholder(tf.int32, [None, 100])
        self.reorder_size_history = tf.placeholder(tf.int32, [None, 100])
        self.order_number_history = tf.placeholder(tf.int32, [None, 100])
        self.product_name = tf.placeholder(tf.int32, [None, 30])
        self.product_name_length = tf.placeholder(tf.int32, [None])
        self.next_is_ordered = tf.placeholder(tf.int32, [None, 100])

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        # Initialize embeddings for product, aisle and department
        product_embeddings = tf.get_variable(
            name='product_embeddings',
            shape=[50000, self.lstm_size],
            dtype=tf.float32
        )
        aisle_embeddings = tf.get_variable(
            name='aisle_embeddings',
            shape=[250, 50],
            dtype=tf.float32
        )
        department_embeddings = tf.get_variable(
            name='department_embeddings',
            shape=[50, 10],
            dtype=tf.float32
        )

        # Initialize embeddings for product name
        product_names = tf.one_hot(self.product_name, 2532)
        product_names = tf.reduce_max(product_names, 1)
        product_names = dense_layer(product_names, 100, activation=tf.nn.relu)

        # Initialize the "buying nothing" vector
        is_none = tf.cast(tf.expand_dims(self.is_none, 1), tf.float32)

        # Concatenate all product-relevant features
        x_product = tf.concat([
            # Replicates the tensor along the second axis (history)
            tf.nn.embedding_lookup(product_embeddings, self.product_id),
            tf.nn.embedding_lookup(aisle_embeddings, self.aisle_id),
            tf.nn.embedding_lookup(department_embeddings, self.department_id),
            is_none,
            product_names
        ], axis=1)
        x_product = tf.tile(tf.expand_dims(x_product, 1), (1, 100, 1))

        # Initialize and concatenate all user-relevant features
        user_embeddings = tf.get_variable(
            name='user_embeddings',
            shape=[207000, self.lstm_size],
            dtype=tf.float32
        )
        x_user = tf.nn.embedding_lookup(user_embeddings, self.user_id)
        x_user = tf.tile(tf.expand_dims(x_user, 1), (1, 100, 1))

        # Initialize and concatenate all history features
        is_ordered_history = tf.one_hot(self.is_ordered_history, 2)
        index_in_order_history = tf.one_hot(self.index_in_order_history, 20)
        order_dow_history = tf.one_hot(self.order_dow_history, 8)
        order_hour_history = tf.one_hot(self.order_hour_history, 25)
        days_since_prior_order_history = tf.one_hot(self.days_since_prior_order_history, 31)
        order_size_history = tf.one_hot(self.order_size_history, 60)
        reorder_size_history = tf.one_hot(self.reorder_size_history, 50)
        order_number_history = tf.one_hot(self.order_number_history, 101)

        index_in_order_history_scalar = tf.expand_dims(tf.cast(self.index_in_order_history, tf.float32) / 20.0, 2)
        order_dow_history_scalar = tf.expand_dims(tf.cast(self.order_dow_history, tf.float32) / 8.0, 2)
        order_hour_history_scalar = tf.expand_dims(tf.cast(self.order_hour_history, tf.float32) / 25.0, 2)
        days_since_prior_order_history_scalar = tf.expand_dims(tf.cast(self.days_since_prior_order_history, tf.float32) / 31.0, 2)
        order_size_history_scalar = tf.expand_dims(tf.cast(self.order_size_history, tf.float32) / 60.0, 2)
        reorder_size_history_scalar = tf.expand_dims(tf.cast(self.reorder_size_history, tf.float32) / 50.0, 2)
        order_number_history_scalar = tf.expand_dims(tf.cast(self.order_number_history, tf.float32) / 100.0, 2)

        x_history = tf.concat([
            is_ordered_history,
            index_in_order_history,
            order_dow_history,
            order_hour_history,
            days_since_prior_order_history,
            order_size_history,
            reorder_size_history,
            order_number_history,
            index_in_order_history_scalar,
            order_dow_history_scalar,
            order_hour_history_scalar,
            days_since_prior_order_history_scalar,
            order_size_history_scalar,
            reorder_size_history_scalar,
            order_number_history_scalar,
        ], axis=2)

        # Construct the final input
        x = tf.concat([x_history, x_product, x_user], axis=2)

        return x

    # Calculate outputs (predictions and representations) for the model 
    def calculate_outputs(self, x):
        # An LSTM layer. The output 'h' is a sequence of hidden states.
        h = lstm_layer(x, self.history_length, self.lstm_size, scope='lstm-1')
        # Concatinate the original information and sequence information
        h = tf.concat([h, x], axis=2)
        
        # A dense layer with ReLU activation function. 
        h_final = time_distributed_dense_layer(h, 50, activation=tf.nn.relu, scope='dense-1')
        # The number of components for the mixture model.
        n_components = 1
        
        # Another dense layer without an activation function. 
        params = time_distributed_dense_layer(h_final, n_components*2, scope='dense-2', activation=None)
        # The output 2-dimensional features is then split into two parts.
        ps, mixing_coefs = tf.split(params, 2, axis=2)

        # Application of sigmoid function to the first part
        ps = tf.nn.sigmoid(ps)
        # Normalization of coefficients
        mixing_coefs = tf.nn.softmax(mixing_coefs - tf.reduce_min(mixing_coefs, 2, keep_dims=True))
        
        # The losses are calculated as the mixed sum, averaged over the sequence length.
        labels = tf.tile(tf.expand_dims(self.next_is_ordered, 2), (1, 1, n_components))
        losses = tf.reduce_sum(mixing_coefs*log_loss(labels, ps), axis=2)
        sequence_mask = tf.cast(tf.sequence_mask(self.history_length, maxlen=100), tf.float32)
        avg_loss = tf.reduce_sum(losses*sequence_mask) / tf.cast(tf.reduce_sum(self.history_length), tf.float32)

        # Access the last-layer representation and the prediction.
        final_temporal_idx = tf.stack([tf.range(tf.shape(self.history_length)[0]), self.history_length - 1], axis=1)
        self.final_states = tf.gather_nd(h_final, final_temporal_idx)
        self.final_predictions = tf.gather_nd(ps, final_temporal_idx)

        self.prediction_tensors = {
            'user_ids': self.user_id,
            'product_ids': self.product_id,
            'final_states': self.final_states,
            'predictions': self.final_predictions
        }
        # Directly output loss
        return avg_loss
    
    # Calculate the loss
    def calculate_loss(self):
        x = self.get_input_sequences()
        return self.calculate_outputs(x)