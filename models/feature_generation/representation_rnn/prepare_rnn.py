"""
Perform preprocessing on product data for RNN training. The final 
resulting arrays are saved to the 'data' directory, ready to be used for training.
"""

from collections import Counter
import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(__file__))
from utils_rnn import make_word_idx, pad_1d, encode_text

if __name__ == '__main__':
    ### Read data
    product_data = pd.read_csv('../../../data/processed/product_data.csv')
    # Remove non-str product name and history
    product_data = product_data.loc[product_data['product_name'].apply(lambda x: isinstance(x, str)),:]
    product_data = product_data.loc[product_data['is_ordered_history'].apply(lambda x: isinstance(x, str)),:]

    ### Product name embedding
    product_data['product_name'] = product_data['product_name'].map(lambda x: x.lower() if type(x)==str else 0)
    product_df = pd.read_csv('../../data/raw/products.csv')
    product_df['product_name'] = product_df['product_name'].map(lambda x: x.lower())
    word_idx = make_word_idx(product_df['product_name'].tolist())
    product_data['product_name_encoded'] = product_data['product_name'].map(lambda x: encode_text(x, word_idx))

    ### Initialize variables (order maximum length 100)
    num_rows = len(product_data)
    user_id = np.zeros(shape=[num_rows], dtype=np.int32)
    product_id = np.zeros(shape=[num_rows], dtype=np.int32)
    aisle_id = np.zeros(shape=[num_rows], dtype=np.int16)
    department_id = np.zeros(shape=[num_rows], dtype=np.int8)
    eval_set = np.zeros(shape=[num_rows], dtype='S5')
    label = np.zeros(shape=[num_rows], dtype=np.int8)
    is_ordered_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    index_in_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_dow_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_hour_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    days_since_prior_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    reorder_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_number_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    product_name = np.zeros(shape=[num_rows, 30], dtype=np.int32)
    product_name_length = np.zeros(shape=[num_rows], dtype=np.int8)
    history_length = np.zeros(shape=[num_rows], dtype=np.int8)

    ### Create nice-shaped input variable
    for i, row in product_data.iterrows():
        # Index Error Check: False
        # i = i - 1
        if i % 10000 == 0:
            print(i, num_rows)
        
        # Avoid over indexing
        if i == num_rows:
            break

        user_id[i] = row['user_id']
        product_id[i] = row['product_id']
        aisle_id[i] = row['aisle_id']
        department_id[i] = row['department_id']
        eval_set[i] = row['eval_set']
        label[i] = row['label']

        is_ordered_history[i, :], history_length[i] = pad_1d(list(map(int, row['is_ordered_history'].split())), 100)
        index_in_order_history[i, :], _ = pad_1d(list(map(int, row['index_in_order_history'].split())), 100)
        order_dow_history[i, :], _ = pad_1d(list(map(int, row['order_dow_history'].split())), 100)
        order_hour_history[i, :], _ = pad_1d(list(map(int, row['order_hour_history'].split())), 100)
        days_since_prior_order_history[i, :], _ = pad_1d(list(map(int, row['days_since_prior_order_history'].split())), 100)
        order_size_history[i, :], _ = pad_1d(list(map(int, row['order_size_history'].split())), 100)
        reorder_size_history[i, :], _ = pad_1d(list(map(int, row['reorder_size_history'].split())), 100)
        order_number_history[i, :], _ = pad_1d(list(map(int, row['order_number_history'].split())), 100)
        product_name[i, :], product_name_length[i] = pad_1d(list(map(int, row['product_name_encoded'].split())), 30)

    ### Save the data for rnn traning
    if not os.path.isdir('data'):
        os.makedirs('data')

    np.save('data/user_id.npy', user_id)
    np.save('data/product_id.npy', product_id)
    np.save('data/aisle_id.npy', aisle_id)
    np.save('data/department_id.npy', department_id)
    np.save('data/eval_set.npy', eval_set)
    np.save('data/label.npy', label)

    np.save('data/is_ordered_history.npy', is_ordered_history)
    np.save('data/index_in_order_history.npy', index_in_order_history)
    np.save('data/order_dow_history.npy', order_dow_history)
    np.save('data/order_hour_history.npy', order_hour_history)
    np.save('data/days_since_prior_order_history.npy', days_since_prior_order_history)
    np.save('data/order_size_history.npy', order_size_history)
    np.save('data/reorder_size_history.npy', reorder_size_history)
    np.save('data/order_number_history.npy', order_number_history)
    np.save('data/product_name.npy', product_name)
    np.save('data/product_name_length.npy', product_name_length)
    np.save('data/history_length.npy', history_length)
