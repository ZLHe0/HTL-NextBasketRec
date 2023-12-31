"""
Perform preprocessing on product data for Matrix Factorization training. The final 
resulting arrays are saved to the 'data' directory, ready to be used for training.
"""

import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # Read data (prior orders)
    prior_products = pd.read_csv('../../../data/raw/order_products__prior.csv', usecols=['order_id', 'product_id'])
    orders = pd.read_csv('../../../data/raw/orders.csv', usecols=['user_id', 'order_id'])
    user_products = prior_products.merge(orders, how='left', on='order_id')
    counts = user_products.groupby(['user_id', 'product_id']).size().rename('count').reset_index()

    # Save data (buying frequency matrix)
    if not os.path.isdir('data'):
        os.makedirs('data')

    np.save('data/i.npy', counts['user_id'].values)
    np.save('data/j.npy', counts['product_id'].values)
    np.save('data/V_ij.npy', counts['count'].values)
