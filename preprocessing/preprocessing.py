"""
Take input raw data files from the `./data/raw` directory, 
perform preprocessing steps such as cleaning and feature extraction, 
and output the processed data to the `./data/processed` directory.
"""

import os
import pandas as pd
import numpy as np
from IPython.display import clear_output
import sys
sys.path.append(os.path.dirname(__file__))
from prep_utils import parse_order, parse_user

if __name__ == '__main__':

    ##########################################
    # Part 1: Create User-wise Dataset
    ##########################################

    ### Data Reading
    orders = pd.read_csv('../data/raw/orders.csv')
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    train_products = pd.read_csv('../data/raw/order_products__train.csv')
    order_products = pd.concat([prior_products, train_products], axis=0)
    products = pd.read_csv('../data/raw/products.csv')

    ### Data Merging
    df = orders.merge(order_products, how='left', on='order_id')
    df = df.merge(products, how='left', on='product_id')
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0).astype(int)
    null_cols = ['product_id', 'aisle_id', 'department_id', 'add_to_cart_order', 'reordered']
    df[null_cols] = df[null_cols].fillna(0).astype(int)

    ### Sampling 
    # 1% user + 1% product with products in each aisle sampled at least 100
    unique_aisles = df['aisle_id'].unique()
    unique_departments = df['department_id'].unique()
    sampled_df = pd.DataFrame()
    # Sample at least ten rows for each aisle and department
    for aisle in unique_aisles:
        sampled_df = sampled_df.append(df[df['aisle_id'] == aisle].sample(n=10))
    for department in unique_departments:
        sampled_df = sampled_df.append(df[df['department_id'] == department].sample(n=10))
    unique_users = df['user_id'].unique()
    num_users_to_sample = int(0.01 * len(unique_users))
    # Identify training & non-training users
    last_order_eval_set = df.groupby('user_id')['eval_set'].last()
    training_users = last_order_eval_set[last_order_eval_set == 'train'].index
    num_training_users_to_sample = int(0.8 * num_users_to_sample)
    num_non_training_users_to_sample = num_users_to_sample - num_training_users_to_sample
    # Sample users
    sampled_training_users = np.random.choice(training_users, size=num_training_users_to_sample, replace=False)
    remaining_users = np.setdiff1d(unique_users, sampled_training_users)
    sampled_non_training_users = np.random.choice(remaining_users, size=num_non_training_users_to_sample, replace=False)
    sampled_df = sampled_df.reset_index(drop=True)
    # Append the sampled users' data to the sampled dataframe
    sampled_df = pd.concat([sampled_df, df[df['user_id'].isin(sampled_training_users)], df[df['user_id'].isin(sampled_non_training_users)]])

    # Identify the remaining data after the first round of sampling
    remaining_df = df.drop(sampled_df.index)
    # Sample to get an additional 1% of the data
    additional_rows = int(0.01 * len(remaining_df))
    additional_sampled_df = remaining_df.sample(n=additional_rows)
    sampled_df = pd.concat([sampled_df, additional_sampled_df])
    sampled_df = sampled_df.reset_index(drop=True)

    ### Create and save user data
    global_counter = 0
    user_data = df.groupby('user_id', sort=False).apply(parse_user).reset_index()
    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')
    user_data.to_csv('../data/processed/user_data.csv', index=False)
    print("User data successfully created")
    # Example Output: 
    '''
    1,"1 2 3","1 2 3","2 3 3","8 7 12","NaN 15.0 21.0","49302_11109 10258_11109 10258_11109_13176","120_108 107_108 107_108_24","16_16 19_16 19_16_4","0_0 1_1 1_1_0",train
    '''



    ##########################################
    # Part 2: Create User-product-pair Dataset
    ##########################################

    ### Read data
    df = pd.read_csv('../data/processed/user_data.csv')
    products = pd.read_csv('../data/raw/products.csv')

    ### Create dictionary for product information
    product_to_aisle = dict(zip(products['product_id'], products['aisle_id']))
    product_to_department = dict(zip(products['product_id'], products['department_id']))
    product_to_name = dict(zip(products['product_id'], products['product_name']))

    ### Initialize variables
    user_ids = []
    product_ids = []
    aisle_ids = []
    department_ids = []
    product_names = []
    eval_sets = []
    is_ordered_histories = []
    index_in_order_histories = []
    order_size_histories = []
    reorder_size_histories = []
    order_dow_histories = []
    order_hour_histories = []
    days_since_prior_order_histories = []
    order_number_histories = []
    labels = []
    longest = 0

    ### Traverse through all users and their products
    longest = 0
    # For each user
    for _, row in df.iterrows():
        if _ % 10000 == 0:
            print(_)
            data = [
            user_ids,
            product_ids,
            aisle_ids,
            department_ids,
            product_names,
            is_ordered_histories,
            index_in_order_histories,
            order_size_histories,
            reorder_size_histories,
            order_dow_histories,
            order_hour_histories,
            days_since_prior_order_histories,
            order_number_histories,
            labels,
            eval_sets
            ]
            # Length Check
            print("Length Check:", list(map(len, data)))

        user_id = row['user_id']
        eval_set = row['eval_set']
        products = row['product_ids']

        products, next_products = ' '.join(products.split()[:-1]), products.split()[-1]

        reorders = row['reorders']
        reorders, next_reorders = ' '.join(reorders.split()[:-1]), reorders.split()[-1]

        product_set = set([int(j) for i in products.split() for j in i.split('_')])
        next_product_set = set([int(i) for i in next_products.split('_')])

        orders = [map(int, i.split('_')) for i in products.split()]
        reorders = [map(int, i.split('_')) for i in reorders.split()]
        next_reorders = map(int, next_reorders.split('_'))

        # For each product
        for product_id in product_set:

            user_ids.append(user_id)
            product_ids.append(product_id)
            labels.append(int(product_id in next_product_set) if eval_set == 'train' else -1)
            eval_sets.append(eval_set)

            # Handle Null
            if product_id in product_to_aisle:
                aisle_ids.append(product_to_aisle[product_id])
            else:
                aisle_ids.append('0') 

            if product_id in product_to_department:
                department_ids.append(product_to_department[product_id])
            else:
                department_ids.append('0')

            if product_id in product_to_name:
                product_names.append(product_to_name[product_id])
            else:
                product_names.append('0')

            is_ordered = []
            index_in_order = []
            order_size = []
            reorder_size = []

            prior_products = set()
            # For each order
            for order in orders:
                is_ordered.append(str(int(product_id in order)))
                order_list = list(order)
                index_in_order.append(str(order_list.index(product_id) + 1) if product_id in order_list else '0')
                order_size.append(str(len(list(order))))
                reorder_size.append(str(len(list(prior_products & set(order)))))
                prior_products |= set(order)

            is_ordered = ' '.join(is_ordered)
            index_in_order = ' '.join(index_in_order)
            order_size = ' '.join(order_size)
            reorder_size = ' '.join(reorder_size)

            is_ordered_histories.append(is_ordered)
            index_in_order_histories.append(index_in_order)
            order_size_histories.append(order_size)
            reorder_size_histories.append(reorder_size)
            order_dow_histories.append(row['order_dows'])
            order_hour_histories.append(row['order_hours'])
            days_since_prior_order_histories.append(row['days_since_prior_orders'])
            order_number_histories.append(row['order_numbers'])

        # Handle the "None" prediction side case
        user_ids.append(user_id)
        product_ids.append(0)
        labels.append(int(max(next_reorders) == 0) if eval_set == 'train' else -1)

        aisle_ids.append(0)
        department_ids.append(0)
        product_names.append(0)
        eval_sets.append(eval_set)

        is_ordered = []
        index_in_order = []
        order_size = []
        reorder_size = []

        for reorder in reorders:
            is_ordered.append(str(int(max(reorder) == 0)))
            index_in_order.append(str(0))
            order_size.append(str(len(list(reorder))))
            reorder_size.append(str(sum(reorder)))

        is_ordered = ' '.join(is_ordered)
        index_in_order = ' '.join(index_in_order)
        order_size = ' '.join(order_size)
        reorder_size = ' '.join(reorder_size)

        is_ordered_histories.append(is_ordered)
        index_in_order_histories.append(index_in_order)
        order_size_histories.append(order_size)
        reorder_size_histories.append(reorder_size)
        order_dow_histories.append(row['order_dows'])
        order_hour_histories.append(row['order_hours'])
        days_since_prior_order_histories.append(row['days_since_prior_orders'])
        order_number_histories.append(row['order_numbers'])


    ### Collect data
    data = [
        user_ids,  # List of user IDs
        product_ids,  # List of product IDs
        aisle_ids,  # List of aisle IDs for each product
        department_ids,  # List of department IDs for each product
        product_names,  # List of product names
        is_ordered_histories,  # List of histories of whether each product was ordered
        index_in_order_histories,  # List of histories of the index of each product in its order
        order_size_histories,  # List of histories of the size of each order
        reorder_size_histories,  # List of histories of the number of reorders in each order
        order_dow_histories,  # List of histories of the day of the week of each order
        order_hour_histories,  # List of histories of the hour of the day of each order
        days_since_prior_order_histories,  # List of histories of the number of days since the prior order
        order_number_histories,  # List of histories of the order number
        labels,  # List of labels indicating whether the product was in the next order
        eval_sets  # List of evaluation set indicators (e.g., 'train', 'test')
    ]

    columns = [
        'user_id',
        'product_id',
        'aisle_id',
        'department_id',
        'product_name',
        'is_ordered_history',
        'index_in_order_history',
        'order_size_history',
        'reorder_size_history',
        'order_dow_history',
        'order_hour_history',
        'days_since_prior_order_history',
        'order_number_history',
        'label',
        'eval_set'
    ]

    ### Save data
    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    df = pd.DataFrame(dict(zip(columns, data)))
    df.to_csv('../data/processed/product_data.csv', index=False)
    
    print("Product data successfully created")

        