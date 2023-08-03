"""
Utility functions for data preprocessing.
"""

import os
import pandas as pd
import numpy as np
from IPython.display import clear_output

def parse_order(x):
    # Input: dataframe of a single order
    # Output: product string with detailed information "_" separated
    series = pd.Series(dtype='object') ### specify dtype

    series['products'] = '_'.join(x['product_id'].values.astype(str).tolist())
    series['reorders'] = '_'.join(x['reordered'].values.astype(str).tolist())
    series['aisles'] = '_'.join(x['aisle_id'].values.astype(str).tolist())
    series['departments'] = '_'.join(x['department_id'].values.astype(str).tolist())

    series['order_number'] = x['order_number'].iloc[0]
    series['order_dow'] = x['order_dow'].iloc[0]
    series['order_hour'] = x['order_hour_of_day'].iloc[0]
    series['days_since_prior_order'] = x['days_since_prior_order'].iloc[0]
    
    # Increment the counter and print the current count
    global global_counter 
    global_counter += 1
    clear_output(wait=True)
    print(f"total loops run: {global_counter}")

    return series

def parse_user(x):
    # Input: dataframe of a single user, including all orders
    # Output: user string with all the bought produce and its detailed information " " separated
    parsed_orders = x.groupby('order_id', sort=False).apply(parse_order)

    series = pd.Series(dtype='object')

    series['order_ids'] = ' '.join(parsed_orders.index.map(str).tolist())
    series['order_numbers'] = ' '.join(parsed_orders['order_number'].map(str).tolist())
    series['order_dows'] = ' '.join(parsed_orders['order_dow'].map(str).tolist())
    series['order_hours'] = ' '.join(parsed_orders['order_hour'].map(str).tolist())
    series['days_since_prior_orders'] = ' '.join(parsed_orders['days_since_prior_order'].map(str).tolist())

    series['product_ids'] = ' '.join(parsed_orders['products'].values.astype(str).tolist())
    series['aisle_ids'] = ' '.join(parsed_orders['aisles'].values.astype(str).tolist())
    series['department_ids'] = ' '.join(parsed_orders['departments'].values.astype(str).tolist())
    series['reorders'] = ' '.join(parsed_orders['reorders'].values.astype(str).tolist())

    series['eval_set'] = x['eval_set'].values[-1]

    return series