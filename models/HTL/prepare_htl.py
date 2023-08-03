"""
Preprocess representations from RNN and MF for benchmark and HTL modeling.
"""
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    ### RNN Feature Representations
    h_df = pd.DataFrame(np.load('../feature_generation/representation_rnn/predictions/final_states.npy'))
    h_df['user_id'] = np.load('../feature_generation/representation_rnn/predictions/user_ids.npy')
    h_df['product_id'] = np.load('../feature_generation/representation_rnn/predictions/product_ids.npy')
    h_df['label'] = np.load('../feature_generation/representation_rnn/data/label.npy')
    h_df['rnn_pred_prob'] = np.load('../feature_generation/representation_rnn/predictions/predictions.npy')

    ### Matrix Factorization Representations
    nnmf_p_matrix = np.load('../feature_generation/representation_mf/predictions/product_embeddings.npy')
    product_emb_df = pd.DataFrame(nnmf_p_matrix, columns=['nnmf_product_{}'.format(i) for i in range(nnmf_p_matrix.shape[1])])
    product_emb_df['product_id'] = np.arange(nnmf_p_matrix.shape[0])
    h_df = h_df.merge(product_emb_df, how='left', on='product_id')

    nnmf_u_matrix = np.load('../feature_generation/representation_mf/predictions/user_embeddings.npy')
    user_emb_df = pd.DataFrame(nnmf_u_matrix, columns=['nnmf_user_{}'.format(i) for i in range(nnmf_u_matrix.shape[1])])
    user_emb_df['user_id'] = np.arange(nnmf_u_matrix.shape[0])
    h_df = h_df.merge(user_emb_df, how='left', on='user_id')

    ### Create features
    drop_cols = [
        'label',
        'user_id',
        'product_id',
        'rnn_pred_prob'
    ]
    user_id = h_df['user_id']
    product_id = h_df['product_id']
    label = h_df['label']

    h_df.drop(drop_cols, axis=1, inplace=True)
    features = h_df.values
    feature_names = h_df.columns.values
    feature_maxs = features.max(axis=0)
    feature_mins = features.min(axis=0)
    feature_means = features.mean(axis=0)

    ### Save features
    if not os.path.isdir('data'):
        os.makedirs('data')

    np.save('data/user_id.npy', user_id)
    np.save('data/product_id.npy', product_id)
    np.save('data/features.npy', features)
    np.save('data/feature_names.npy', h_df.columns)
    np.save('data/feature_maxs.npy', feature_maxs)
    np.save('data/feature_mins.npy', feature_mins)
    np.save('data/feature_means.npy', feature_means)
    np.save('data/label.npy', label)