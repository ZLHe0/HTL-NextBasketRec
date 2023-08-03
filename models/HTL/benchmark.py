"""
Create two benchmark models based on learned representations and evaluate the performance.
"""
import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from plotnine import ggplot, aes, geom_line, geom_abline, ggtitle, ggsave
from plotnine.data import economics
from utils_htl import cal_f_value

sys.path.append(os.path.join(os.getcwd(), '..'))
from utils_htl import tune_regularization

if __name__ == "__main__":
    ####################################
    # Benchmark 1: Logistic Regression with RNN&MF features
    ####################################

    ### Read data
    features = np.load('data/features.npy')
    feature_names = np.load('data/feature_names.npy', allow_pickle=True)
    label = np.load('data/label.npy')
    product_id = np.load('data/product_id.npy')
    product_df = pd.DataFrame(data=features, columns=feature_names)
    product_df['label'] = label
    product_df['product_id'] = product_id
    drop_cols = ['label', 'product_id']

    ### Generate Interaction terms
    for i in range(25):
        product_df[f'interaction_{i}'] = product_df[f'nnmf_user_{i}'] * product_df[f'nnmf_product_{i}']

    ### training / testing dataset splitting
    # Remove data without true labels
    df = product_df[product_df['label'] != -1]
    train_val_df, test_df = train_test_split(df, train_size=0.8, random_state=42)
    train_df, val_df = train_test_split(train_val_df, train_size=0.8, random_state=43)
    train_df, val_df, test_df = train_df.dropna(), val_df.dropna(), test_df.dropna()

    Y_train, Y_val = train_df['label'].astype(int).astype(float), val_df['label'].astype(int).astype(float)
    X_train, X_val = train_df.drop(drop_cols, axis=1), val_df.drop(drop_cols, axis=1)
    X_test, Y_test = test_df.drop(drop_cols, axis=1), test_df['label'].astype(int).astype(float)

    # Initialize a scaler on the training data, then apply it to all datasets for standardization
    scaler = StandardScaler() 
    X_train_s = scaler.fit_transform(X_train)
    X_train, X_val = pd.DataFrame(X_train_s, columns=X_train.columns), pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    test_products = test_df['product_id']
    test_labels = test_df['label'].astype(int).astype(float)

    ### Training
    # Regularization parameter
    C_values = np.linspace(0.01, 10, 10)

    best_C, best_log_loss, best_model = tune_regularization(X_train, Y_train, X_val, Y_val, C_values)
    print('Training Logistic Regression Models...')
    print(f'Best Regularization Parameter: {best_C}, Best Validation Log Loss: {best_log_loss}')

    ### Save the model parameter and the prediction results
    test_preds = best_model.predict_proba(X_test)[:,1]
    best_model_coeff = best_model.coef_
    best_model_inter = best_model.intercept_
    dirname = 'predictions_logistic'
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    np.save(os.path.join(dirname, 'product_ids.npy'), test_products)
    np.save(os.path.join(dirname, 'predictions.npy'), test_preds)
    np.save(os.path.join(dirname, 'labels.npy'), test_labels)
    np.save(os.path.join(dirname, 'coefficients.npy'), best_model_coeff)
    np.save(os.path.join(dirname, 'intercepts.npy'), best_model_inter)

    ### Performance Evaluation
    # Load Data
    logistic_df = pd.DataFrame({
        'product_id': np.load('predictions_logistic/product_ids.npy'),
        'prediction_gbm': np.load('predictions_logistic/predictions.npy'),
        'label': np.load('predictions_logistic/labels.npy')
    })
    # 
    true_label = logistic_df['label']
    print("True Label Distribution:", true_label.value_counts())
    # gbm
    pred_logistic = (logistic_df['prediction_gbm'] > 0.5).astype(int)
    print(cal_f_value(pred_logistic, true_label, "Logistic"))

    ### Create and save the ROC plot
    if not os.path.exists('benchmark_comparison'):
        os.makedirs('benchmark_comparison')

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(true_label, logistic_df['prediction_gbm'])
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    # Compute ROC AUC
    roc_auc = auc(fpr, tpr)
    roc_plot = (ggplot(roc_df, aes(x='FPR', y='TPR')) +
                geom_line(color='blue') +
                geom_abline(intercept=0, slope=1, color='black', linetype='dashed') +
                ggtitle(f'ROC Curve (AUC = {roc_auc:.2f})'))

    ggsave(plot=roc_plot, filename='roc_plot_logistic.png', path='benchmark_comparison')

    ####################################
    # Benchmark 2: RNN
    ####################################

    ### Performance Evaluation
    # Predict results
    RNN_prediction = pd.DataFrame({'pred_prob': np.load('../feature_generation/representation_rnn/predictions/predictions.npy').flatten()})
    RNN_prediction['pred_label'] = (RNN_prediction['pred_prob'] > 0.5).astype(int)
    # True Labels for first-stage modelling
    stage_true_label = np.load('../feature_generation/representation_rnn/data/label.npy')
    # Append Label Column
    RNN_prediction['label'] = stage_true_label
    # Dataset with labels
    comparison_df = RNN_prediction[RNN_prediction['label']!=-1]

    print(cal_f_value(RNN_prediction['pred_label'], RNN_prediction['label'], "RNN"))

    ### Create and save the ROC plot
    if not os.path.exists('benchmark_comparison'):
        os.makedirs('benchmark_comparison')

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(comparison_df['label'], comparison_df['pred_prob'])
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    # Compute ROC AUC
    roc_auc = auc(fpr, tpr)
    # Create the ROC plot
    roc_plot = (ggplot(roc_df, aes(x='FPR', y='TPR')) +
                geom_line(color='blue') +
                geom_abline(intercept=0, slope=1, color='black', linetype='dashed') +
                ggtitle(f'ROC Curve (AUC = {roc_auc:.2f})'))

    ggsave(plot=roc_plot, filename='roc_plot_rnn.png', path='benchmark_comparison')
