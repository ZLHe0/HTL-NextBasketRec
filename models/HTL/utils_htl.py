"""
Utility functions for HTL training and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_curve, auc
from plotnine import ggplot, aes, geom_line, geom_abline, ggtitle

# Create the the HTL-transformed data
def create_ULM_data(y, y0, X, X0):
    K = len(X)
    
    # Add X0 to the end of the list
    Xapp = X.copy()
    Xapp.append(X0)
    
    # Get the dimensions of each matrix in the list
    dimensions = [(x.shape[0], x.shape[1]) for x in Xapp]
    
    # Create a list of matrices filled with zeros
    zeros_list = [np.zeros((d[0], d[1])) for d in dimensions]
    
    # Initialize an empty list to store the final matrix
    X_final = []
    
    # Loop through each element in the list, adding the appropriate matrices
    for i in range(K + 1):
        row_list = []
        for j in range(K + 1):
            if j == i:
                row_list.append(Xapp[i])
            elif j == K:
                row_list.append(Xapp[i])
            else:
                row_list.append(zeros_list[i])
        X_final.append(np.hstack(row_list))  # column bind in python is achieved through np.hstack
    
    # Combine the rows to get the final matrix
    X_final = np.vstack(X_final)  # row bind in python is achieved through np.vstack
    
    ### Stack y
    y.append(y0)
    y_final = np.concatenate(y)  # concatenate in python is achieved through np.concatenate
    
    return {'y_ULM': y_final, 'X_ULM': X_final}

# Calculate the theory-motivated regularization vector
def create_vlambda(X, X0):
    N = sum([x.shape[0] for x in X]) + X0.shape[0]
    
    # Calculate the theoretical penalty values n_k/N * (\sqrt{\log p / n_k}) for each block
    values = [x.shape[0]/N * np.sqrt(np.log(x.shape[1]) / x.shape[0]) for x in X]
    
    # Create the result vector by repeating each value n_k times
    result = np.concatenate([np.repeat(values[i], X[i].shape[1]) for i in range(len(X))])
    
    # For the target task estimation problem, set lambda to be the global one (\sqrt{\log p/N})
    global_lambda = np.sqrt(np.log(X0.shape[1]) / N)
    result = np.concatenate([result, np.repeat(global_lambda, X0.shape[1])])
    
    return result

# Function to tune the regularization parameter of a logistic regression model
def tune_regularization(X_train, Y_train, X_val, Y_val, C_values):
    """    
    Parameters:
    X_train: Training data features
    Y_train: Training data labels
    X_val: Validation data features
    Y_val: Validation data labels
    C_values: List of regularization parameters to try
    
    Returns:
    best_C: The regularization parameter that resulted in the lowest log loss on the validation set
    best_log_loss: The best log loss achieved on the validation set
    """
    # Initialize variables to store the best log loss and corresponding regularization parameter
    best_log_loss = float('inf')
    best_C = None

    # Iterate over the regularization parameters
    for C in C_values:
        log_reg = LogisticRegression(penalty='l1', C=C, solver='liblinear', class_weight={0:1, 1:5})
        log_reg.fit(X_train, Y_train)
        # Predict the probabilities of the validation set
        val_preds = log_reg.predict_proba(X_val)[:, 1]
        # Calculate the log loss on the validation set
        val_log_loss = log_loss(Y_val, val_preds)
        print(f'Regularization Parameter: {C}, Validation Log Loss: {val_log_loss}')

        # Update the best log loss and regularization parameter
        if val_log_loss < best_log_loss:
            best_log_loss = val_log_loss
            best_C = C
            best_model = log_reg

    return best_C, best_log_loss, best_model

# Calculate recall, precision and F-score
def cal_f_value(pred, label, model):
    TP = sum((label==1) & (pred==1))
    FP = sum((pred==1) & (label==0))
    FN = sum((pred==0) & (label==1))
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F_value = 2 / (1/recall + 1/precision)
    summary = [model, recall, precision, F_value]
    summary_name = ["model", "recall", "precision", "F_value"]
    return(pd.DataFrame(dict(zip(summary_name, summary)), index=[0]))

# Plot ROC curve and calculate AUC score
def plot_roc(true_label, predicted_probabilities):
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(true_label, predicted_probabilities)
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})

    # Compute ROC AUC
    roc_auc = auc(fpr, tpr)

    # Create the ROC plot
    roc_plot = (ggplot(roc_df, aes(x='FPR', y='TPR')) +
                geom_line(color='blue') +
                geom_abline(intercept=0, slope=1, color='black', linetype='dashed') +
                ggtitle(f'ROC Curve (AUC = {roc_auc:.2f})'))

    print(roc_plot)
    print(roc_df)


