"""
Preprocess RNN and MF features, create department-specific logistic regression models and
HTL-logistic regression models, and evaluate the performance.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.append(os.getcwd())
from utils_htl import create_ULM_data, create_vlambda, tune_regularization

if __name__ == "__main__":
    ### Preprocess data
    # Read data
    product_id = np.load('data/product_id.npy')
    features = np.load('data/features.npy')
    feature_names = np.load('data/feature_names.npy', allow_pickle=True)
    label = np.load('data/label.npy')

    product_df = pd.DataFrame(data=features, columns=feature_names)
    product_df['product_id'] = product_id
    product_df['label'] = label

    # Dictionary to obtain aisle and department information
    products = pd.read_csv('../../data/raw/products.csv')
    product_to_aisle = dict(zip(products['product_id'], products['aisle_id']))
    product_to_department = dict(zip(products['product_id'], products['department_id']))
    product_to_name = dict(zip(products['product_id'], products['product_name']))
    # Map aisle and department information to the product_df
    product_df['aisle_id'] = product_df['product_id'].map(product_to_aisle)
    product_df['department_id'] = product_df['product_id'].map(product_to_department)

    # Count the number of samples in each aisle and department
    aisle_counts = product_df['aisle_id'].value_counts().tolist()
    department_counts = product_df['department_id'].value_counts().tolist()
    print("aisle_counts:", aisle_counts)
    print("department_counts:", department_counts)
    # Assign department_id to each product in the dataframe
    product_df = product_df.assign(department_id=product_df['product_id'].map(product_to_department))
    drop_cols = ['label','product_id', 'label', 'aisle_id','department_id']

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



    ### Build department-specific models
    department_models = {}
    C_values = [0.1] 

    # set the `rank_dep_of_interest`th largest department (deli) as the target department
    # and the rest as the source departments
    rank_dep_of_interest = 9
    common_departments = product_df['department_id'].value_counts().nlargest(rank_dep_of_interest).index.tolist()

    # Initialize lists to hold source and target data
    X, y = [], []
    X0, y0 = None, None

    for i, department in enumerate(common_departments):
        # Filter the dataframe to only include rows from the current department
        print(f"Results for ({i+1}th largest) {department}:")
        depart_train_df = train_df[train_df['department_id'] == department]
        depart_val_df = val_df[val_df['department_id'] == department]
        
        Y_train, Y_val = depart_train_df['label'].astype(int).astype(float), depart_val_df['label'].astype(int).astype(float)
        X_train, X_val = depart_train_df.drop(drop_cols, axis=1), depart_val_df.drop(drop_cols, axis=1)
        
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

        print(f"Input Data size:{X_train.shape}")
        
        if i == (rank_dep_of_interest - 1):
            # Save the ith department's data as the target data
            X0, y0 = X_train, Y_train
        else:
            # Save the other departments' data as the source data
            X.append(X_train)
            y.append(Y_train)
        
        # Fit the department-specific logistic regression model
        best_C, best_log_loss, best_model = tune_regularization(X_train, Y_train, X_val, Y_val, C_values)
        print(f"Best_log_loss: {best_log_loss}")

        # Store the best model in each department and its details in a dictionary
        department_models[department] = {'best_C': best_C, 'best_log_loss': best_log_loss, 'best_model': best_model}



    ### Build HTL models
    # Create HTL-structured data
    ULM_data = create_ULM_data(y, y0, X, X0)
    vlambda = create_vlambda(X, X0) # Regularization vector, based on theoretical optimal values
    X_ULM, y_ULM = ULM_data['X_ULM'] / vlambda, ULM_data['y_ULM'] # Add feature-specific regularization 

    # Fit the HTL-Logistic Regression model
    log_reg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', class_weight={0:1, 1:5})
    log_reg.fit(X_ULM, y_ULM)

    # Extract the coefficients for the target task
    HTL_coef = log_reg.coef_[:, -X0.shape[1]:]
    HTL_intercept = log_reg.intercept_



    ### Compare the performance in the testing dataset
    # Extract Coefficients
    department = common_departments[rank_dep_of_interest-1]

    # Calculate Predictions
    depart_test_df = test_df[test_df['department_id'] == department]
    depart_X_test = depart_test_df.drop(drop_cols, axis=1)
    depart_X_test = pd.DataFrame(scaler.transform(depart_X_test), columns=depart_X_test.columns)
    depart_Y_test = depart_test_df['label'].astype(int).astype(float)

    # Performance of single-task (fine-tuned) method
    coeff1 = department_models[department]['best_model'].coef_.flatten()
    intercept1 = department_models[department]['best_model'].intercept_
    depart_pred_prob = 1 / (1 + np.exp(-(np.dot(depart_X_test, coeff1) + intercept1)))
    logloss_single_task = log_loss(depart_Y_test, depart_pred_prob)

    # Performance of global method
    coeff_pool = np.load("predictions_logistic/coefficients.npy").flatten()
    intercept_pool = np.load("predictions_logistic/intercepts.npy")
    pool_pred_prob = 1 / (1 + np.exp(-(np.dot(depart_X_test, coeff_pool) + intercept_pool)))
    logloss_global = log_loss(depart_Y_test, pool_pred_prob)

    # Performance of HTL method
    coeffHTL = HTL_coef.flatten()
    HTL_pred_prob = 1 / (1 + np.exp(-(np.dot(depart_X_test, coeffHTL) + HTL_intercept)))
    logloss_htl = log_loss(depart_Y_test, HTL_pred_prob)

    # Save performance comparison
    df = pd.DataFrame({
        'Model': ['Single Task', 'Global', 'HTL'],
        'Log Loss': [logloss_single_task, logloss_global, logloss_htl]
    })
    if not os.path.exists('model_comparison'):
        os.makedirs('model_comparison')
    df.to_csv(f'model_comparison/{department}_logloss_comparison.csv', index=False)
