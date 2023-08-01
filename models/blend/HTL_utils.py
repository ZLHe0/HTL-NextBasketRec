import numpy as np
import pandas as pd

# Function to create the ULM data
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


# Function to create the default penalty vector
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

