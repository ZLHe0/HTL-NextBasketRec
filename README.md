# HTL-NextBasketRec
An implementation of the Heterogeneous Transfer Learning (HTL) method for next-basket recommendation scenarios.
> Additional details will be added after the paper is published.

## Sample Code: Model Training and Evaluation
To generate the results, please adhere to the following steps:

1. Assemble a user-wise dataset and a dataset of user-product pairs:
    ```shell
    cd preprocessing
    python preprocessing.py
    ```

2. Implement a Matrix Factorization (MF) model to derive informative representations from the user-product interaction:
    ```shell
    cd ../models/features/representation_mf
    python prepare_mf.py
    python train_mf
    ```

3. Implement a Recurrent Neural Network (RNN) model to derive informative representations from the purchase history:
    ```shell
    cd ../representation_rnn
    python prepare_rnn.py
    python train_rnn.py
    ```

4. Construct and evaluate the HTL model:
    ```shell
    cd ../../model
    python prepare_htl.py
    # Execute a benchmark model to demonstrate the efficacy of feature-based transfer learning
    python benchmark.py
    # Train the HTL-based fine-tuned model and perform comparisons with benchmark transfer learning models and pre-trained models
    python train_htl.py
    ```

## Acknowledgements
This project makes use of data and solutions from the [Instacart Market Basket Analysis competition](https://www.kaggle.com/c/instacart-market-basket-analysis) on Kaggle.
