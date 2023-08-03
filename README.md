# HTL-NextBasketRec
Heterogeneous Transfer Learning (HTL) method for next-basket recommendation tasks.

## Instructions
To obtain the result, follow these steps:

1. Create a user-wise dataset and a user-product pair dataset:
    ```shell
    cd preprocessing
    python preprocessing.py
    ```

2. Train a MF model to learn informative representations from the purchase history:
    ```shell
    cd ../models/features/representation_mf
    python prepare_mf.py
    python train_mf
    ```

3. Train a Matrix Factorization model to learn informative representations from the user-product interaction:
    ```shell
    cd ../representation_rnn
    python prepare_rnn.py
    python train_rnn.py
    ```

4. Model building and evaluation:
    ```shell
    cd ../../model
    python prepare_htl.py
    # Train a benchmark model to illustrate the performance of feature-based fine-tuning
    python benchmark.py
    # Train a HTL model and compare it with fine-tuned models and pooled models
    python train_htl.py
    ```
