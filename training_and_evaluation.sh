### To obtain the result, run the following python files sequentially

# Create a user-wise dataset and a user-product pair dataset
cd preprocessing
python preprocessing.py

# Train a MF model to learn informative representations from the purchase history
cd ../models/features/representation_mf
python prepare_mf.py
python train_mf

# Train a Matrix Factorization model to learn informative representations from the user-product interaction
cd ../representation_rnn
python prepare_rnn.py
python train_rnn.py

# Model building and evaluation
cd ../../model
python prepare_htl.py
# Train a benchmark model to illustrate the performance of feature-based fine-tuning
python benchmark.py
# Train a HTL model and compare it with fine-tuned models and pooled models
python train_htl.py