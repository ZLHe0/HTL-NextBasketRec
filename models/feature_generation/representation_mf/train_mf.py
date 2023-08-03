"""
Train a Matrix Factorization model to learn informative representations from the user-product interaction
"""
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from utils_mf import DataReader, nnmf

if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    nnmf = nnmf(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.005,
        rank=25,
        batch_size=4096,
        num_training_steps=8000,
        early_stopping_steps=1000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=0,
        min_steps_to_checkpoint=4000,
        log_interval=200,
        num_validation_batches=1,
        loss_averaging_window=200,
    )
    nnmf.fit()
    nnmf.restore()
    nnmf.predict()




