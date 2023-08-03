"""
Train a RNN model to learn informative representations from the purchase history
"""
import os
import sys
import tensorflow as tf # Make sure to use 1.x version

sys.path.append(os.path.join(os.getcwd(), '..'))
from utils_rnn import DataReader, rnn

if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    # Parameters for the RNN model training
    # Training step is set to be 800 because of the limit of computing resources
    # Better perfomance can be obtained via more intensive model traning
    nn = rnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.001,
        lstm_size=300,
        batch_size=128,
        num_training_steps=800,
        early_stopping_steps=100,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=0.5,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=200,
        log_interval=20,
        num_validation_batches=4,
    )
    # Training Model
    nn.fit() 
    # Restore Models
    nn.restore()
    # Generate Outputs
    nn.predict()



