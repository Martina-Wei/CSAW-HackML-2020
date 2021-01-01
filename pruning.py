import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from loader import load_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("bd_model", help="bad model")
parser.add_argument("-v", "--validate", dest="validate_data_path")
parser.add_argument("-t", "--test", dest="test_data_path")
args = parser.parse_args()

def main():
    model = keras.models.load_model(args.bd_model)
    origin_model = keras.models.load_model(args.bd_model)

    x_vali, y_vali = load_data(args.validate_data_path)
    
    batch_size = 128
    epochs = 2
    num_images = len(y_vali)
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    model_for_pruning.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    logdir = tempfile.mkdtemp()

    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]
  
    model_for_pruning.fit(x_vali, y_vali,
                    batch_size=batch_size, epochs=epochs, validation_split=0.1,
                    callbacks=callbacks)

    
    # Test Accuracy

    x_test, _ = load_data(args.test_data_path)
    model_for_pruning_label = np.argmax(model.predict(x_test), axis=1).reshape(-1)
    print("accuracy {}/ {}".format(np.sum(model_for_pruning_label!=0), len(y_vali)))

if __name__ == '__main__':
    main()
