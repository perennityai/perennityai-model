import os
import re
import glob
import json
import argparse
import numpy as np
import tensorflow as tf
from utils.common import JSONHandler
from utils.common import Log
from utils import TokenizerFactory
from data_loader import GGMTTFRecordDataset
from trainer import SimpleSeq2SeqTrainer
from trainer import NoAttenSeq2SeqTrainer
from trainer import AttenSeq2SeqTrainer

# Running eagerly can introduce significant overhead, especially in distributed training scenarios 
# (e.g., when using MirroredStrategy or MultiWorkerMirroredStrategy).
# Enable Eagle mode to run on kaggle platform
# tf.config.run_functions_eagerly(True)


# Enable Datapipe line debug
# tf.data.experimental.enable_debug_mode()

def train(config):
    """
    Simulated training function that logs training progress.

    This function initiates the training process by setting up logging based on the configuration and verbosity level,
    then instantiates a `Seq2SeqTrainer` object to handle the training process according to the specified configuration.
    Logging information is output to a log file, and detailed logging is optionally enabled.

    Parameters:
    ----------
    config : dict
        A dictionary containing training configuration options, such as the path for saving outputs, training parameters,
        and model specifications. This dictionary should at least contain the key `"output_path"` to specify where to
        save the training logs.

    Workflow:
    ---------
    1. Initializes a `Log` instance for handling log output based on the given `output_path` and `verbose` flag.
    2. Instantiates a `Seq2SeqTrainer` object, passing in the `config` and `logger`.
    3. Calls the `train` method of `Seq2SeqTrainer` to begin the training process.
    4. Logs a final message upon successful completion of training.

    Example:
    --------
    Assuming a configuration dictionary with necessary paths and parameters:
    >>> train(config)

    Notes:
    ------
    - The `Log` and `Seq2SeqTrainer` classes must be predefined. `Log` should support `info` and other logging methods,
      while `Seq2SeqTrainer` should implement a `train` method to conduct the actual training.
    - `train.log` will be saved in the `output_path` directory as specified in the `config`.
    """

    # Set up logging
    logger = Log(log_file=os.path.join(config.get("output_path"), "seq2seq_train.log"), verbose=config.get("verbose"))

    # Create loader
    data_loader = GGMTTFRecordDataset(config=config, logger=logger)

    

    if config.get("dev_mode", 0):
        # Example usage
        predictions = ["hello", "how are you", "good morning", "good night", "thank you"]
        actual_labels = ["hola", "cómo estás", "buenos días", "buenas noches", "gracias"]
    else:
        # Sample training data (input-output pairs)
        actual_labels, predictions = data_loader.get_inference_data()

    logger.info("actual_labels : ", len(actual_labels))
    logger.info("predictions : ", len(predictions))

    # Given inference output, give me the actual value.
    dataset = [predictions, actual_labels]

    if config.get("seq2seq_model_type") == 'simple_seq2seq':
        # Instantiate trainer1
        trainer = SimpleSeq2SeqTrainer(config, name="simple_seq2seq", logger=logger)
    elif config.get("seq2seq_model_type") == 'noatten_seq2seq':
        # Instantiate trainer2
        trainer = NoAttenSeq2SeqTrainer(config, name="noatten_seq2seq", logger=logger)
    elif config.get("seq2seq_model_type") == 'multi_atten_seq2seq':
        # Instantiate trainer3
        trainer = AttenSeq2SeqTrainer(config, name="multi_atten_seq2seq", logger=logger)
    else:
        raise ValueError("Unknown seq2seq")

    # Start training
    epochs = config.get("atten_seq2seq_epochs") if "multi_atten" in trainer.name else config.get("seq2seq_epochs")
    # Train each trainer
    trainer.train(dataset, 
                  batch_size=config.get("seq2seq_batch_size"), 
                  epochs=epochs, 
                  validation_split=config.get("seq2seq_validation_split"))
    
    logger.info("Training completed successfully.")



def parse_ini_to_dict(file_path):
    """Reads an .ini file and parses all sections into a single dictionary."""
    import configparser
    config = configparser.ConfigParser()
    config.read(file_path)

    parsed_data = {}
    for section in config.sections():
        
        if 'all_features' == section:
            parsed_data['feature_columns'] = config[section]['ALL_FEATURE_COLUMNS'].split("\t")
        else:
             # Convert each section to a dictionary
            parsed_data[section] =  dict(config.items(section))

    return parsed_data

def parse_yaml_to_dict(file_path):
    """Reads a YAML file and parses all content into a single dictionary."""
    import yaml
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def main():
    parser = argparse.ArgumentParser(description="Train a machine learning model with specified configuration.")
    parser.add_argument('--config', required=True, help="Path to the configuration file (YAML format).")
    
    args = parser.parse_args()

    # Load configuration
    if '.json' in args.config:
        config = JSONHandler.read_json(args.config)
    elif '.ini' in args.config:
        config = parse_ini_to_dict(args.config)
    elif '.yaml' in args.config:
        config = parse_yaml_to_dict(args.config)
    else:
        raise ValueError(f"Unknown config file format! {args.config}")
    
    cpu_count_to_use = config.get('cpu', 2)
    os.environ['OMP_NUM_THREADS'] = str(cpu_count_to_use)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(cpu_count_to_use)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(cpu_count_to_use)

    # Set inter and intra-op thread numbers
    tf.config.threading.set_inter_op_parallelism_threads(cpu_count_to_use)
    tf.config.threading.set_intra_op_parallelism_threads(cpu_count_to_use)


    # List all physical CPU devices
    physical_devices = tf.config.list_physical_devices('CPU')
    print("Physical Devices:", physical_devices)

    # If there are CPUs, attempt to expose all logical CPUs (usually 8 for c3.2xlarge)
    if physical_devices:
        try:
            tf.config.set_logical_device_configuration(
                physical_devices[0], 
                [tf.config.LogicalDeviceConfiguration()] * cpu_count_to_use  # Expose all 8 logical devices
            )
            logical_devices = tf.config.list_logical_devices('CPU')
            print("Logical Devices:", logical_devices)
        except RuntimeError as e:
            print("Error setting logical devices:", e)
    else:
        print("No CPUs detected.")

    
        
    # Execute with XLA compilation enabled
    use_jit_compile = bool(config.get("use_xla_accelerator"))
    if use_jit_compile:
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_cpu_global_jit"
        os.environ['cpu_global_jit'] = str(config.get("use_xla_accelerator"))
        # - use experimental_jit_scope, or
        # - use tf.function(jit_compile=True).
    
    # Apply the decorator dynamically
    if use_jit_compile:
        @tf.function(jit_compile=True)
        def xla_inner_compute(config):
            train(config)
    else:
        @tf.function(jit_compile=False)
        def xla_inner_compute(config):
            train(config)
    
    # Todo: Fix xla_inner_compute(config)
    # RuntimeError: Detected a call to `Model.fit` inside a `tf.function`. 
    # `Model.fit is a high-level endpoint that manages its own `tf.function`. 
    # Please move the call to `Model.fit` outside of all enclosing `tf.function`s. 
    # Note that you can call a `Model` directly on `Tensor`s inside a `tf.function` like: `model(x)`.
    
    # Start training
    train(config)

if __name__ == '__main__':
    main()
