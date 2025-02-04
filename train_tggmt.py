import os
import argparse
import tensorflow as tf
from utils.common import JSONHandler
from utils.common import Log
from trainer import TGGMTTrainer

# Running eagerly can introduce significant overhead, especially in distributed training scenarios 
# (e.g., when using MirroredStrategy or MultiWorkerMirroredStrategy).
# Enable Eagle mode to run on kaggle platform (.numpy() will be available on tensors)
tf.config.run_functions_eagerly(True)


# Enable Datapipe line debug
# tf.data.experimental.enable_debug_mode()

def train(config):
    """
    Simulated training function that logs training progress.

    This function initiates the training process by setting up logging based on the configuration and verbosity level,
    then instantiates a `TGGMTTrainer` object to handle the training process according to the specified configuration.
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
    2. Instantiates a `TGGMTTrainer` object, passing in the `config` and `logger`.
    3. Calls the `train` method of `TGGMTTrainer` to begin the training process.
    4. Logs a final message upon successful completion of training.

    Example:
    --------
    Assuming a configuration dictionary with necessary paths and parameters:
    >>> train(config)

    Notes:
    ------
    - The `Log` and `TGGMTTrainer` classes must be predefined. `Log` should support `info` and other logging methods,
      while `TGGMTTrainer` should implement a `train` method to conduct the actual training.
    - `train.log` will be saved in the `output_path` directory as specified in the `config`.
    """

    # Set up logging
    logger = Log(log_file=os.path.join(config.get("output_path"), "train.log"), verbose=config.get("verbose"))
    
    # Instantiate trainer
    trainer = TGGMTTrainer(config, logger=logger)
        
    # Start training
    trainer.train()

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

    # Start training
    train(config)

if __name__ == '__main__':
    main()
