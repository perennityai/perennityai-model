import os
import argparse
import tensorflow as tf
from utils.common import JSONHandler
from utils.common import Log
from trainer import Seq2SeqTuner

# # Enable Eagle mode to run on kaggle platform
# tf.config.run_functions_eagerly(True)
# # Enable Datapipe line debug
# tf.data.experimental.enable_debug_mode()

def tune(config):
    """
    Simulated tuning function that logs tuning progress.

    This function initiates the tuning process by setting up logging based on the configuration and verbosity level,
    then instantiates a `Seq2SeqTuner` object to handle the tuning process according to the specified configuration.
    Logging information is output to a log file, and detailed logging is optionally enabled.

    Parameters:
    ----------
    config : dict
        A dictionary containing tuning configuration options, such as the path for saving outputs, tuning parameters,
        and model specifications. This dictionary should at least contain the key `"output_path"` to specify where to
        save the tuning logs.

    Workflow:
    ---------
    1. Initializes a `Log` instance for handling log output based on the given `output_path` and `verbose` flag.
    2. Instantiates a `Seq2SeqTuner` object, passing in the `config` and `logger`.
    3. Calls the `tune` method of `Seq2SeqTuner` to begin the tuning process.
    4. Logs a final message upon successful completion of tuning.

    Example:
    --------
    Assuming a configuration dictionary with necessary paths and parameters:
    >>> tune(config)

    Notes:
    ------
    - The `Log` and `Seq2SeqTuner` classes must be predefined. `Log` should support `info` and other logging methods,
      while `Seq2SeqTuner` should implement a `tune` method to conduct the actual tuning.
    - `tune.log` will be saved in the `output_path` directory as specified in the `config`.
    """


    # Set up logging
    logger = Log(log_file=os.path.join(config.get("output_path"), "seq2seq_tuner.log"), verbose=config.get("verbose"))

    # Instantiate tuneer
    tuner = Seq2SeqTuner(config, logger=logger)

    # Start tuning
    tuner.tune()

    logger.info("tuning completed successfully.")


def parse_ini_to_dict(file_path):
    """Reads an .ini file and parses all sections into a single dictionary."""
    import configparser
    config = configparser.ConfigParser()
    config.read(file_path)
    
    # Convert each section to a dictionary
    parsed_data = {section: dict(config.items(section)) for section in config.sections()}
    
    return parsed_data


def parse_yaml_to_dict(file_path):
    """Reads a YAML file and parses all content into a single dictionary."""
    import yaml
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def cast_strings_to_numbers(data):
    """
    Cast string values in a dictionary to int or float, if possible.
    
    Parameters:
        data (dict): A dictionary with string values that may represent numbers.
        
    Returns:
        dict: A new dictionary with numeric values casted where possible.
    """
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            try:
                # Check if the value is an integer
                if value.isdigit():
                    converted_data[key] = int(value)
                else:
                    # Try casting to float for values that may be floats
                    converted_data[key] = float(value)
            except ValueError:
                # Leave the value as it is if casting fails
                converted_data[key] = value
        else:
            # Keep non-string values as they are
            converted_data[key] = value
    
    return converted_data

def main():
    parser = argparse.ArgumentParser(description="tune a machine learning model with specified configuration.")
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

    
    # Start tuning
    tune(config)

if __name__ == '__main__':
    main()
