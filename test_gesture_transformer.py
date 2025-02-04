import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.common import JSONHandler
from utils.common import Log
from utils import TokenizerFactory
from models import TransformerModelFactory
from tester import TransformerModelTester
from utils import ScoreCalculator
from data_loader import GGMTTFRecordDataset



def test(config):
    """
    Simulated testing function that logs testing progress.

    This function initiates the testing process by setting up logging based on the configuration and verbosity level,
    then instantiates a `Gesturetester` object to handle the testing process according to the specified configuration.
    Logging information is output to a log file, and detailed logging is optionally enabled.

    Parameters:
    ----------
    config : dict
        A dictionary containing testing configuration options, such as the path for saving outputs, testing parameters,
        and model specifications. This dictionary should at least contain the key `"output_path"` to specify where to
        save the testing logs.

    Workflow:
    ---------
    1. Initializes a `Log` instance for handling log output based on the given `output_path` and `verbose` flag.
    2. Instantiates a `Gesturetester` object, passing in the `config` and `logger`.
    3. Calls the `test` method of `Gesturetester` to begin the testing process.
    4. Logs a final message upon successful completion of testing.

    Example:
    --------
    Assuming a configuration dictionary with necessary paths and parameters:
    >>> test(config)

    Notes:
    ------
    - The `Log` and `Gesturetester` classes must be predefined. `Log` should support `info` and other logging methods,
      while `Gesturetester` should implement a `test` method to conduct the actual testing.
    - `test.log` will be saved in the `output_path` directory as specified in the `config`.
    """

     # Define variable to hold output data
    output_dict = {}

    num_test_sample = config.get("num_test_sample") # set 2M  for all

    output_path = config.get("output_path")
    # Set up logging
    logger = Log(log_file=os.path.join(config.get("output_path"), "test.log"), name="TEST_GESTURE::", verbose=config.get("verbose"))
    # Set up tokenizer
    tokenizer = TokenizerFactory.create_tokenizer(config)

    # Define model path
    keras_model_path = os.path.join(output_path, "model") if config.get("test_keras_model") else None

    # Define TFLite filename
    tflite_model_filename = os.path.join(output_path,  "model.tflite")  if config.get("test_tflite_model") else None

    # Instantiate model tester
    model_tester = TransformerModelTester(config, tokenizer=tokenizer, 
                               tflite_model_filename=tflite_model_filename, 
                               keras_model_path=keras_model_path, 
                               model_loader=TransformerModelFactory,
                               logger=logger)

    # Setup test data
    data_loader = GGMTTFRecordDataset(config=config, model_tester=model_tester, logger=logger)

    # Get all files
    if config.get("test_using_supplement_dataset", 0):
        logger.info("Using supplemental dataset for testing!")
        dataset_files = glob.glob(os.path.join(config.get("supplement_dataset_path"), "*.tfrecord"))
    else:
        logger.info("Using non-supplemental dataset for testing!")
        dataset_files = glob.glob(os.path.join(config.get("dataset_path"), "*.tfrecord"))
    
    # dataset_files = [dataset_files[-1]]

    # Prepare dataset
    test_len = config.get("train_len")
    test_ds = data_loader.create_padded_dataset(tfrecord_files=dataset_files[:test_len], batch_size=1, shuffle=True)
    eval_ds = data_loader.create_padded_dataset(tfrecord_files=dataset_files[test_len:], batch_size=1, shuffle=True)
    logger.info("Loaded the index -1 of the entire dataset files")

    threshold = float(config.get('threshold', 0.0) if not None else  config.get('test_distance_score_threshold', 0.0))
    
    logger.info("Score threshold : ", threshold)

    if config.get("test_keras_model", 0):
        correctly_classified = 0
        try:
            # Loop thrould test dataset and print predictions
            for i, (source,  phrase, sim_score) in enumerate(test_ds):
                # # Test reverse
                # if np.random.uniform(0, 1) < 0.5:
                #     logger.info("Reverse : ", source.shape)
                #     source = tf.reverse(source, axis=[1])
                #     phrase = tokenizer.reverse_unbatched_token(phrase)

                results, correct, score = model_tester.test_keras_model(source, 
                                                               phrase, 
                                                               threshold=threshold, 
                                                               display=True, 
                                                               score=True)
                correctly_classified+=correct

                if i % 100 == 0:
                    logger.info(f"Processed {i} out of : {num_test_sample}, correctly_classified : {correctly_classified}")

                output_dict.update(results)
                # Condition to break
                if num_test_sample == i-1:
                    accuracy = float(correctly_classified/num_test_sample) * 100.0
                    accuracy_str = f"Test Accuracy : {accuracy}"
                    output_dict[accuracy_str] = accuracy_str

                    logger.info(f"Processed {num_test_sample} out of : {num_test_sample}, {accuracy_str}")
                    break
        except Exception as e:
            logger.info(f"Exception: {e}")

    if config.get("test_tflite_model", 0):
            correctly_classified = 0
            for i, (source,  phrase, sim_score) in enumerate(test_ds):
                # # Test reverse
                # if np.random.uniform(0, 1) < 0.5:
                #     source = tf.reverse(source, axis=[1])
                #     phrase = tokenizer.reverse_unbatched_token(phrase)
                #     logger.info("Reverse phrase : ", phrase.shape)

                results, correct, score = model_tester.test_tflite_model(source, 
                                                                phrase, 
                                                                threshold=threshold, 
                                                                display=True, 
                                                                score=True)
                correctly_classified+=correct
                
                output_dict.update(results)
                # Condition to break
                if num_test_sample == i-1:
                    accuracy = float(correctly_classified/num_test_sample) * 100.0
                    accuracy_str = f"Test Accuracy : {accuracy}"

                    logger.info(f"Processed {num_test_sample} out of : {num_test_sample}, {accuracy_str}")
                    break

            # Define Evaluation file
            evaluation_file = os.path.join(output_path, "evaluation_results.json")

            # Save the JSON to a file
            JSONHandler.write_json(evaluation_file, json.dumps(output_dict))

    logger.info("Testing completed successfully.")

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

def main():
    parser = argparse.ArgumentParser(description="test a machine learning model with specified configuration.")
    parser.add_argument('--config', required=True, help="Path to the configuration file (YAML format).")
    parser.add_argument('--threshold', required=False, help="Prediction to target distance score")
    
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
    if args.threshold:
        config['threshold'] = float(args.threshold)

    # Start testing
    test(config)

if __name__ == '__main__':
    main()
