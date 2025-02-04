import os
import glob
import numpy as np
import tensorflow as tf
from utils import TokenizerFactory
from utils import ScoreCalculator
from utils import JSONHandler




class TransformerModelTester:
    def __init__(self, config, tokenizer=None, tflite_model_filename=None, keras_model_path=None, model_loader=None, logger=None):
        self.config = config
        self.model_loader = model_loader
        self.logger = logger


        # Set up tokenizer
        if tokenizer is None:
            self.tokenizer = TokenizerFactory.create_tokenizer(config)
        else:
            self.tokenizer = tokenizer

        self.prediction_fn = None
        if tflite_model_filename is not None:
            if not os.path.exists(tflite_model_filename):
                raise ValueError("tflite file doesn't exist!")
        
            print("Using TFLite Model for Inference")

            interpreter = tf.lite.Interpreter(tflite_model_filename)
            REQUIRED_SIGNATURE = "serving_default"

            self.rev_character_map = {j:i for i,j in self.tokenizer.token_map.items()}
            found_signatures = list(interpreter.get_signature_list().keys())
            if REQUIRED_SIGNATURE not in found_signatures:
                raise KernelEvalException('Required input signature not found.')

            self.prediction_fn = interpreter.get_signature_runner("serving_default")
            self.logger.info("Initialized or loaded TFLite Model")

        self.keras_model = None
        if keras_model_path is not None:
            if os.path.exists(keras_model_path):
                if "model" ==  os.path.basename(keras_model_path):
                    raise ValueError(f"Remove model folder in path! {keras_model_path}")
                print("Using Keras Model for Inference")

                keras_model_path = os.path.join(keras_model_path, "model")
                # Load the Keras model from the .h5 file
                config = JSONHandler.read_json(os.path.join(keras_model_path, "config.json"))
                                      
                self.keras_model = self.model_loader.load_model(config, compile=False)
                self.logger.info("Initialized or loaded Keras Model")

        if tflite_model_filename is None and keras_model_path is None:
            self.logger.info("keras_model_path : ", keras_model_path, "tflite_model_filename", tflite_model_filename)
            raise ValueError("Model paths cannot be None")
        
    def set_keras_model(self, model):
        # For testing models
        self.keras_model = model

    def predict_with_tflite(self, source):
            return self.prediction_fn(inputs=source[0])
        
    def test_tflite_model(self, source, target, threshold=0.0, display=False, score=False):
        if self.prediction_fn is None:
            raise ValueError("Initailized TFLite model first by providing its path in __init_()!")
        out_dict ={}
        
        def display_pred(source, target):
            correct_count, distance_score = 0, 0.0
             # Convert the tensor to a NumPy array or list
            phrase_as_numpy = tf.reshape(target, [-1]).numpy()

            target_text = "".join([self.tokenizer.reverse_token_map[_] for _ in phrase_as_numpy])
            # Predict sequence
            output = self.prediction_fn(inputs=source[0])
            prediction_str = "".join([self.rev_character_map.get(s, "") for s in np.argmax(output["outputs"], axis=1)])

            target_text = self.tokenizer.clean_text(target_text)
            prediction_str = self.tokenizer.clean_text(prediction_str)

            if score and threshold:
                distance_score = ScoreCalculator.distance_score(prediction_str, 
                                                                target_text
                                                                )
                
                if distance_score <= threshold:
                    correct_count+=1

            if display:
                if score:
                    self.logger.info(f"threshold:     {threshold}")
                    self.logger.info(f"distance:     {distance_score}")
                self.logger.info(f"target:     {target_text}")
                self.logger.info(f"prediction: {prediction_str}\n")



            out_dict[prediction_str] = target_text

            return out_dict, correct_count, distance_score

        return display_pred(source, target)
    
    def test_keras_model(self, source, target, threshold=0.0, display=False, score=False):
        """
        inputs : 
                source : landmarks of size (sequence_length, number of features)
                target : corresponding text tokenized sequence
        output :
                dict : paired prediction as key and corresponding text as value
        """
        if self.keras_model is None:
            raise ValueError("Initailized keras model first by providing its path in __init_()!")
        
        out_dict = {}
        
        def display_pred(source, target):
            correct_count, distance_score = 0, 0.0
            prediction_str = ""
            target_text = ""
            preds = self.keras_model.predict_source(source, self.tokenizer.bos_token_idx).numpy()
            bs = 1
            
            for j in range(bs):
                target_text = "".join(self.tokenizer.reverse_token_map[int(idx.numpy())] for idx in target[j, :])
                prediction = "".join(self.tokenizer.reverse_token_map[idx] for idx in preds[j, :] if idx != self.tokenizer.eos_token_idx)
                
                # Remove special tokens
                prediction_str = self.tokenizer.clean_text(prediction)
                target_text = self.tokenizer.clean_text(target_text)

                if score:
                    distance_score = ScoreCalculator.distance_score(prediction_str, 
                                                                    target_text
                                                                    )
                    if distance_score <= threshold:
                        correct_count+=1
                    
                if display:
                    if score:
                        self.logger.info(f"threshold:     {threshold}")
                        self.logger.info(f"distance_score:     {distance_score}")
                    self.logger.info(f"target:     {target_text}")
                    self.logger.info(f"prediction: {prediction_str}\n")

                # insert
                out_dict[prediction_str] = target_text

            return out_dict, correct_count, distance_score

        return display_pred(source, target)