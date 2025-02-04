import os
import re
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from .score_calculator import ScoreCalculator



from utils.gesture_recognition import TokenizerFactory

class CalculateCombinedMetrics:
    def __init__(self, config={}, 
                 is_bleu_scores=True,
                 is_rouge1_scores=True,
                 is_rougeL_scores=True,
                 is_wer_scores=True,
                 is_cer_scores=True,
                 is_edit_dist_scores=True,
                 is_meteor_scores=True,
                 is_damerau_dist_scores=True, 
                 fixed_weights={},
                 logger=None):
        
        # Set Attrubutes
        self.token_level = config.get("token_level")
        self.is_bleu_scores=is_bleu_scores
        self.is_rouge1_scores=is_rouge1_scores
        self.is_rougeL_scores=is_rougeL_scores
        self.is_wer_scores=is_wer_scores
        self.is_cer_scores=is_cer_scores
        self.is_edit_dist_scores=is_edit_dist_scores
        self.is_meteor_scores=is_meteor_scores
        self.is_damerau_dist_scores=is_damerau_dist_scores
        self.logger = logger


        # Get metrics count for weight calculation
        self.active_metrics = self.get_active_metrics() or {}

        # Asign fixed weight
        fixed_weights={    
                        "edit_dist": config.get("edit_dist_weight"),
                        "damerau_dist":  config.get("damerau_dist_weight"),
                        "cer":  config.get("cer_weight")} if config.get("token_level")=="char_level" else {    
                        "edit_dist": config.get("edit_dist_weight"),
                        "meteor":  config.get("meteor_weight"),
                        "bleu":  config.get("bleu_weight"),
                        "rougeL":  config.get("rougeL_weight"),
                        "rouge1":  config.get("rouge1_weight"),
                        "wer_weight":  config.get("wer_weight")}

        self.weights = self.compute_weights(self.active_metrics, fixed_weights=fixed_weights)

        # Initialize the metrics as TensorFlow variables (empty for now)
        self.bleu_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.rouge1_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.rougeL_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.wer_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.cer_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.edit_dist_scores = tf.Variable(initial_value=[], shape=(None,), dtype=tf.float32, trainable=False)
        self.combined_scores = tf.Variable(initial_value=[], shape=(None,), dtype=tf.float32, trainable=False)

        self.meteor_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.damerau_dist_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        
        # Crate tokenizer
        self.tokenizer = TokenizerFactory.create_tokenizer(config)

        #
        self.score_calculator = ScoreCalculator(self.tokenizer, logger=self.logger)

    def from_config(cls, config):
        return cls(config=config)
    
    def get_active_metrics(self):
        active_metrics = {
            'edit_dist': False,
            'meteor': False,
            'bleu': False,
            'rouge1': False,
            'rougeL': False,
            'wer': False,
            'cer': False,
            'damerau_dist': False
        }

        if self.is_edit_dist_scores:
            active_metrics['edit_dist'] = True
            
        if self.token_level == "word_level":
            if self.is_bleu_scores:
                active_metrics['bleu'] = True
            if self.is_rouge1_scores:
                active_metrics['rouge1'] = True
            if self.is_rougeL_scores:
                active_metrics['rougeL'] = True
            if self.is_wer_scores:
                active_metrics['wer'] = True
            if self.is_meteor_scores:
                active_metrics['meteor'] = True
        elif self.token_level == "char_level":
            if self.is_cer_scores:
                active_metrics['cer'] = True
            if self.is_damerau_dist_scores:
                active_metrics['damerau_dist'] = True

        return active_metrics
    
    def compute_metrics(self, pred_text, tgt_text):
        """
        Compute scores using TensorFlow operations and Python functions via `tf.py_function`.
        This function returns both the BLEU score and the predicted/target texts as TensorFlow tensors.

        Char-Level Metrics:
            These metrics operate on or are affected by individual characters rather than entire words.
            - CER (Character Error Rate)
            - Damerau-Levenshtein Distance Score
        Word-Level Metrics:
            These metrics operate on or are affected by word-level granularity.
            - BLEU Score
            - ROUGE-L Score
            - ROUGE-1 Score
            - WER (Word Error Rate)
            - MET (METEOR Score)
        """
    
        # Use `tf.py_function` to wrap the numpy-based function
        bleu_score, rougeL_score, rouge1_score, wer_score, cer_score, met_score, damerau_dist_score = tf.py_function(
            func=self.score_calculator.np_compute,  # The Python function to call
            inp=[pred_text, tgt_text],  # Inputs to the function
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)  # TensorFlow-compatible output types
        )

        return bleu_score, rougeL_score, rouge1_score, wer_score, cer_score, met_score, damerau_dist_score

    def update_state(self, fwd_preds, fwd_target, edit_dist):
        """
        Updates the state with the metrics for a new batch of predicted and target texts.
        """
        # Decode token to Tensor string
        fwd_predicted_texts, fwd_target_texts = self.tokenizer.tensor_decode(fwd_preds, fwd_target)
        
        # bwd_predicted_texts, bwd_target_texts = self.tokenizer.tensor_decode(preds_bwd, reversed_target)

        # # Compute scores using text-based comparison
        # fwd_bleu_score, fwd_rougeL_score, fwd_rouge1_score, fwd_wer_score, fwd_cer_score, fwd_met_score, fwd_damerau_dist_score  = self.compute_metrics(fwd_predicted_texts, fwd_target_texts)
        # bwd_bleu_score, bwd_rougeL_score, bwd_rouge1_score, bwd_wer_score, bwd_cer_score, bwd_met_score, bwd_damerau_dist_score  = self.compute_metrics(bwd_predicted_texts, bwd_target_texts)

        # bleu_score, rougeL_score, rouge1_score, wer_score, cer_score, met_score, damerau_dist_score = (fwd_bleu_score+bwd_bleu_score)/2, \
        #     (fwd_rougeL_score+bwd_rougeL_score)/2, (fwd_rouge1_score+bwd_rouge1_score)/2, (fwd_wer_score+bwd_wer_score)/2, (fwd_cer_score+bwd_cer_score)/2, \
        #         (fwd_met_score+bwd_met_score)/2, (fwd_damerau_dist_score+bwd_damerau_dist_score)/2

        bleu_score, rougeL_score, rouge1_score, wer_score, cer_score, met_score, damerau_dist_score =  self.compute_metrics(fwd_predicted_texts, fwd_target_texts)
        
        # The correctness of prediction is 100.00 - distance
        # tf.print("damerau_dist_score",  damerau_dist_score)
       
        scale_factor = 100.0
        combined_score = 0.0

        # Update TensorFlow variables by concatenating new scores to the existing ones
        if self.is_edit_dist_scores:
            self.edit_dist_scores.assign([edit_dist])
            avg_edit_dist = tf.reduce_mean(self.edit_dist_scores)
            scaled_avg_edit_dist = tf.clip_by_value(tf.maximum(scale_factor - avg_edit_dist, 0.0), 0.0, 100.0)
            combined_score += self.weights.get("edit_dist", 0.4) * scaled_avg_edit_dist 
            # tf.print("scaled_avg_edit_dist",  scaled_avg_edit_dist)
        if self.token_level == "char_level":
            if self.is_cer_scores:
                self.cer_scores.assign(tf.concat([self.cer_scores, tf.reshape(cer_score, [-1])], axis=0))
                avg_cer = tf.reduce_mean(self.cer_scores)
                scaled_avg_cer = tf.clip_by_value(tf.maximum(scale_factor - avg_cer, 0.0), 0.0, 100.0)
                combined_score -=  self.weights.get("cer", 0.1) * scaled_avg_cer 
                # tf.print("avg_cer",  avg_cer)
                # tf.print("scaled_avg_cer",  scaled_avg_cer)
                # tf.print("combined_score",  combined_score)
            if self.is_damerau_dist_scores:
                self.damerau_dist_scores.assign(tf.concat([self.damerau_dist_scores, tf.reshape(damerau_dist_score, [-1])], axis=0))
                avg_damerau_dist = tf.reduce_mean(self.damerau_dist_scores)
                scaled_avg_damerau_dist = tf.clip_by_value(tf.maximum(scale_factor - avg_damerau_dist, 0.0), 0.0, 100.0) # Correctness
                # tf.print("scaled_avg_damerau_dist",  scaled_avg_damerau_dist)

                combined_score += self.weights.get("damerau_dist", 0.5) * scaled_avg_damerau_dist 
                # tf.print("combined_score",  combined_score)
        elif self.token_level == "word_level":
            if self.is_bleu_scores:
                self.bleu_scores.assign(tf.concat([self.bleu_scores, tf.reshape(bleu_score, [-1])], axis=0))
                avg_bleu = tf.reduce_mean(self.bleu_scores)
                scaled_avg_bleu = tf.clip_by_value(tf.maximum(scale_factor - avg_bleu, 0.0), 0.0, 100.0)
                combined_score +=  self.weights.get("bleu", 0.1) * scaled_avg_bleu
            if self.is_rougeL_scores:
                self.rougeL_scores.assign(tf.concat([self.rougeL_scores, tf.reshape(rougeL_score, [-1])], axis=0)) 
                avg_rougeL = tf.reduce_mean(self.rougeL_scores)
                scaled_avg_rougeL = tf.clip_by_value(tf.maximum(scale_factor - avg_rougeL, 0.0), 0.0, 100.0)
                combined_score += self.weights.get("rougeL", 0.1) * scaled_avg_rougeL
            if self.is_rouge1_scores:
                self.rouge1_scores.assign(tf.concat([self.rouge1_scores, tf.reshape(rouge1_score, [-1])], axis=0))
                avg_rouge1 = tf.reduce_mean(self.rouge1_scores)
                scaled_avg_rouge1 = tf.clip_by_value(tf.maximum(scale_factor - avg_rouge1, 0.0), 0.0, 100.0)
                combined_score += self.weights.get("rouge1", 0.1) * scaled_avg_rouge1
            if self.is_wer_scores:
                self.wer_scores.assign(tf.concat([self.wer_scores, tf.reshape(wer_score, [-1])], axis=0))
                avg_wer = tf.reduce_mean(self.wer_scores)
                scaled_avg_wer = tf.clip_by_value(tf.maximum(scale_factor - avg_wer, 0.0), 0.0, 100.0)
                combined_score -= self.weights.get("wer", 0.1) * scaled_avg_wer
            
            if self.is_meteor_scores:
                self.meteor_scores.assign(tf.concat([self.meteor_scores, tf.reshape(met_score, [-1])], axis=0))
                avg_meteor = tf.reduce_mean(self.meteor_scores)
                scaled_avg_meteor = tf.clip_by_value(tf.maximum(scale_factor - avg_meteor, 0.0), 0.0, 100.0)
                combined_score += self.weights.get("meteor", 0.2) * scaled_avg_meteor
   
        # Ensure combined score is not less than zero and its within 0 and 100
        combined_score = tf.clip_by_value(tf.maximum(combined_score, 0.0), 0.0, 100.0)
        self.combined_scores.assign([combined_score])
        
    
    def compute_weights(
        self,
        active_metrics, 
        fixed_weights=None, 
        base_weight=1.0, 
        normalize=True
        ):
        """
        Compute weights for active metrics dynamically.

        Args:
            active_metrics (dict): A dictionary where keys are metric names (str) and 
                                values are booleans indicating whether the metric is active.
                                Example: {'edit_dist': True, 'meteor': False, 'bleu': True}
            fixed_weights (dict): Optional dictionary of fixed weights for certain metrics.
                                Example: {'damerau_dist': 0.6, 'error': 0.1}
            base_weight (float): The default weight for each active metric (if not specified in fixed_weights).
            normalize (bool): Whether to normalize weights so they sum to 1.0.

        Returns:
            dict: A dictionary mapping metric names to their computed weights.
        """
        # Filter active metrics
        active_metric_names = [metric for metric, is_active in active_metrics.items() if is_active]

        # Initialize weights
        weights = {metric: base_weight for metric in active_metric_names}

        # Apply fixed weights
        if fixed_weights:
            for metric, weight in fixed_weights.items():
                if metric in weights:
                    weights[metric] = weight

        # Compute remaining weight to distribute among other metrics
        total_fixed_weight = sum(fixed_weights.values()) if fixed_weights else 0.0
        remaining_weight = max(0.0, 1.0 - total_fixed_weight) if normalize else base_weight * len(weights)

        # Distribute remaining weight proportionally among metrics without fixed weights
        unfixed_metrics = [m for m in weights if m not in fixed_weights]
        if unfixed_metrics:
            shared_weight = remaining_weight / len(unfixed_metrics)
            for metric in unfixed_metrics:
                weights[metric] = shared_weight

        # Normalize weights to sum to 1.0 if required
        if normalize:
            self.logger.debug("weights : ", weights)
            total_weight = sum(weights.values())
            weights = {metric: weight / total_weight for metric, weight in weights.items()}

        return weights


    def result(self):
        """
        Computes the average of all accumulated metrics and returns the combined score.
        """
        return tf.reduce_mean(self.combined_scores)

    def reset_states(self):
        """
        Resets the state of the metrics to empty lists for the next evaluation phase.
        """
        self.bleu_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.rouge1_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.rougeL_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.wer_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.cer_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.edit_dist_scores = tf.Variable(initial_value=[], shape=(None,), dtype=tf.float32, trainable=False)
        self.meteor_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.damerau_dist_scores = tf.Variable([], shape=(None,), dtype=tf.float32, trainable=False)
        self.combined_scores = tf.Variable(initial_value=[], shape=(None,), dtype=tf.float32, trainable=False)