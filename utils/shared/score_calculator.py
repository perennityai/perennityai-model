import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from jiwer import wer, cer
from difflib import SequenceMatcher
from rouge_score import rouge_scorer
from pyrsdameraulevenshtein import similarity_str, normalized_distance_str
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Define the smoothing function globally or within your model
smoothing_function = SmoothingFunction().method1
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class ScoreCalculator:
    def __init__(self, tokenizer, logger=None):
        self.tokenizer = tokenizer
        self.logger = logger


    def np_compute(self, pred_text_np, tgt_text_np, similarity=False):
        """
        Compute BLEU score using numpy-based inputs, called from within a TensorFlow function.
        """
        pred_text, tgt_text = ScoreCalculator.convert_tensor_to_python_string(pred_text_np, tgt_text_np)

        # Compute BLEU score using word-based comparison
        if len(tgt_text) > 0 and len(pred_text) > 0:
            bleu_score = ScoreCalculator.bleu_score(tgt_text, pred_text)
        else:
            bleu_score = 0.0

        # Compute metrics score using text-based comparison
        rouge1_score, rougeL_score = ScoreCalculator.rouge1L_score(tgt_text, pred_text)
        wer_score = ScoreCalculator.wer_score(tgt_text, pred_text)
        cer_score = ScoreCalculator.cer_score(tgt_text, pred_text)
        met_score = ScoreCalculator.meteor_score(tgt_text, pred_text)

        #Calculate DamerauLevenshtein.distance and normalize the Score
        if not similarity:
            norm_damerau_dist_score = ScoreCalculator.distance_score(tgt_text, 
                                                                    pred_text
                                                                    )
        else:
            norm_damerau_dist_score = ScoreCalculator.similarity_score(tgt_text, 
                                                                    pred_text
                                                                    )

        bleu_score = np.nan_to_num(bleu_score, nan=0.0)
        rouge1_score = np.nan_to_num(rouge1_score, nan=0.0)
        rougeL_score = np.nan_to_num(rougeL_score, nan=0.0)
        wer_score = np.nan_to_num(wer_score, nan=0.0)
        cer_score = np.nan_to_num(cer_score, nan=0.0)
        met_score = np.nan_to_num(met_score, nan=0.0)
        norm_damerau_dist_score = np.nan_to_num(norm_damerau_dist_score, nan=0.0)

        self.logger.debug("damerau_dist_score : ", norm_damerau_dist_score)
        self.logger.debug("tgt_text : ", tgt_text[:64].strip())
        self.logger.debug("pred_text : ", pred_text[:64].strip())
        self.logger.debug("")
        

        return bleu_score, rougeL_score, rouge1_score, wer_score, cer_score, met_score, norm_damerau_dist_score

 
    @staticmethod
    def convert_tensor_to_python_string(pred_text_np, tgt_text_np):

        # Convert EagerTensors (passed as NumPy arrays) to Python strings
        if isinstance(pred_text_np, np.ndarray):
            if isinstance(pred_text_np[0], bytes):
                # Convert bytes to strings
                pred_text = ' '.join(np.array([byte_str.decode('utf-8') for byte_str in pred_text_np]))
            else:
                pred_text = np.array2string(pred_text_np.numpy())
        elif isinstance(pred_text_np, tf.Tensor):
            if isinstance(pred_text_np, EagerTensor):
                
               # Convert the EagerTensor to a NumPy array or Python object
                pred_text_np = pred_text_np.numpy()

                # Check if the result is a string
                if isinstance(pred_text_np, str):
                    pred_text = pred_text_np  # Use the string directly
                elif isinstance(pred_text_np, bytes):
                    # Convert bytes to strings
                    pred_text = ' '.join(np.array([str(byte_str) for byte_str in pred_text_np]))
                else:
                    # If it's a NumPy array, convert it to a string
                    pred_text = np.array2string(pred_text_np)
            else:
                try:
                    pred_text = np.array2string(pred_text_np.numpy().decode('utf-8'))
                except:
                    pred_text = np.array2string(pred_text_np.numpy())
        elif isinstance(pred_text_np, EagerTensor):
            pred_text = np.array2string(pred_text_np.numpy())
        elif isinstance(pred_text_np, (np.ndarray, tf.Tensor)):
            pred_text = " ".join(map(str, pred_text_np.numpy().flatten()))  # Convert tensor to NumPy and process
        else:
            pred_text = str(pred_text_np)

        if isinstance(tgt_text_np, np.ndarray):
            if isinstance(tgt_text_np[0], bytes):
                # Convert bytes to strings
                tgt_text = ' '.join(np.array([byte_str.decode('utf-8') for byte_str in tgt_text_np]))
            else:
                tgt_text = tgt_text_np.numpy()
        elif isinstance(tgt_text_np, EagerTensor):
            # Convert the EagerTensor to a NumPy array or Python object
            tgt_text_np = tgt_text_np.numpy()

            # Check if the result is a string
            if isinstance(tgt_text_np, str):
                tgt_text = tgt_text_np  # Use the string directly
            elif isinstance(tgt_text_np, bytes):
                # Convert bytes to strings
                tgt_text = ' '.join(np.array([str(byte_str) for byte_str in tgt_text_np]))
            else:
                # If it's a NumPy array, convert it to a string
                tgt_text = np.array2string(tgt_text_np)
        elif isinstance(tgt_text_np, tf.Tensor):
            if isinstance(tgt_text_np, EagerTensor):
                tgt_text = np.array2string(tgt_text_np.numpy()) 
            else:
                try:
                    tgt_text = np.array2string(tgt_text_np.numpy().decode('utf-8')) 
                except:
                    tgt_text = np.array2string(tgt_text_np.numpy()) 

        elif isinstance(tgt_text_np, (np.ndarray, tf.Tensor)):
            tgt_text = " ".join(map(str, tgt_text_np.numpy().flatten()))
        elif isinstance(tgt_text_np, EagerTensor):
            tgt_text = np.array2string(tgt_text_np.numpy()) 
        else:
            tgt_text = str(tgt_text_np)

        if not isinstance(pred_text, str) or not isinstance(tgt_text, str):
            raise TypeError(f"Should be string pred_text : {type(pred_text)}, tgt_text : {type(tgt_text)}")
        
        return pred_text, tgt_text
    
    @staticmethod
    def calculate_edit_distance(preds, target, argmax=True):
        # Compute edit distance for evaluation
        if argmax:
            preds = tf.argmax(preds, axis=2)
                            
        normalized_distance = tf.edit_distance(
            tf.sparse.from_dense(target),
            tf.sparse.from_dense(tf.cast(preds, tf.int32)),
            normalize=True
        )
        return normalized_distance
    
    @staticmethod
    def rouge1L_score(tgt_text, pred_text):
        """
        Compute the ROUGE-1 and ROUGE-L scores between the target text and the predicted text.

        Args:
            tgt_text (str): The ground truth target text.
            pred_text (str): The predicted text.

        Returns:
            tuple: A tuple containing:
                - rouge1_score (float): F-measure of ROUGE-1 (unigram overlap).
                - rougeL_score (float): F-measure of ROUGE-L (longest common subsequence).

        Example:
            tgt_text = "The cat is on the mat"
            pred_text = "The cat sat on the mat"
            rouge1L_score(tgt_text, pred_text) -> (0.8, 0.75)
        """
        if not isinstance(tgt_text, str) or not isinstance(pred_text, str):
            pred_text, tgt_text = ScoreCalculator.convert_tensor_to_python_string(tgt_text, pred_text)
        rouge_scores = rouge_scorer_instance.score(tgt_text, pred_text)
        rouge1_score = rouge_scores['rouge1'].fmeasure
        rougeL_score = rouge_scores['rougeL'].fmeasure
        return rouge1_score, rougeL_score
        
    @staticmethod
    def bleu_score(tgt_text, pred_text):
        """
        Compute the BLEU (Bilingual Evaluation Understudy) score between the target text and predicted text.

        Args:
            tgt_text (str): The ground truth target text.
            pred_text (str): The predicted text.

        Returns:
            float: BLEU score indicating the overlap between target and predicted texts.

        Example:
            tgt_text = "The cat is on the mat"
            pred_text = "The cat sat on the mat"
            blue_score(tgt_text, pred_text) -> 0.75
        """
        if not isinstance(tgt_text, str) or not isinstance(pred_text, str):
            pred_text, tgt_text = ScoreCalculator.convert_tensor_to_python_string(tgt_text, pred_text)
        return sentence_bleu([tgt_text.split()], pred_text.split(), smoothing_function=SmoothingFunction().method1)

    @staticmethod
    def meteor_score(tgt_text, pred_text):
        """
        Compute the METEOR (Metric for Evaluation of Translation with Explicit ORdering) score.

        Args:
            tgt_text (str): The ground truth target text.
            pred_text (str): The predicted text.

        Returns:
            float: METEOR score reflecting how well the predicted text matches the target text.

        Example:
            tgt_text = "The cat is on the mat"
            pred_text = "The cat sat on the mat"
            meteor_score(tgt_text, pred_text) -> 0.85
        """
        if not isinstance(tgt_text, str) or not isinstance(pred_text, str):
            pred_text, tgt_text = ScoreCalculator.convert_tensor_to_python_string(tgt_text, pred_text)
        return meteor_score([tgt_text.split()], pred_text.split())
    
    @staticmethod
    def cer_score(tgt_text, pred_text):
        """
        Calculate the Character Error Rate (CER) between the target and predicted text.
        CER measures the minimum number of character edits (insertions, deletions, substitutions) 
        required to transform one string into another.

        Args:
            target_text (str): The ground truth target text.
            predicted_text (str): The predicted text.

        Returns:
            float: The CER value ranging from 0 (perfect match) to 1 (completely different).

        Example:
            target_text = "hello world"
            predicted_text = "helo world"
            cer_score(target_text, predicted_text) -> 0.1
        """
        if not isinstance(tgt_text, str) or not isinstance(pred_text, str):
            pred_text, tgt_text = ScoreCalculator.convert_tensor_to_python_string(tgt_text, pred_text)
        target_len = len(tgt_text)
        if target_len == 0:
            return 1.0
        return 1.0 - SequenceMatcher(None, tgt_text, pred_text).ratio()

    @staticmethod
    def wer_score(tgt_text, pred_text):
        """
        Compute the Word Error Rate (WER) between the target and predicted text.
        WER measures the number of word-level edits (insertions, deletions, substitutions) needed
        to transform one text into another, normalized by the total number of words in the target text.

        Args:
            tgt_text (str): The ground truth target text.
            pred_text (str): The predicted text.

        Returns:
            float: The WER value ranging from 0 (perfect match) to 1 (completely different).

        Example:
            tgt_text = "The cat is on the mat"
            pred_text = "The cat sat on the mat"
            wer_score(tgt_text, pred_text) -> 0.2
        """
        if not isinstance(tgt_text, str) or not isinstance(pred_text, str):
            pred_text, tgt_text = ScoreCalculator.convert_tensor_to_python_string(tgt_text, pred_text)
        return wer(tgt_text, pred_text)

    @staticmethod  
    def similarity_score(pred_text, tgt_text, scale=True):
        """
        Compute the similarity score between the predicted text and the target text.

        The function calculates the similarity based on the Damerau-Levenshtein distance 
        or any other provided distance. The similarity score is returned as a percentage 
        between 0.0 and 100.0, where 100.0 represents identical strings.

        Parameters:
        pred_text (str): The predicted text to be compared.
        tgt_text (str): The target or reference text to compare against.
        distance (float, optional): A precomputed distance score. If not provided, the 
                                    Damerau-Levenshtein distance is used.

        Returns:
        float: The similarity score between the predicted text and the target text, 
            expressed as a percentage between 0.0 and 100.0.

        If both strings are empty, the function will return a similarity of 100.0,
        treating empty strings as identical.

        Example:
        >>> similarity = compute_similarity("hello world", "helo world")
        >>> print(similarity)
        95.0
        
        Notes:
        - The distance is normalized by the length of the longer of the two strings.
        - The similarity score is inversely proportional to the normalized distance.
        - A lower distance results in a higher similarity score.
        """
        if not isinstance(tgt_text, str) or not isinstance(pred_text, str):
            pred_text, tgt_text = ScoreCalculator.convert_tensor_to_python_string(tgt_text, pred_text)

        # Calculate the maximum length between the predicted and target texts
        max_len = max(len(pred_text), len(tgt_text))

        # Edge case: If both texts are empty, treat them as identical
        if max_len == 0:
            return 0.0  # Treat empty strings as identical
        
        # If no distance is provided, calculate Damerau-Levenshtein distance
        sim_score = similarity_str(list(tgt_text), list(pred_text))

        # Return similarity as a percentage (0.0 to 100.0)
        if scale:
            # Return distance as a percentage (0.0 to 100.0)
            return sim_score * 100.0
        else:
            return sim_score

    @staticmethod  
    def distance_score(pred_text, tgt_text, scale=True):
        """
        Compute the similarity score between the predicted text and the target text.

        The function calculates the similarity based on the Damerau-Levenshtein distance 
        or any other provided distance. The similarity score is returned as a percentage 
        between 0.0 and 100.0, where 100.0 represents identical strings.

        Parameters:
        pred_text (str): The predicted text to be compared.
        tgt_text (str): The target or reference text to compare against.
        distance (float, optional): A precomputed distance score. If not provided, the 
                                    Damerau-Levenshtein distance is used.

        Returns:
        float: The similarity score between the predicted text and the target text, 
            expressed as a percentage between 0.0 and 100.0.

        If both strings are empty, the function will return a similarity of 100.0,
        treating empty strings as identical.

        Example:
        >>> similarity = compute_similarity("hello world", "helo world")
        >>> print(similarity)
        95.0
        
        Notes:
        - The distance is normalized by the length of the longer of the two strings.
        - The similarity score is inversely proportional to the normalized distance.
        - A lower distance results in a higher similarity score.
        """
        if not isinstance(tgt_text, str) or not isinstance(pred_text, str):
            pred_text, tgt_text = ScoreCalculator.convert_tensor_to_python_string(tgt_text, pred_text)

        # Calculate the maximum length between the predicted and target texts
        max_len = max(len(pred_text), len(tgt_text))

        # Edge case: If both texts are empty, treat them as identical
        if max_len == 0:
            return 0.0  # Treat empty strings as identical

        # calculate Damerau-Levenshtein distance
        norm_distance = normalized_distance_str(list(tgt_text), list(pred_text))

        if scale:
            # Return distance as a percentage (0.0 to 100.0)
            return norm_distance * 100.0
        else:
            return norm_distance
    
    @staticmethod
    def pad_text(pred_text, pad_token, target_maxlen):
        """
        Pads pred_text to target_maxlen using pad_token.

        Args:
            pred_text (str or bytes): The text to pad.
            pad_token (str or bytes): The padding token.
            target_maxlen (int): The desired length after padding.

        Returns:
            str or bytes: The padded text.
        """
        # Determine the type of pred_text
        if isinstance(pred_text, bytes):
            # Option 1: Convert to str
            pred_text_str = pred_text.decode('utf-8')  # Adjust encoding as needed
            pad_char = pad_token if isinstance(pad_token, str) else pad_token.decode('utf-8')
            
            # Validate pad_char length
            if len(pad_char) != 1:
                raise ValueError("pad_token must be a single character string.")
            
            # Apply ljust
            padded_text_str = pred_text_str.ljust(target_maxlen, pad_char)
            return padded_text_str.encode('utf-8')  # Return as bytes if necessary
        
        elif isinstance(pred_text, str):
            # Ensure pad_token is str
            pad_char = pad_token if isinstance(pad_token, str) else pad_token.decode('utf-8')
            
            # Validate pad_char length
            if len(pad_char) != 1:
                raise ValueError("pad_token must be a single character string.")
            
            # Apply ljust
            padded_text = pred_text.ljust(target_maxlen, pad_char)
            return padded_text
        
        else:
            raise TypeError("pred_text must be either a str or bytes object.")

    