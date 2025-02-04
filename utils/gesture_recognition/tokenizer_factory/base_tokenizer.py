import os
import json
import tensorflow as tf
from utils.common import JSONHandler

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

class BaseTokenizer:
    def __init__(self, token_file=None, default_tokens=[], num_classes=92, target_maxlen=64):
        """
        Base tokenizer class for token mapping and lookup table creation.
        
        Args:
            token_file (str): Path to JSON file for loading existing token map.
            default_tokens (list): List of default special tokens.
        """
        # Load token map from a JSON file or initialize an empty dictionary        
        self.token_map = JSONHandler.read_json(token_file) or {}

        # Text max
        self.target_maxlen = target_maxlen

        if len(default_tokens) < 4 and len(default_tokens) > 5:
            raise ValueError("Incorrect special token list")

        # Special tokens, override in subclasses as needed
        self.pad_token = default_tokens[0]
        self.bos_token =  default_tokens[1]
        self.eos_token =  default_tokens[2]
        self.unk_token =  default_tokens[3]

        if len(default_tokens) == 5:
            # Assign space token
            self.space_token = default_tokens[4]
             # Initialize special tokens based on input or default list
            self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token, self.space_token]
        else:
            # Initialize special tokens based on input or default list
            self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        # Assign unique indices to tokens if not already in the token map
        self.next_token_idx = max(self.token_map.values(), default=0) + 1
        for token in self.special_tokens:
            if token not in self.token_map:
                self.token_map[token] = self.next_token_idx
                self.next_token_idx += 1


        # Extract indices
        self.pad_token_idx = self.token_map[self.pad_token]
        self.bos_token_idx = self.token_map[self.bos_token]
        self.eos_token_idx = self.token_map[self.eos_token]
        self.unk_token_idx = self.token_map[self.unk_token]

        # Validate parameters in config
        if num_classes != self.next_token_idx:
            raise ValueError(f"Adjust parameters in config.json to num_classes : {self.next_token_idx} instead of {num_classes}, bos_token_idx: {self.bos_token_idx}, eos_token_idx : {self.eos_token_idx}")

        self.num_classes = self.next_token_idx
        
        # Create reverse token map
        self.reverse_token_map = {idx: token for token, idx in self.token_map.items()}
        # Create lookup tables
        self.table = self.create_lookup_table(self.token_map, default_value=self.token_map[self.unk_token])
        # Create reverse loopup table
        self.reverse_table = self.create_reverse_lookup_table(self.reverse_token_map, default_value=self.unk_token)

    @classmethod
    def from_config(cls, config):
        # Allow flexibility in config overrides for subclasses
        if config.get("token_level") == "char_level":
            default_tokens=[config.get("char_pad_token"),
                            config.get("char_bos_token"),
                            config.get("char_eos_token"),
                            config.get("char_unk_token"),
                            config.get("char_space_token")]
        else:
            default_tokens=[config.get("gpt_pad_token"),
                            config.get("gpt_bos_token"),
                            config.get("gpt_eos_token"),
                            config.get("gpt_unk_token"),
                            config.get("gpt_space_token")]

        return cls(**{
            "token_file": config.get("token_file", ''),
            "default_tokens": config.get("default_tokens", default_tokens),
            "num_classes": config["num_classes"],
            "target_maxlen": config["target_maxlen"]
        })


    def create_lookup_table(self, mapping, default_value):
        """Creates a static hash lookup table from a dictionary."""
        keys = tf.constant(list(mapping.keys()))
        values = tf.constant(list(mapping.values()))
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=default_value, name="lookup_table"
        )
    
    def create_reverse_lookup_table(self, rev_character_map, default_value):
        # Convert the dictionary to keys and values tensors
        keys = tf.constant(list(rev_character_map.keys()), dtype=tf.int64)
        values = tf.constant(list(rev_character_map.values()), dtype=tf.string)
        # Create a StaticHashTable for character lookup
        return tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=default_value,
            name="reverse_lookup_table"
        )
    
    def reverse_unbatched_token(self, sequence):
        """
        Reverse an unbatched sequence of tokens but keep start and end tokens in place.

        Args:
            sequence: Tensor of a single sequence of tokens (1, seq_length).
            start_token_idx: Index of the start token.
            end_token_idx: Index of the end token.

        Returns:
            Tensor with reversed sequence, excluding start and end tokens.
        """
    
        # Remove the batch dimension (convert (1, seq_length) to (seq_length,))
        sequence = tf.squeeze(sequence, axis=0)

        # Ensure the sequence has at least 3 tokens (start, middle, end)
        if tf.size(sequence) <= 2:
            return sequence

        # Identify start and end tokens
        start_token = sequence[:1]  # First token
        end_token = sequence[-1:]   # Last token

        # Reverse the middle part of the sequence
        middle_tokens = sequence[1:-1]
        reversed_middle = tf.reverse(middle_tokens, axis=[0])

        # Concatenate start, reversed middle, and end tokens
        reversed_sequence = tf.concat([start_token, reversed_middle, end_token], axis=0)

        # Add back the batch dimension (convert (seq_length,) to (1, seq_length))
        return tf.expand_dims(reversed_sequence, axis=0)

    
    def reverse_batched_2D_token(self, targets):
        """
        Reverse sequences in targets but keep start and end tokens in place.

        Args:
            targets: Tensor of target sequences (batch_size, seq_length).
        Returns:
            Tensor with reversed sequences, excluding start and end tokens.
        """
        # Identify the start and end tokens
        start_tokens = targets[:, :1]  # First token in each sequence
        end_tokens = targets[:, -1:]  # Last token in each sequence

        # Reverse the middle portion of the sequence
        reversed_middle = tf.reverse(targets[:, 1:-1], axis=[1])

        # Concatenate start, reversed middle, and end tokens
        reversed_targets = tf.concat([start_tokens, reversed_middle, end_tokens], axis=1)
        return reversed_targets
    
    
    def reverse_batched_3D_token(self, targets):
        """
        Reverse sequences in targets but keep start and end tokens in place.
        
        Args:
            targets: Tensor of target sequences with shape (batch_size, seq_len, num_features).
        
        Returns:
            Tensor with reversed sequences, excluding start and end tokens, maintaining input shape.
        """
        # Identify the start and end tokens along the sequence dimension
        start_tokens = targets[:, :1, :]  # First token (batch_size, 1, num_features)
        end_tokens = targets[:, -1:, :]  # Last token (batch_size, 1, num_features)

        # Reverse the middle portion of the sequence along the sequence dimension
        middle_tokens = targets[:, 1:-1, :]  # Middle tokens (batch_size, seq_len-2, num_features)
        reversed_middle = tf.reverse(middle_tokens, axis=[1])  # Reverse along seq_len (axis=1)

        # Concatenate start, reversed middle, and end tokens along the sequence dimension
        reversed_targets = tf.concat([start_tokens, reversed_middle, end_tokens], axis=1)

        return reversed_targets
    
    def add_space_token(self, text, char_level=False):
        # Add <space> or _ explicitly during tokenization.
        if char_level:
            replaced_text = tf.strings.regex_replace(text, " ", self.space_token)
        else:
            replaced_text = tf.strings.regex_replace(text, " ", " " + self.space_token + " ")
        return replaced_text
    
    def remove_space_token(self, text, char_level=True):
        if char_level:
            replaced_text = tf.strings.regex_replace(text, self.space_token, " ")
        else:
            # Replace all occurrences of space_token with an empty string
            text = tf.strings.regex_replace(text, self.space_token, "")

        # Replace multiple consecutive spaces with a single space
        text = tf.strings.regex_replace(text, r"\s+", " ")

        # Optionally, strip leading and trailing spaces
        text = tf.strings.strip(text)
        return replaced_text
    
    def clean_text(self, text):
        # Remove the specified space_token
        text = text.replace(f"{self.bos_token}", "").replace(f"{self.eos_token}", "").replace(f"{self.pad_token}", "").replace(f"{self.space_token}", " ")
        # Strip leading and trailing spaces
        return text.strip()
    
    def encode(self, tokens):
        """Encodes a list of tokens into indices."""
        return [self.token_map.get(token, self.token_map[self.unk_token]) for token in tokens]

    def decode(self, indices):
        """Decodes a list of indices back to tokens."""
        return [self.reverse_token_map.get(index, self.unk_token) for index in indices]
    
    def clean_prediction(self, preds):
        # Convert logits to class/character indices (take argmax across the last dimension)
        # preds_indices = tf.argmax(preds, axis=-1)  # Shape now [None, None]

        # Create a mask where the end token is found, so everything after it is set to padding
        def truncate_after_eos_token(sequence):
            # Find positions where eos_token_idx exists
            eos_token_positions = tf.where(tf.equal(sequence, self.eos_token_idx))

            # If eos_token_idx is found, we keep everything up to its first occurrence
            if tf.size(eos_token_positions) > 0:
                # Take the first occurrence of the eos_token_idx
                end_position = eos_token_positions[0][0]  # first index of eos_token_idx
                sequence = sequence[:end_position + 1]  # Keep up to and including the eos_token_idx
            else:
                # If eos_token_idx isn't found, we keep the whole sequence
                sequence = sequence

            # # Pad sequence to max_length
            # sequence = tf.pad(sequence, 
            #                          paddings=[[0, self.target_maxlen - tf.size(sequence)]], 
            #                          constant_values=self.pad_token_idx)

            return sequence
        
        # Apply the truncate function for each prediction in the batch
        cleaned_preds = tf.map_fn(truncate_after_eos_token, preds, dtype=tf.int64)

        return cleaned_preds
    
    def tensor_clean(self, pred_texts, target_texts, clean_padding=True, clean_bos_token=True, clean_eos_token=True):
        """
        Decodes tensor predictions and targets into text using TensorFlow operations.
        """   

        if clean_padding:
            # Create the regular expression dynamically using the pad_token
            pattern = f"{self.pad_token}"
            # Clean predicted and target texts using tf.regex_replace (removes any pattern of 'P')
            clean_target_texts = tf.strings.regex_replace(target_texts, pattern, '')

            # Clean predicted and target texts using tf.regex_replace (removes any pattern of 'P')
            clean_pred_texts = tf.strings.regex_replace(pred_texts, pattern, '')
        else:
            clean_target_texts = target_texts

        if clean_bos_token:
            # Define regex patterns to remove the start tokens
            pattern_start = f"{self.bos_token}"  # Exact match for start token

            # Clean the token list by removing start tokens using regex
            clean_pred_texts = tf.strings.regex_replace(clean_pred_texts, pattern_start, '')
            clean_target_texts  = tf.strings.regex_replace(clean_target_texts, pattern_start, '')
            

        if clean_eos_token:
            # Define regex patterns to remove the end tokens
            pattern_end = f"{self.eos_token}"  # Exact match for end token

            # Clean the token list by removing end tokens using regex
            clean_pred_texts = tf.strings.regex_replace(clean_pred_texts, pattern_end, '')
            clean_target_texts  = tf.strings.regex_replace(clean_target_texts, pattern_end, '')

        # Clean space token
        pattern = f"{self.space_token}"
        clean_pred_texts = tf.strings.regex_replace(clean_pred_texts, pattern_end, ' ')
        clean_target_texts  = tf.strings.regex_replace(clean_target_texts, pattern_end, ' ')     
    
        return clean_pred_texts, clean_target_texts

    def tensor_decode(self, preds, target, remove_special_tokens=True):
        """
        Decodes tensor predictions and targets into text using TensorFlow operations.
        """ 

        # Cast preds and target to int64 (to match the key tensor data type)
        target = tf.cast(target, tf.int64)
        preds = tf.cast(preds, tf.int64)
    
        # Use TensorFlow StaticHashTable to look up character representations
        pred_texts = tf.strings.reduce_join(self.reverse_table.lookup(preds), axis=1)
        target_texts = tf.strings.reduce_join(self.reverse_table.lookup(target), axis=1)

        if remove_special_tokens:
            pred_texts, target_texts =  self.tensor_clean(pred_texts, target_texts, clean_padding=True, clean_bos_token=True, clean_eos_token=True)


        return pred_texts, target_texts
