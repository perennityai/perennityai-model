
import os
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import jax
import jax.numpy as jnp
from jax import random

class DataAugmentation:
    """
    Explanation:
        Inputs : are (2D) flattened landmarks e.g (5, 156) or (5, 1629)
        Random Translation:
            Shifts the x and y coordinates of the landmarks by a random amount in the range [-translation_range, translation_range].
        Random Rotation:
            Rotates the landmarks around their center by a random angle between [-angle_range, angle_range].
            Rotation is done using a 2D rotation matrix.
        Random Scaling:
            Scales the landmarks relative to their center by a factor between 1 - scale_range and 1 + scale_range.
        Random Shearing:
            Applies a shear transformation to the x coordinates relative to the y coordinates.
        Key Assumptions:
            Input Shape: The input landmark array has a shape of (5, 156).
                         The transformations are applied only to the 2D coordinates (x, y), while the z coordinate is left unchanged.

        Summary of Limits:
            Augmentation	Parameter	        Lower Upper 	Typical Range
            Translation  	translation_range	0.0	  0.3	     0.0 to 0.1
            Rotation	    angle_range	        0°	  45°	     0° to 30°
            Scaling	        scale_range	        0.5	  2.0	     0.8 to 1.2
            Shearing	    shear_range	        0.0	  0.5	     0.0 to 0.2


        Here are augmentation methods tailored to gesture data:

            1. Time-Series/Sequential Data Augmentation
            a. Time Warping
                stretch or compress the time dimension while maintaining sequence order.
            Use Case: Mimics gestures performed at different speeds.
                Implementation:
                    def time_warp(data, rate=0.2):
                        scale = 1 + np.random.uniform(-rate, rate)
                        return tf.image.resize(data, [int(data.shape[1] * scale), data.shape[2]])
            b. Time Masking
                Mask random segments in the sequence with zeros or mean values.
            Use Case: Encourages the model to focus on unmasked parts of the sequence.
                Implementation:
          
                def time_mask(data, mask_ratio=0.1):
                    mask_len = int(data.shape[1] * mask_ratio)
                    start = np.random.randint(0, data.shape[1] - mask_len)
                    data[:, start:start + mask_len, :] = 0
                    return data
            c. Sequence Shuffling
                Slightly shuffle the sequence within a window to mimic noisy or imperfect data.
            Use Case: Adds noise for robustness.
                Implementation:
           
                def shuffle_sequence(data, max_shift=3):
                    shift = np.random.randint(-max_shift, max_shift)
                    return tf.roll(data, shift, axis=1)
            2. Spatial Data Augmentation (e.g., Coordinates or Landmarks)
                a. Random Rotation
                    Apply small random rotations to the spatial representation of gestures.
            Use Case: Mimics gestures viewed from slightly different angles.
                Implementation:
            
                def random_rotation(data, max_angle=10):
                    angle = np.deg2rad(np.random.uniform(-max_angle, max_angle))
                    rotation_matrix = tf.constant([[tf.cos(angle), -tf.sin(angle)], 
                                                    [tf.sin(angle), tf.cos(angle)]])
                    return tf.einsum('ijk,kl->ijl', data, rotation_matrix)
            b. Scaling and Translation
                Scale and shift gesture coordinates randomly.
            Use Case: Mimics variations in gesture intensity or position.
                Implementation:
           
                def scale_and_translate(data, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
                    scale = tf.random.uniform([], *scale_range)
                    shift = tf.random.uniform([], *shift_range)
                    return data * scale + shift
            3. Noise Injection
                a. Gaussian Noise
                    Add random Gaussian noise to the sequence data.
            Use Case: Simulates sensor noise or imperfect measurements.
            
                def add_gaussian_noise(data, stddev=0.01):
                    noise = tf.random.normal(shape=data.shape, mean=0.0, stddev=stddev)
                    return data + noise
            b. Dropout
                Randomly set some feature values to zero.
            Use Case: Increases robustness to missing or corrupted data.
            
                def random_dropout(data, drop_rate=0.1):
                    mask = tf.random.uniform(data.shape) > drop_rate
                    return data * tf.cast(mask, data.dtype)
            4. Domain-Specific Augmentation
            a. Gesture Mirroring
                Reflect gestures along the vertical axis to simulate mirrored gestures.
            Use Case: Enhances robustness for both left- and right-handed gestures.
            
                def mirror_gesture(data):
                    return tf.concat([data[:, :, ::-1], data[:, :, :]], axis=-1)
            b. Sequence Reversal
                Reverse the sequence order to add diversity.
            Use Case: Helps with recognizing gestures performed in reverse.
            
                def reverse_sequence(data):
                    return tf.reverse(data, axis=[1])
            5. Combining Augmentations
                Combine augmentations dynamically for each batch to ensure diverse training samples:
                def augment_gesture_data(data):
                    data = time_warp(data)
                    data = random_rotation(data)
                    data = add_gaussian_noise(data)
                    return data
            Integrating Augmentations into TGGMT
            Apply Before Gesture Model Inference: Augment the input batch before passing it to the gesture_transformer:

            gest_inputs_augmented = augment_gesture_data(gest_inputs)
            gest_predictions = self.gesture_transformer(gest_inputs_augmented, training=False)
            Maintain Consistency with Labels: Ensure augmentations do not alter label interpretations (e.g., flipping, rotation).

            Dynamic Augmentation: Apply augmentations conditionally during training to maintain variability:

            python
            Copy
            Edit
            if training:
                gest_inputs = augment_gesture_data(gest_inputs)
            Key Considerations
            Keep Augmentations Realistic: Avoid augmentations that deviate significantly from real-world data.
            Test Performance: Validate that augmentations improve metrics without introducing noise that confuses the model.
            Ensure Efficiency: Use vectorized TensorFlow operations for augmentations to avoid slowing down training.
    """
    def __init__(self, 
                 aug_param, 
                 num_augmentations=0,
                 seed=42,
                 num_features=26, 
                 channels=3, 
                 logger=None):
        self.num_augmentations = num_augmentations
        self.seed = seed
        self.num_features = num_features
        self.channels = channels
        self.total_num_features = num_features * channels
        self.logger = logger
        # Set the seed for reproducibility
        tf.random.set_seed(self.seed)
        # Ensure augmentation parameters matches the number of functions
        num_augmentations_fn = 4
        if len(aug_param) < num_augmentations_fn:
            raise ValueError(f"Parameter dictionary must have at least {num_augmentations_fn} key-value pairs.")
        # load augmentation param
        self.aug_param = aug_param
        self.rand = 0
            

    # Method to set an attribute using __setattr__
    def set_attribute(self, attr_name, value):
        self.__setattr__(attr_name, value)

    # Method to get an attribute using __getattribute__
    def get_attribute(self, attr_name):
        return self.__getattribute__(attr_name)

    @staticmethod
    def shuffle_sequence(data, max_shift=3):
        """
        Slightly shuffle the sequence within a window to mimic noisy or imperfect data.
        
        Args:
            data: A tensor of shape [batch_size, seq_len, features].
            max_shift: Maximum number of positions to shift.

        Returns:
            Tensor with shuffled sequence.
        """
        # Generate a random shift value between [-max_shift, max_shift]
        shift = tf.random.uniform(
            shape=[], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32
        )

        # Roll the tensor along the sequence axis
        return tf.roll(data, shift=shift, axis=1)
    
    @staticmethod
    def time_warp(data, rate=0.2):
        """
        Randomly stretch or compress the time dimension while preserving the batch and feature dimensions.

        Args:
            data: Tensor of shape [batch_size, seq_len, features].
            rate: Float, maximum stretch/compression rate.

        Returns:
            Tensor with the same shape as the input.
        """
        batch_size, seq_len, features = tf.unstack(tf.shape(data))
        
        # Generate a random warp factor within the range [1 - rate, 1 + rate]
        warp_factor = tf.random.uniform(shape=[], minval=1.0 - rate, maxval=1.0 + rate)
        
        # Calculate the new sequence length after warping
        new_seq_len = tf.cast(tf.cast(seq_len, tf.float32) * warp_factor, tf.int32)
        
        # Resize the sequence using interpolation
        # Add a dummy channel dimension for compatibility with tf.image.resize
        data_expanded = tf.expand_dims(data, axis=-1)  # Shape: [batch_size, seq_len, features, 1]
        
        # Resize along the time dimension
        warped_data = tf.image.resize(
            data_expanded,  # Input tensor with dummy channel
            size=[new_seq_len, features],  # Resize to new sequence length and original features
            method='bilinear'  # Bilinear interpolation
        )
        
        # Remove the dummy channel dimension
        warped_data = tf.squeeze(warped_data, axis=-1)  # Shape: [batch_size, new_seq_len, features]
        
        # If the new sequence is longer than the original, truncate it
        if new_seq_len > seq_len:
            warped_data = warped_data[:, :seq_len, :]
        # If the new sequence is shorter, pad it with zeros
        elif new_seq_len < seq_len:
            padding = tf.zeros([batch_size, seq_len - new_seq_len, features], dtype=data.dtype)
            warped_data = tf.concat([warped_data, padding], axis=1)
        
        return warped_data


    @staticmethod
    def time_mask(data, mask_ratio=0.1):
        """
        Mask random segments in the sequence with zeros.

        Args:
            data: Tensor of shape [batch_size, seq_len, features].
            mask_ratio: Float, proportion of the sequence to mask.

        Returns:
            Tensor with masked regions.
        """
        seq_len = tf.cast(tf.shape(data)[1], tf.float32)  # Cast seq_len to float32
        mask_len = tf.cast(seq_len * mask_ratio, tf.int32)  # Compute mask length as int32

        start = tf.random.uniform([], 0, tf.shape(data)[1] - mask_len, dtype=tf.int32)  # Random start position
        mask = tf.concat([
            tf.ones([start], dtype=data.dtype),
            tf.zeros([mask_len], dtype=data.dtype),
            tf.ones([tf.shape(data)[1] - start - mask_len], dtype=data.dtype)
        ], axis=0)
        mask = tf.expand_dims(mask, axis=0)  # Add batch dimension
        mask = tf.tile(mask, [tf.shape(data)[0], 1])  # Apply to all batches
        return data * tf.expand_dims(mask, axis=-1)


    @staticmethod
    def random_rotation(data, max_angle):
        """
        Apply random rotations to spatial gesture data.

        Args:
            data: Tensor of shape [batch_size, seq_len, features] (split to x,y,z first, first 2 are spatial).
            max_angle: Maximum rotation angle in degrees.

        Returns:
            Rotated tensor with original shape preserved.
        """
        batch_size, seq_len, features = tf.unstack(tf.shape(data))
        
        # Generate a random rotation angle in radians within [-max_angle, max_angle]
        angle_degrees = tf.random.uniform(shape=[], minval=-max_angle, maxval=max_angle)
        angle_radians = angle_degrees * (math.pi / 180.0)  # Convert degrees to radians
        
        # Construct the 2D rotation matrix
        cos_theta = tf.math.cos(angle_radians)
        sin_theta = tf.math.sin(angle_radians)
        rotation_matrix = tf.stack([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])  # Shape: [2, 2]
        
        # Extract the spatial components (x, y) and leave the rest unchanged
        spatial_data = data[:, :, :2]  # Shape: [batch_size, seq_len, 2]
        other_data = data[:, :, 2:]   # Shape: [batch_size, seq_len, features - 2]
        
        # Apply the rotation matrix to the spatial components
        # Reshape spatial_data for matrix multiplication: [batch_size * seq_len, 2]
        spatial_data_reshaped = tf.reshape(spatial_data, [-1, 2])
        rotated_spatial_data = tf.matmul(spatial_data_reshaped, rotation_matrix)  # Shape: [batch_size * seq_len, 2]
        
        # Reshape back to [batch_size, seq_len, 2]
        rotated_spatial_data = tf.reshape(rotated_spatial_data, [batch_size, seq_len, 2])
        
        # Concatenate the rotated spatial data with the other data
        rotated_data = tf.concat([rotated_spatial_data, other_data], axis=-1)  # Shape: [batch_size, seq_len, features]
        
        return rotated_data


    @staticmethod
    def scale_and_translate(data, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
        """
        Apply random scaling and translation.

        Args:
            data: Tensor of shape [batch_size, seq_len, features].
            scale_range: Tuple, min and max scaling factor.
            shift_range: Tuple, min and max translation.

        Returns:
            Transformed tensor.

        Negative Bounds Only: 
            If you only want to allow movement in one direction (e.g., left or backward), you can set the bounds to be negative.
            
            Example: 
            [-0.2,0] allows only leftward or backward shifts.

        Positive Bounds Only: If you only want to allow movement in the opposite direction (e.g., right or forward), you can set the bounds to be positive.
            Example: 
            [0,0.2] allows only rightward or forward shifts.

        Symmetric Bounds: If you want to allow movement in both directions, the bounds should include both negative and positive values.
            Example: 
            [-0.2,0.2] allows shifts in either direction.
        """
        batch_size, seq_len, features = tf.unstack(tf.shape(data))
        
        # Generate random scaling factors for each feature
        scale_factors = tf.random.uniform(
            shape=[1, 1, features],  # Shape: [1, 1, features] to broadcast across batch and sequence
            minval=scale_range[0],
            maxval=scale_range[1]
        )
        
        # Generate random shift values for each feature
        shift_values = tf.random.uniform(
            shape=[1, 1, features],  # Shape: [1, 1, features] to broadcast across batch and sequence
            minval=shift_range[0],
            maxval=shift_range[1]
        )
        
        # Apply scaling and translation
        transformed_data = data * scale_factors + shift_values
        
        return transformed_data

    @staticmethod
    def add_gaussian_noise(data, stddev=0.01):
        """
        Add random Gaussian noise to the sequence data.

        Args:
            data: Tensor of shape [batch_size, seq_len, features].
            stddev: Standard deviation of the Gaussian noise.

        Returns:
            Noisy tensor.
        """
        # Generate Gaussian noise with the same shape as the input data
        noise = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=stddev, dtype=data.dtype)
                
        return data + noise

    @staticmethod
    def random_dropout(data, drop_rate=0.1):
        """
        Randomly set some feature values to zero.

        Args:
            data: Tensor of shape [batch_size, seq_len, features].
            drop_rate: Float, proportion of features to drop.

        Returns:
            Tensor with dropped values.
        """
        # Generate a random mask with the same shape as the input data
        mask = tf.random.uniform(shape=tf.shape(data), minval=0.0, maxval=1.0, dtype=data.dtype)
        
        # Apply the dropout mask: set values to zero where the mask is less than drop_rate
        dropout_mask = tf.cast(mask > drop_rate, dtype=data.dtype)
        dropped_data = data * dropout_mask
        
        return dropped_data

    @staticmethod
    def mirror_gesture(data):
        """
        Reflect gestures along the vertical axis.

        Args:
            data: Tensor of shape [batch_size, seq_len, features].

        Returns:
            Mirrored tensor.
        """
        # Negate the horizontal spatial component (e.g., x-coordinate)
        mirrored_data = tf.concat([
            -data[:, :, 0:1],  # Negate the x-coordinate (first feature)
            data[:, :, 1:2],   # Keep the y-coordinate (second feature) unchanged
            data[:, :, 2:]     # Keep the remaining features unchanged
        ], axis=-1)
        
        return mirrored_data
    
    @staticmethod
    def reverse_sequence(data):
        """
        Reverse the sequence order.

        Args:
            data: Tensor of shape [batch_size, seq_len, features].

        Returns:
            Reversed tensor.
        """
        return tf.reverse(data, axis=[1])

    @staticmethod
    def augment_gesture_data(data, config={}):
        """
        Apply a random combination of augmentations to gesture data based on the specified mode.

        Args:
            data: Tensor of shape [batch_size, seq_len, features].
            dam: Data augmentation mode (1=light, 2=moderate, 3=strong, 4=all).

        Returns:
            Augmented tensor.
        """
        if not config:
            max_angle=15
            scale_range=(0.9, 1.1) 
            shift_range=(-0.1, 0.1)
            time_rate=0.2
            drop_rate=0.2
            stddev=0.01
            dam=0
        else:
            max_angle=config.get('angle_range', 15)
            scale_range=(config.get('scale_range', 0.9), config.get('scale_l_range', 0.1)) 
            shift_range=(config.get('shift_range', -0.1), config.get('shift_l_range', 0.1))
            time_rate=config.get('time_rate', 0.2)
            drop_rate=config.get('drop_rate', 0.2)
            stddev=config.get('stddev', 0.01)
            dam=config.get('dam',0)

        if dam == 0:
            # No augmentation
            return data

        if dam == 1:
            # Light augmentations
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.add_gaussian_noise(data, stddev=stddev)
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.random_dropout(data, drop_rate=drop_rate)

        elif dam == 2:
            # Moderate augmentations
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.scale_and_translate(data, scale_range=scale_range, shift_range=shift_range)
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.random_rotation(data, max_angle)

        elif dam == 3:
            # Strong augmentations
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.time_warp(data, rate=time_rate)
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.mirror_gesture(data)

        elif dam == 4:
            # All augmentations
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.add_gaussian_noise(data, stddev=stddev)
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.random_dropout(data, drop_rate=drop_rate)
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.scale_and_translate(data, scale_range=scale_range, shift_range=shift_range)
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.random_rotation(data, max_angle)
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.time_warp(data, rate=time_rate)
            if tf.random.uniform(shape=[]) > 0.5:
                data = DataAugmentation.mirror_gesture(data)

        return data

    @staticmethod
    def reverse_batched_token(targets):
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
    
    def reshape_landmarks(self, landmarks_flattened):
        """Reshapes 2D flattened landmarks to 3D landmarks (seq_len, num_features, channels)."""
        return tf.reshape(landmarks_flattened, [-1, self.num_features, self.channels])

    def interpolate_or_pad(self, data, max_len=100, mode="start"):
        """
        Interpolates or pads a TensorFlow tensor along the sequence dimension to ensure it matches `max_len`.

        Args:
            data (tf.Tensor): Input tensor of shape `(seq_len, feature_dim1, feature_dim2)`.
            max_len (int): Target sequence length after interpolation or padding.
            mode (str): Padding mode, only "start" is supported for simplicity.

        Returns:
            tf.Tensor: Processed data of shape `(max_len, feature_dim1, feature_dim2)`.
            tf.Tensor: Mask tensor indicating valid positions (1 for valid, 0 for padded).
        """
        seq_len = tf.shape(data)[0]
        feature_dim1, feature_dim2 = tf.shape(data)[1], tf.shape(data)[2]
        
        # Calculate the difference between the max length and current length
        diff = max_len - seq_len

        # Case 1: Crop the data if it's longer than max_len
        if diff <= 0:
            data = tf.image.resize(data, [max_len, feature_dim1])  # Resize to `max_len`
            mask = tf.ones((max_len,), dtype=tf.float32)  # Create a mask of 1s for all valid positions
            return data, mask

        # Case 2: Pad the data if it's shorter than max_len
        coef = 0  # Padding coefficient (multiplies with padding for flexibility)
        padding = tf.zeros((diff, feature_dim1, feature_dim2), dtype=tf.float32)  # Zero-padding tensor
        mask_padding = tf.zeros((diff,), dtype=tf.float32)  # Mask for padding positions
        mask = tf.ones((seq_len,), dtype=tf.float32)  # Mask for valid positions

        # Concatenate original data and padding
        data = tf.concat([data, padding * coef], axis=0)
        mask = tf.concat([mask, mask_padding * coef], axis=0)

        return data, mask

    
    def sample_row_with_condition(self, participant_id, sequence_id, df):
        """
        Samples a row from a Pandas DataFrame based on conditions using TensorFlow for randomness.

        Args:
            participant_id (int): ID of the participant to match.
            sequence_id (int): ID of the sequence to exclude.
            df (pd.DataFrame): DataFrame containing `participant_id` and `sequence_id` columns.

        Returns:
            pd.Series or None: A sampled row as a Pandas Series if matching rows exist, else None.
        """
        # Step 1: Create a boolean mask for the conditions
        mask = (df['participant_id'] == participant_id) & (df['sequence_id'] != sequence_id)

        # Step 2: Filter rows based on the mask
        filtered_df = df[mask]

        # Step 3: Check if any rows meet the condition
        if not filtered_df.empty:
            # Step 4: Use TensorFlow to sample one row randomly
            random_index = tf.random.uniform(
                shape=[], 
                minval=0, 
                maxval=len(filtered_df), 
                dtype=tf.int32
            ).numpy()
            return filtered_df.iloc[random_index]
        else:
            # Step 5: Return None if no rows match
            return None

    def reverse_sample(self, landmarks, phrase):
        if not tf.equal(tf.strings.length(phrase), 0):
            characters = tf.strings.unicode_split(phrase, 'UTF-8')  # Shape: (N,)
            reversed_characters = tf.reverse(characters, axis=[0])  # Shape: (N,)
            reversed_phrase = tf.strings.reduce_join(reversed_characters)  # Shape: []
            reversed_landmarks = tf.reverse(landmarks, axis=[1])
            return reversed_landmarks, reversed_phrase
        else:
            return landmarks, phrase
    
    def flip_data(self, data, landmarks, symmetry_mapping, flip_aug=0.5):
        """
        Flips the input data based on symmetry mapping and a random probability.

        Args:
            data (tf.Tensor): A tensor of shape (batch_size, num_landmarks, num_features).
            landmarks (tf.Tensor): A tensor of landmark IDs.
            symmetry_mapping (tf.Tensor): A tensor mapping landmarks to their corresponding flipped IDs.
            flip_aug (float): Probability of applying the flip augmentation.

        Returns:
            tf.Tensor: Flipped data if augmentation is applied, else the original data.
        """
        # Map landmarks to their flipped counterparts
        flipped_landmarks = tf.gather(symmetry_mapping, landmarks)

        # Compute the flip array
        landmarks_expanded = tf.expand_dims(landmarks, axis=1)
        flipped_landmarks_expanded = tf.expand_dims(flipped_landmarks, axis=0)
        flip_array = tf.where(landmarks_expanded == flipped_landmarks_expanded)

        # Conditionally flip the data based on random threshold
        def flip(data, flip_array):
            data_flipped = tf.tensor_scatter_nd_update(data, flip_array, tf.gather(data, flip_array[:, 1], axis=1))
            data_flipped[:, :, 0] = -data_flipped[:, :, 0]  # Flip X-coordinates
            return data_flipped

        # Randomly decide whether to apply the flip
        should_flip = tf.random.uniform(()) < flip_aug
        return tf.cond(should_flip, lambda: flip(data, flip_array), lambda: data)
    
    def unbatched_outer_cutmix(self, landmarks1, phrase1, score1, landmarks2, phrase2, score2):
        # Generate a random cutoff value between 0 and 1
        cut_off = tf.random.uniform(shape=[], minval=0, maxval=1)
        
        # Determine the cutoff points for the phrases based on their lengths
        phrase1_length = tf.cast(tf.strings.length(phrase1), tf.float32)  # Length of phrase1
        phrase2_length = tf.cast(tf.strings.length(phrase2), tf.float32)  # Length of phrase2

        # Function to find the cutoff index (simplified)
        def find_cutoff_index(phrase_length, cut_off):
            # Calculate the cutoff index directly
            return tf.cast(tf.round(phrase_length * cut_off), tf.int32)

        # Find the cutoff indices for phrase1 and phrase2
        cut_off_phrase1 = find_cutoff_index(phrase1_length, cut_off)
        cut_off_phrase2 = find_cutoff_index(phrase2_length, cut_off)
        
        # Determine the cutoff points for the landmarks based on their lengths
        landmarks1_length = tf.cast(tf.shape(landmarks1)[0], tf.float32)  # Length of landmarks1 sequence
        landmarks2_length = tf.cast(tf.shape(landmarks2)[0], tf.float32)  # Length of landmarks2 sequence
        cut_off_landmarks1 = tf.clip_by_value(tf.round(landmarks1_length * cut_off), 1, landmarks1_length - 1)
        cut_off_landmarks2 = tf.clip_by_value(tf.round(landmarks2_length * cut_off), 1, landmarks2_length - 1)

        # Randomly decide whether to swap from landmarks2 to landmarks1 or vice versa
        if tf.random.uniform(shape=[]) < 0.5:
            # Combine phrase2 (after cutoff) with phrase1 (up to cutoff)
            new_phrase = tf.strings.join([
                tf.strings.substr(phrase2, cut_off_phrase2, tf.strings.length(phrase2) - cut_off_phrase2),
                tf.strings.substr(phrase1, 0, cut_off_phrase1)
            ])
            # Combine landmarks2 (after cutoff) with landmarks1 (up to cutoff)
            new_landmarks = tf.concat([
                landmarks2[tf.cast(cut_off_landmarks2, tf.int32):, :],
                landmarks1[:tf.cast(cut_off_landmarks1, tf.int32), :]
            ], axis=0)
            # Compute the new score as a weighted combination based on the cutoff
            new_score = cut_off * score1 + (1 - cut_off) * score2
        else:
            # Combine phrase1 (after cutoff) with phrase2 (up to cutoff)
            new_phrase = tf.strings.join([
                tf.strings.substr(phrase1, cut_off_phrase1, tf.strings.length(phrase1) - cut_off_phrase1),
                tf.strings.substr(phrase2, 0, cut_off_phrase2)
            ])
            # Combine landmarks1 (after cutoff) with landmarks2 (up to cutoff)
            new_landmarks = tf.concat([
                landmarks1[tf.cast(cut_off_landmarks1, tf.int32):, :],
                landmarks2[:tf.cast(cut_off_landmarks2, tf.int32), :]
            ], axis=0)
            # Compute the new score as a weighted combination based on the cutoff
            new_score = cut_off * score2 + (1 - cut_off) * score1

        # Return the mixed landmarks, phrase, and score
        return new_landmarks, new_phrase, new_score

    def unbatched_score_based_outer_cutmix(self, landmarks1, phrase1, score1, landmarks2, phrase2, score2):
        # Generate a random cutoff value between 0 and 1
        cut_off = tf.random.uniform(shape=[], minval=0, maxval=1)
        
        # Determine the cutoff points for the phrases based on their lengths
        phrase1_length = tf.cast(tf.strings.length(phrase1), tf.float32)  # Length of phrase1
        phrase2_length = tf.cast(tf.strings.length(phrase2), tf.float32)  # Length of phrase2
        cut_off_phrase1 = tf.clip_by_value(tf.round(phrase1_length * cut_off), 1, phrase1_length - 1)
        cut_off_phrase2 = tf.clip_by_value(tf.round(phrase2_length * cut_off), 1, phrase2_length - 1)
        
        # Determine the cutoff points for the landmarks based on their lengths
        landmarks1_length = tf.cast(tf.shape(landmarks1)[0], tf.float32)  # Length of landmarks1 sequence
        landmarks2_length = tf.cast(tf.shape(landmarks2)[0], tf.float32)  # Length of landmarks2 sequence
        cut_off_landmarks1 = tf.clip_by_value(tf.round(landmarks1_length * cut_off), 1, landmarks1_length - 1)
        cut_off_landmarks2 = tf.clip_by_value(tf.round(landmarks2_length * cut_off), 1, landmarks2_length - 1)

        # Randomly decide whether to swap from landmarks2 to landmarks1 or vice versa
        if tf.random.uniform(shape=[]) < 0.5:
            # Combine phrase2 (after cutoff) with phrase1 (up to cutoff)
            new_phrase = tf.strings.join([
                tf.strings.substr(phrase2, tf.cast(cut_off_phrase2, tf.int32), tf.cast(phrase2_length - cut_off_phrase2, tf.int32)),
                tf.strings.substr(phrase1, 0, tf.cast(cut_off_phrase1, tf.int32))
            ])
            # Combine landmarks2 (after cutoff) with landmarks1 (up to cutoff)
            new_landmarks = tf.concat([
                landmarks2[tf.cast(cut_off_landmarks2, tf.int32):, :],
                landmarks1[:tf.cast(cut_off_landmarks1, tf.int32), :]
            ], axis=0)
            # Compute the new score as a weighted combination based on the cutoff
            new_score = cut_off * score1 + (1 - cut_off) * score2
        else:
            # Combine phrase1 (after cutoff) with phrase2 (up to cutoff)
            new_phrase = tf.strings.join([
                tf.strings.substr(phrase1, tf.cast(cut_off_phrase1, tf.int32), tf.cast(phrase1_length - cut_off_phrase1, tf.int32)),
                tf.strings.substr(phrase2, 0, tf.cast(cut_off_phrase2, tf.int32))
            ])
            # Combine landmarks1 (after cutoff) with landmarks2 (up to cutoff)
            new_landmarks = tf.concat([
                landmarks1[tf.cast(cut_off_landmarks1, tf.int32):, :],
                landmarks2[:tf.cast(cut_off_landmarks2, tf.int32), :]
            ], axis=0)
            # Compute the new score as a weighted combination based on the cutoff
            new_score = cut_off * score2 + (1 - cut_off) * score1

        # Return the mixed landmarks, phrase, and score
        return new_landmarks, new_phrase, new_score

    def batched_outer_cutmix(self, landmark1, phrase1, score1, landmark2, phrase2, score2, eos_token, pad_token):
        # Extract BOS tokens
        bos_column = phrase1[:, :1] 
        
        # Remove BOS from phrase1 and EOS from phrase2
        phrase1 = phrase1[:, 1:]  # Remove BOS from phrase1
        phrase2 = phrase2[:, 1:]  # Remove BOS from phrase2

        # Generate a random cutoff value between 0 and 1
        cut_off = tf.random.uniform(shape=[], minval=0, maxval=1)

        # Determine the cutoff points for the phrases based on their lengths
        phrase1_length = tf.shape(phrase1)[1]  # Length without BOS
        phrase2_length = tf.shape(phrase2)[1]  # Length without EOS
        cut_off_phrase1 = tf.clip_by_value(tf.round(tf.cast(phrase1_length, tf.float32) * cut_off), 1, tf.cast(phrase1_length, tf.float32) - 1)
        cut_off_phrase2 = tf.clip_by_value(tf.round(tf.cast(phrase2_length, tf.float32) * cut_off), 1, tf.cast(phrase2_length, tf.float32) - 1)

        # Determine the cutoff points for the landmarks based on their lengths
        landmark1_length = tf.shape(landmark1)[1]  # Length of each landmark sequence (128)
        landmark2_length = tf.shape(landmark2)[1]  # Length of each landmark sequence (128)
        cut_off_landmark1 = tf.clip_by_value(tf.round(tf.cast(landmark1_length, tf.float32) * cut_off), 1, tf.cast(landmark1_length, tf.float32) - 1)
        cut_off_landmark2 = tf.clip_by_value(tf.round(tf.cast(landmark2_length, tf.float32) * cut_off), 1, tf.cast(landmark2_length, tf.float32) - 1)

        # Randomly decide whether to swap from landmark2 to landmark1 or vice versa
        if tf.random.uniform(shape=[]) < 0.5:
            mixed_phrase = tf.concat([
                phrase2[:, tf.cast(cut_off_phrase2, tf.int32):], 
                phrase1[:, :tf.cast(cut_off_phrase1, tf.int32)]
            ], axis=1)
            mixed_landmark = tf.concat([
                landmark2[:, tf.cast(cut_off_landmark2, tf.int32):, :], 
                landmark1[:, :tf.cast(cut_off_landmark1, tf.int32), :]
            ], axis=1)
            new_score = cut_off * score1 + (1 - cut_off) * score2
        else:
            mixed_phrase = tf.concat([
                phrase1[:, tf.cast(cut_off_phrase1, tf.int32):], 
                phrase2[:, :tf.cast(cut_off_phrase2, tf.int32)]
            ], axis=1)
            mixed_landmark = tf.concat([
                landmark1[:, tf.cast(cut_off_landmark1, tf.int32):, :], 
                landmark2[:, :tf.cast(cut_off_landmark2, tf.int32), :]
            ], axis=1)
            new_score = cut_off * score2 + (1 - cut_off) * score1

        # Add BOS tokens back
        new_phrase = tf.concat([bos_column, mixed_phrase], axis=1)

        # # Remove pad_token and eos_token while preserving batch size
        # mask = tf.logical_and(new_phrase != eos_token, new_phrase != pad_token)
        # new_phrase = tf.ragged.boolean_mask(new_phrase, mask).to_tensor(default_value=pad_token)

        # # Add EOS token at the end
        # eos_column = tf.fill([tf.shape(new_phrase)[0], 1], eos_token)
        # new_phrase = tf.concat([new_phrase, eos_column], axis=1)

        # # Pad back to original phrase length if needed
        # original_length = tf.shape(phrase1)[1] + 1  # Including EOS
        # current_length = tf.shape(new_phrase)[1]
        
        # pad_amount = tf.maximum(original_length - current_length, 0)  # Ensure non-negative padding
        # pad_tokens = tf.fill([tf.shape(new_phrase)[0], pad_amount], pad_token)
        # new_phrase = tf.concat([new_phrase, pad_tokens], axis=1)

        # Return the mixed landmarks, phrases, and score
        return mixed_landmark, new_phrase, new_score


    def batched_outer_cutmix_jax(self, key, landmark1, phrase1, score1, landmark2, phrase2, score2):
        """
        Efficient implementation of the batched outer CutMix operation using JAX.

        Args:
            key: PRNG key for randomness in JAX.
            landmark1, phrase1, score1: Inputs from the first batch (convert to JAX arrays).
            landmark2, phrase2, score2: Inputs from the second batch (convert to JAX arrays).

        Returns:
            new_landmark, new_phrase, new_score: Mixed outputs as JAX arrays.
        """
        # Ensure inputs are JAX arrays
        landmark1 = jnp.asarray(landmark1)
        phrase1 = jnp.asarray(phrase1)
        score1 = jnp.asarray(score1)
        landmark2 = jnp.asarray(landmark2)
        phrase2 = jnp.asarray(phrase2)
        score2 = jnp.asarray(score2)

        # Generate a random cutoff value between 0 and 1
        key, subkey = random.split(key)
        cut_off = random.uniform(subkey, shape=())

        # Determine cutoff points for phrases
        phrase1_length = phrase1.shape[1]
        phrase2_length = phrase2.shape[1]
        cut_off_phrase1 = jnp.clip(jnp.round(phrase1_length * cut_off), 1, phrase1_length - 1).astype(int)
        cut_off_phrase2 = jnp.clip(jnp.round(phrase2_length * cut_off), 1, phrase2_length - 1).astype(int)

        # Determine cutoff points for landmarks
        landmark1_length = landmark1.shape[1]
        landmark2_length = landmark2.shape[1]
        cut_off_landmark1 = jnp.clip(jnp.round(landmark1_length * cut_off), 1, landmark1_length - 1).astype(int)
        cut_off_landmark2 = jnp.clip(jnp.round(landmark2_length * cut_off), 1, landmark2_length - 1).astype(int)

        # Randomly decide swap direction
        key, subkey = random.split(key)
        swap_condition = random.uniform(subkey, shape=()) < 0.5

        def combine_batches():
            # Combine phrase2 and phrase1 (swap order)
            new_phrase = jnp.concatenate(
                [phrase2[:, cut_off_phrase2:], phrase1[:, :cut_off_phrase1]], axis=1
            )
            # Combine landmark2 and landmark1 (swap order)
            new_landmark = jnp.concatenate(
                [landmark2[:, cut_off_landmark2:, :], landmark1[:, :cut_off_landmark1, :]],
                axis=1,
            )
            # Compute new score
            new_score = cut_off * score1 + (1 - cut_off) * score2
            return new_landmark, new_phrase, new_score

        def combine_batches_reverse():
            # Combine phrase1 and phrase2 (original order)
            new_phrase = jnp.concatenate(
                [phrase1[:, cut_off_phrase1:], phrase2[:, :cut_off_phrase2]], axis=1
            )
            # Combine landmark1 and landmark2 (original order)
            new_landmark = jnp.concatenate(
                [landmark1[:, cut_off_landmark1:, :], landmark2[:, :cut_off_landmark2, :]],
                axis=1,
            )
            # Compute new score
            new_score = cut_off * score2 + (1 - cut_off) * score1
            return new_landmark, new_phrase, new_score

        # Ensure inputs in JAX operations are arrays
        new_landmark, new_phrase, new_score = jax.lax.cond(
            swap_condition,
            lambda: combine_batches(),
            lambda: combine_batches_reverse(),
        )

        # Convert JAX outputs to TensorFlow tensors
        new_landmark = tf.convert_to_tensor(new_landmark)
        new_phrase = tf.convert_to_tensor(new_phrase)
        new_score = tf.convert_to_tensor(new_score)

        return new_landmark, new_phrase, new_score


    def apply_flip(self, landmarks):
        """Applies random flip augmentation to a batch of inputs and targets."""
        batch_size = tf.shape(landmarks)[0]
        flipped_inputs = tf.image.random_flip_left_right(landmarks)
        return flipped_inputs

    def random_translation(self, landmarks_flattened):
        """Shifts the landmarks randomly in the x and y directions, keeping z intact."""
        translation_range = self.aug_param.get('translation_range', 0.0)  # Set default to 0.0 if not present
        if translation_range is None or translation_range <= 0:
            return landmarks_flattened
        landmarks = self.reshape_landmarks(landmarks_flattened)  # Reshape to 3D
        self.logger.debug('translation_range : ', translation_range,  ' rand : ', self.rand)
        #Set default translation range if it's None
        if translation_range is None:
            translation_range = 0.0  # Set to 0 or any default value to avoid translation
            raise ValueError("translation_range cannot be None.")

        translation = tf.random.uniform(shape=(2,), minval=-translation_range, maxval=translation_range)

        # Extract x and y coordinates
        xy = landmarks[:, :, :2]  # Keep x and y only

        # Apply translation only to x and y
        xy_augmented = xy + translation

        # Recombine with the original z coordinates
        augmented_landmarks = tf.concat([xy_augmented, landmarks[:, :, 2:]], axis=-1)  # Keep z unchanged

        return augmented_landmarks

    def random_rotation(self, landmarks_flattened):
        """Rotates the landmarks around their center by a random angle."""

        angle_range = self.aug_param.get('angle_range', 0.0)  # Set default to 0.0 if not present
        if angle_range is None or angle_range <= 0:
            return landmarks_flattened
        self.logger.debug('angle_range : ', angle_range,  ' rand : ', self.rand)

        landmarks = self.reshape_landmarks(landmarks_flattened)  # Reshape to 3D
        angle = tf.random.uniform([], -angle_range, angle_range)
        radians = tf.cast(tf.multiply(angle, np.pi / 180.0), tf.float32)

        # Extract x and y coordinates
        xy = landmarks[:, :, :2]

        # Compute the center of the landmarks (for each sequence step)
        center = tf.reduce_mean(xy, axis=1, keepdims=True)

        # Create rotation matrix using tf operations
        cos_angle = tf.cos(radians)
        sin_angle = tf.sin(radians)
        rotation_matrix = tf.stack([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        # Apply rotation
        centered_xy = xy - center  # Center the landmarks
        rotated_xy = tf.tensordot(centered_xy, rotation_matrix, axes=1) + center

        # Recombine with the original z coordinates
        augmented_landmarks = tf.concat([rotated_xy, landmarks[:, :, 2:]], axis=-1)

        return augmented_landmarks

    def random_scaling(self, landmarks_flattened, scale_range=0.2):
        """Scales the landmarks by a random factor, keeping the center unchanged."""

        scale_range = self.aug_param.get('scale_range', 0.0)  # Set default to 0.0 if not present
        if scale_range is None or scale_range <= 0:
            return landmarks_flattened
        self.logger.debug('scale_range : ', scale_range,  ' rand : ', self.rand)

        landmarks = self.reshape_landmarks(landmarks_flattened)  # Reshape to 3D
        scale = tf.random.uniform([], 1 - scale_range, 1 + scale_range)

        # Extract x and y coordinates
        xy = landmarks[:, :, :2]

        # Compute the center of the landmarks
        center = tf.reduce_mean(xy, axis=1, keepdims=True)

        # Scale landmarks relative to the center
        scaled_xy = (xy - center) * scale + center

        # Recombine with the original z coordinates
        augmented_landmarks = tf.concat([scaled_xy, landmarks[:, :, 2:]], axis=-1)

        return augmented_landmarks

    def random_shearing(self, landmarks_flattened, shear_range=0.2):
        """Applies a random shear transformation to the landmarks."""

        shear_range = self.aug_param.get('shear_range', 0.0)  # Set default to 0.0 if not present
        if shear_range is None or shear_range <= 0:
            return landmarks_flattened
        self.logger.debug('shear_range : ', shear_range,  ' rand : ', self.rand)

        landmarks = self.reshape_landmarks(landmarks_flattened)  # Reshape to 3D
        shear_factor = tf.random.uniform([], -shear_range, shear_range)

        # Extract x and y coordinates
        xy = landmarks[:, :, :2]

        # Create shear matrix (shear along the x-axis)
        shear_matrix = tf.stack([[1.0, shear_factor], [0.0, 1.0]])

        # Apply shear transformation
        sheared_xy = tf.tensordot(xy, shear_matrix, axes=1)

        # Recombine with the original z coordinates
        augmented_landmarks = tf.concat([sheared_xy, landmarks[:, :, 2:]], axis=-1)

        return augmented_landmarks

    def augment_landmarks(self, landmarks_flattened, out_shape=2):
        """Args
               landmarks_flattened : 3D Tensor (length, num_features, channels) 
               out_shape: int for specifying output shape
        """
        # Define Augmentation methods
        aug_methods = [[0,1], # DAM1
                       [1,0,2], # DAM2
                       [3,0,1], # DAM3
                       [2,0,1,3]] # DAM4

        # Randomly select augmentation methods
        self.rand = self.aug_param.get('dam', 10)  # Set default to 10 if not present
        if self.rand == 10:
            self.rand = random.randint(0, len(aug_methods) - 1)

        # Select method to apply
        selected_methods = aug_methods[self.rand]

        if tf.not_equal(tf.rank(landmarks_flattened), 3):
            landmarks_flattened = tf.reshape(landmarks_flattened, [-1, self.num_features, self.channels])

        if 0 in selected_methods:
            landmarks_flattened = self.random_translation(landmarks_flattened)
        if 1 in selected_methods:
            landmarks_flattened = self.random_rotation(landmarks_flattened)
        if 2 in selected_methods:
            landmarks_flattened = self.random_scaling(landmarks_flattened)
        if 3 in selected_methods:
            landmarks_flattened = self.random_shearing(landmarks_flattened)
        if not selected_methods:
            raise ValueError("Error: Invalid augmentation method selected.")

        # Convert 3D landmarks back to a desired shape
        if out_shape == 1:
            reshaped_landmarks = tf.reshape(landmarks_flattened, [-1])
        elif out_shape == 2:
            reshaped_landmarks = tf.reshape(landmarks_flattened, [-1, self.total_num_features])
        elif out_shape == 3:
            reshaped_landmarks = tf.reshape(landmarks_flattened, [-1, self.num_features, self.channels])
        else:
            raise ValueError("Invalid shape. Use '1D', '2D', or '3D'.")
        return reshaped_landmarks

    def augment_dataset(self, datasets, file_dataset, num_component=2):
        """
        Args
        datasets: list of tf.data.Dataset objects
        file_dataset: sliced dataset of landmarks, phrases, and contexts yet to be concatenated

        Returns:

        """
        # Apply data augmentation to landmarks_data multiple times
        if self.num_augmentations > 0:
            for _ in range(self.num_augmentations):
                if num_component == 2:
                    augmented_dataset = file_dataset.map(lambda x, y: ((self.augment_landmarks(x, out_shape=1), y)))
                else:
                    augmented_dataset = file_dataset.map(lambda x, y, z: ((self.augment_landmarks(x, out_shape=1), y, z)))
                # Append the augmented dataset to the list
                datasets.append(augmented_dataset)
        else:
            datasets.append(file_dataset)


# aug_param = {"translation_range":0.3, # Tunable Parameter
#              "angle_range":45, 
#              "scale_range":0.2, 
#              "shear_range":0.2,
#              "dam":0} 