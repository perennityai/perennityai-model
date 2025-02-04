import tensorflow as tf

def reverse_batched_token(self, targets):
    """
    Reverse sequences in targets but keep start and end tokens in place.
    
    Args:
        targets: Tensor of target sequences with shape (batch_size, seq_len, num_features).
    
    Returns:
        Tensor with reversed sequences, excluding start and end tokens, maintaining input shape.
    """
    # Ensure input shape is 3D
    tf.debugging.assert_rank(targets, 3, message="Input must have shape (batch_size, seq_len, num_features)")

    # Identify the start and end tokens along the sequence dimension
    start_tokens = targets[:, :1, :]  # First token (batch_size, 1, num_features)
    end_tokens = targets[:, -1:, :]  # Last token (batch_size, 1, num_features)

    # Reverse the middle portion of the sequence along the sequence dimension
    middle_tokens = targets[:, 1:-1, :]  # Middle tokens (batch_size, seq_len-2, num_features)
    reversed_middle = tf.reverse(middle_tokens, axis=[1])  # Reverse along seq_len (axis=1)

    # Concatenate start, reversed middle, and end tokens along the sequence dimension
    reversed_targets = tf.concat([start_tokens, reversed_middle, end_tokens], axis=1)

    return reversed_targets



# Example input: batch of 2 sequences, each of length 5, with 3 features per token
input_targets = tf.constant([
    [[101, 1, 1], [10, 2, 2], [20, 3, 3], [30, 4, 4], [102, 5, 5]],  # Batch 1
    [[201, 6, 6], [15, 7, 7], [25, 8, 8], [35, 9, 9], [202, 10, 10]]  # Batch 2
])

# Initialize and call the function
reversed_targets = reverse_batched_token(None, input_targets)

tf.print("Reversed Targets:")
tf.print(reversed_targets)
