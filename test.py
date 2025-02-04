import tensorflow as tf
from utils import DataAugmentation


@tf.function
def test_augment(data, config={}):
    return DataAugmentation.augment_gesture_data(data, config=config)

# Example input
config = {"translation_range": 0.3,
                      "angle_range": 35, 
                      "scale_range": 0.2,
                      "scale_l_range": 0.1, 
                      "shear_range": 0.2,
                      "shift_range": 0.2,
                      "shift_l_range": 0.1,
                      "time_rate": 0.2,
                      "drop_rate": 0.2,
                      "stddev": 0.01,
                      "dam": 1}
data = tf.random.normal([32, 100, 3])  # Example input [batch_size, seq_len, features]
print(data.shape)
augmented_data = test_augment(data, config=config)
print(augmented_data.numpy())
print(augmented_data.shape)



