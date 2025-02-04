# perennityai-tggmt Repository
The **PerennityAI Three-stage Gesture Generative Multi-modal Transformer (TGGMT)** is a deep learning-based system designed to recognize and interpret gestures, including sign language and dynamic gestures, from diverse datasets. The system uses a three-stage training approach that integrates gesture recognition and natural language generation in a synergistic workflow, leveraging advanced transformer-based architectures.

# Key Features and Stages:
## 1. Stage 1: Multi-modal Transformer Training
- The Multi-modal Transformer is trained on the original gesture dataset to establish a robust baseline model for gesture recognition.
- Both token-level and word-level configurations are supported to handle granular and contextual gesture representations.
- **Bidirectional Loss Training:**
    - The model is trained on both the original and reversed sequences to capture bidirectional dependencies in gesture patterns.
    - Losses from the forward and reversed sequences are computed independently and combined during training to improve model robustness.
    - Improves the model's ability to understand gesture sequences regardless of their temporal direction.
## Stage 2: Dataset Scoring and CutMix Augmentation
- A scoring mechanism is applied to the dataset to evaluate gesture sequences.
- A CutMix approach is employed to generate auxiliary data, blending sequences and labels to enhance diversity and improve model robustness.
- The augmented dataset is used for auxiliary learning, reinforcing the Multi-modal Transformer.

# Stage 3: GPT-2 Integration and Conditional Gradient Updates
- The gesture model is integrated with GPT-2, enabling gesture-conditioned natural language generation.
## a. GPT training is performed in two mode
- Next Word Prediction: Trains the model to predict the next word in a sequence, supporting word-level text generation.
- Next Sub-word or Character Prediction: Focuses on finer granularity, enabling robust character-based generative capabilities.
## b. Data Augmentation Transformations such as:
- Time Warping: Simulates gestures performed at varying speeds.
- Scale and Translate: Mimics variations in gesture size and position.
- Random Rotation: Adds robustness to different viewing angles.
- Gaussian Noise: Simulates real-world imperfections in sensor data
- Mirror Gesture: Enables the model to handle mirrored gestures.
- Random Dropout: Simulates feature corruption or loss.
## c. Validation loss-driven augmentation strategy (DAM 0-N):
- The system implements a validation loss-driven augmentation strategy (DAM 0-N), dynamically scaling augmentation intensity based on experiment-specific configurations to ensure progressive model refinement instead of abrupt early stopping.
- This strategy actively monitors edit accuracy thresholds and adjusts augmentation levels in response to validation loss stability, optimizing gesture recognition generalization while mitigating overfitting risks. Furthermore,
## d. Conditional gradient:
- Updates are applied to selectively refine the Transformer based on model performance metrics.

# Technical Highlights:
## Three-Stage Training Framework:
- Stage-wise training ensures a progressive enhancement of gesture and generative capabilities.

## Preprocessing Utilities:
- Includes tools for dataset handling, feature engineering, and hyperparameter tuning.

## Stack Strategy:
- Groups diverse augmentations into manageable subsets (DAM1 to DAM4), facilitating robust training.

## Token and Word Configurations:
- Offers flexibility in recognizing gestures at different levels of granularity.

## Generative Model Integration:
- Seamlessly combines gesture recognition with GPT-2 for multi-modal applications.

## Dual GPT-2 Training Modes:
- Supports next word prediction and next sub-word/character prediction for versatile language generation.

## Use Cases:
- Sign Language Translation: From gestures to contextual natural language text.
- Dynamic Gesture Recognition: For interactive AI systems and assistive technologies.
- Gesture-conditioned Text Generation: Enhancing generative language models with auxiliary gesture signals.

## Key Features of TGGMT
- **1. Three-Stage Training Framework**
    - Stage 1: Trains a Multi-modal Transformer on the original dataset to establish a robust baseline for gesture recognition.
    - Stage 2: Augments the dataset using scoring and CutMix techniques, introducing diversity for auxiliary learning.
    - Stage 3: Integrates the gesture model with GPT-2, leveraging a conditional gradient update strategy to enhance gesture-informed text generation.
- **2. Multi-modal Data Integration**
    - Seamlessly combines gesture recognition and natural language generation, creating a unified system for multi-modal tasks.
    - Supports gesture-conditioned text generation and token-level/word-level configurations for dynamic interaction.
- **3. Advanced Data Augmentation Pipeline**
    - Implements a comprehensive augmentation strategy to improve model generalization:
    - Time Warping: Simulates gestures at varying speeds.
    - Scale and Translation: Mimics changes in gesture size and position.
    - Random Rotation: Adds robustness to different viewing angles.
    - Gaussian Noise, Dropout, and Mirroring: Improves handling of noisy, incomplete, or mirrored gestures.
    - Organizes augmentations into stacked strategies (DAM1 to DAM4) for progressive learning.
- **4. Scoring and Auxiliary Learning**
    - Scores datasets to evaluate sequence quality and combines them with CutMix for enhanced training.
    - Encourages auxiliary learning by blending original and augmented datasets, improving model performance under varied conditions.
- **5. GPT-2 Integration with Conditional Updates**
    - Combines gesture recognition with GPT-2 for gesture-informed text generation.
    - Applies conditional gradient updates to the Transformer model, updating weights based on performance metrics.
- **6. Distributed and Configurable Training**
    - Supports distributed training with TensorFlow strategies for efficient scaling across multiple GPUs.
    - Offers flexible configurations for dataset preprocessing, hyperparameter tuning, and model customization.
- **7. Robust Metrics and Evaluation**
    - Tracks performance with key metrics:
    - Edit Accuracy: Evaluates gesture recognition fidelity.
    - Perplexity: Measures language model efficiency.
    - Enables conditional gesture model updates based on metric thresholds, ensuring high-quality predictions.
- **8. Domain Adaptability**
    - Tailored for gesture-based applications such as:
    - Sign Language Translation: From gestures to contextual text.
    - Human-Computer Interaction: Dynamic gesture recognition for assistive or interactive technologies.
    - Gesture-Conditioned Language Generation: Enhances creative content generation.

# Why Use TGGMT?
- Boosts performance of generative language models by incorporating gesture-based contextual signals.
- Ideal for multi-modal applications such as human-computer interaction, assistive technologies, and creative content generation.
- Provides flexibility in training configurations with modular components and conditional updates.

# Key Components
1. **preprocessing/**
Contains preprocessing scripts and modules, including feature engineering code to prepare input data for training.

- ```perennityai-feature-engine/```: Feature engineering pipeline that processes and prepares raw data for use in model training, ensuring that it is in the correct format and includes necessary transformations.
2. **models/**
This directory includes various deep learning model architectures for gesture recognition. The models use transformer-based approaches for both individual gesture tokens and video-based gesture recognition.

- ```transformer_gesture_recognition/```: Contains models and scripts specific to gesture recognition using transformers, including:

- ```base_transformer.py```: The base class for transformer models used for gesture recognition.
- ```static_transformer.py```: Transformer model variant for token-level gesture recognition, focused on static gestures.
- ```fingerspelling_transformer.py```: Transformer model variant for word-level gesture recognition, focused on fingerspelling.
- ```dynamic_transformer.py```: Transformer model variant for word-level gesture recognition, focused on dynamic gestures.

- transformer_gesture_video/: Models that handle video-based gesture recognition using transformers with both spatial and temporal components:

- ```base_video_transformer.py```: Base class for video-based gesture recognition models.
- ```temporal_transformer.py```: Transformer model focused on processing the temporal (time-dependent) aspect of gesture videos.
- ```spatio_temporal_transformer.py```: Combines both spatial and temporal attention to model gestures in videos effectively.

3. **tuner/:** 
- Contains scripts for hyperparameter tuning of models using Keras Tuner, which helps in optimizing model performance.

- ```transformer_gesture_hypermodel.py```: Hyperparameter tuner for the gesture recognition transformer models.
- ```transformer_gesture_video_hypermodel.py```: Hyperparameter tuner for the video-based gesture recognition models.

4. **trainers/**
Includes scripts and base classes to train the models on the datasets.

- ```gesture_trainer.py```: Base class for model training, supporting multiple model types, including gesture recognition models.
- ```gesture_tuner.py```: Trainer script specifically designed to train models with federation datasets, enabling fine-tuning and model validation.

5. **utils/**
Contains utility modules that help with various tasks, such as logging, error handling, and data processing.

6. **gpt2_wrapper: **
- Extends GPT-2 with gesture-conditioned inputs for enhanced text generation.
- common/: General utilities for logging, error handling, and file processing (e.g., reading and writing CSV/JSON files).
- ```csvhandler.py, json_handler.py, logger.py```: Utilities to manage logs and handle files (CSV, JSON).
- ```error_handler.py```: Provides mechanisms to catch and log errors during training or inference.
- gesture_recognition/: Utilities specifically for gesture recognition, such as data augmentation, tokenization, and metrics calculation.
- ```callbacks.py```: Custom training callbacks for monitoring model performance during training.
- ```char_tokenizer.py```: Tokenizer that interprets gestures at the character level.
- ```tokenizer_factory.py```: Factory for creating tokenizers, likely used for word-level gesture recognition.
- ```data_augmentation.py```: Augmentation functions to apply to gesture data (e.g., rotating, scaling, etc.).
- ```combined_metrics_calculator.py```: Utility for calculating combined metrics for gesture recognition tasks.

# Use Cases:
- Gesture-conditioned text generation.
- Multi-modal AI systems for interactive applications.
- Improving generative language models with auxiliary tasks.

# Ripository Structure 
``` bash
perennityai-tggmt/
├── doc/
│   ├── documentaation.docx
├── readme/
│   ├── transformer_model/
│   │   └── README.md
│   ├── tggmt_model/
│       └── README.md
│   ├── vision_model/
│       └── README.md
├── configs/
│   ├── static_config.json                        # Main configuration file for static gesture processing, covering global settings.
│   ├── dynamic_config.json                       # Main configuration file for dynamic gesture processing, covering global settings.
│   ├── fingerspelling_config.json                # Main configuration file for fingerspelling gesture processing, covering global settings.
│   ├── kaggle_fingerspelling_config.json         # Main configuration file for the Kaggle fingerspelling dataset, covering global settings.
│   ├── static_features_config.ini                # Configuration file detailing feature engineering settings for static gesture data.
│   ├── dynamic_features_config.ini               # Configuration file detailing feature engineering settings for dynamic gesture data.
│   ├── fingerspelling_features_config.ini        # Configuration file detailing feature engineering settings for fingerspelling gesture data.
│   └── kaggle_fingerspelling_features_config.ini # Configuration file for feature engineering specific to the Kaggle fingerspelling dataset.and dynamic features.
├── datasets/
│   ├── federation_0_dataset/
│   |   ├── preprocessed
│   │   │   ├── fingerspelling/
│   │   │   │   └── federation_0_fingerspelling.csv
│   │   │   └── non_fingerspelling/
│   │   │       └── federation_0_non_fingerspelling.csv
│   │   ├── train_landmarks
│   │   |   ├── 1.tfrecord  
│   │   |   ├── ....
│   │   |   └── n.tfrecord
│   |   ├── ggmt_char_to_prediction_index.json
│   |   └── ggmt_phrase_to_prediction_index.json
│   ├── ...
│   └── federation_5_dataset/
│   |   ├── preprocessed
│   │   │   ├── fingerspelling/
│   │   │   │   └── federation_0_fingerspelling.csv
│   │   │   └── non_fingerspelling/
│   │   │       └── federation_0_non_fingerspelling.csv
│   │   ├── train_landmarks
│   │   |   ├── 1.tfrecord  
│   │   |   ├── ....
│   │   |   └── n.tfrecord
│   |   ├── ggmt_char_to_prediction_index.json
│   |   └── ggmt_phrase_to_prediction_index.json
├── preprocessing/
│   ├── perennityai-feature-engine/
│   │   └── __init__.py
├── models/
│   ├── gesture_gmt/
│   │   ├── ts_ggmt.py                   # Three-stage GGMT model
│   ├── gesture_seq2seq/
│   │   ├── attention_seq2seq.py          # Multi-attention translator (LSTM)
│   │   ├── no_attention_seq2seq.py       # Single attention translator (LSTM)
│   │   ├── simple_seq2seq.py             # Single attention translator
│   ├── transformer_gesture_recognition/
│   │   ├── base_transformer.py           # Base transformer model with polymorphic design
│   │   ├── static_transformer.py         # Transformer model variant for token-level
│   │   ├── fingerspelling_transformer.py # Transformer model variant for word-level (fingerspelling)
│   │   ├── dynamic_transformer.py        # Transformer model variant for word-level (dynamic)
│   │   └── __init__.py
│   ├── transformer_gesture_video/
│   │   ├── base_video_transformer.py      # Base transformer model for video-based gestures
│   │   ├── temporal_transformer.py        # Transformer model with temporal attention for video gestures
│   │   ├── spatio_temporal_transformer.py # Transformer model with both spatial and temporal attention
│   │   └── __init__.py
│   ├── tuner/
│   │   ├── transformer_gesture_hypermodel.py # Keras tuner for gesture recognition model
│   │   ├── transformer_gesture_video_hypermodel.py # Keras tuner for gesture video recognition model
│   ├── model_factory.py                   # All models factory class
│   ├── gpt_model.py                       # GPT-2 Model
│   ├── rl_env_search.py                   # Reinforcement Learning Agent
│   │   
│   └── __init__.py
├── trainers/
│   ├── gesture_trainer.py               # Base class for model training, supports multiple models
│   ├── gesture_tuner.py                 # Trainer for federation datasets
│   ├── gesture_rl_agent_trainer.py      # Reinforcement Learning Agent Trainer
│   ├── ggmt_trainer.py                  # Gesture Generative Transformer Model Trainer
│   ├── gpt_trainer.py                   # GPT-2 Trainer
│   └── __init__.py
├── utils/
│   ├── common/
│   │   ├── csvhandler.py                 # CSV and parquet file reader and writer
│   │   ├── error_handler.py              # Utility to log messages
│   │   ├── json_handler.py               # Utility to log messages
│   │   ├── configpartser_handler.py      # Utility to log messages
│   │   ├── logger.py                     # Utility to log messages
│   │   └── __init__.py
│   ├── gesture_recognition/
│   │   ├── callbacks.py                  # Utility to define training callbacks
│   │   ├── char_tokenizer.py             # Utility to char interpret tokens
│   │   ├── word_tokenizer.py             # Utility to char interpret tokens
│   │   ├── tokenizer_factory.py          # Utility to word and char interpret tokens
│   │   ├── data_augmentation.py          # Utility to augment landmarks
│   │   ├── combined_metrics_calculator.py# Utility to gesture transformer calculate metrics
│   │   └── __init__.py
│   ├── shared/
│   │   ├── combined_metrics_calculator.py    # Model Custom metrics calculator
│   │   ├── score_calculator.py               # metrics algorithms implementation
│   └── __init__.py
├── tester/
│   ├── transformer_model_tester.py       # Implementation for Keras and TFLite Model Inference
├── seq2seq_tuner.py                      # Script to tune the seq-to-seq model
├── test_gesture_transformer.py           # Script to test the trained gesture transformer model
├── train_gesture_rl_agent.py             # Script to train a reinforcement learning agent for gestures
├── train_gesture_transformer.py          # Script to train the fingerspelling gesture transformer model
├── train_ggmt.py                         # Script to train the TGGMT model
├── train_gpt.py                          # Script to train the GPT-2 model
├── train_seq2seq.py                      # Script to train the seq-to-seq model
├── train_tggmt.py                        # Script to train the TGGMT model
├── tune_gesture_transformer.py           # Script to tune the fingerspelling gesture transformer
```

## Installation and Setup

### 1. Clone the Repository

Use the following command to clone the repository:

```bash
git clone https://YOUR_GITHUB_SECRET_TOKEN@github.com/perennity/perennityai-tggmt.git
cd perennityai-tggmt
pip install -r requirements.txt
```

The data processor is maintained in another repo. mkdir preprocessing and clone the perennityai_feature_engine repo into a local directory ./preprocessing, using ```cd preprocessing && git clone https://github.com/perennityai/perennityai-feature-engine.git```


2. Prerequisites
Ensure you have Python 3.x installed.

3. Create and Activate a Conda Environment
Create a new environment and install required dependencies:

```bash
conda create --name ggmt-env python=3.10
conda activate ggmt-env
pip install -r requirements.txt
```

4. Set Up Jupyter Kernel
To use this environment in Jupyter Notebook, add it as a kernel:

```bash
python -m ipykernel install --user --name ggmt-env --display-name "ggmt-env"
```

# Configuration
Configuration files are located in the configs/ directory. Each configuration file is responsible for a specific aspect of the system:

- Global Settings: Defined in config.ini.
- Feature Engineering: Managed through ```*_features_config.ini```.

Make sure to modify these files based on your dataset and model requirements before training.

# Preprocessing
Preprocessing is handled through the preprocessing/ directory, which contains feature engineering modules. The ```perennityai-feature-engine/``` folder contains the core preprocessing functions used to process input data.

# Models
The models/ directory contains the model architectures for gesture recognition:

- Transformer Gesture Recognition Models:
- ```base_transformer.py```: base class transformer model for gesture recognition.
- ```static_transformer.py```: Static transformer model for gesture recognition.
- ```dynamic_transformer.py```: Dynamic transformer model for gesture recognition.
- ```fingerspelling_transformer.py```: Fingerspelling transformer model for gesture recognition.

- Video-based Gesture Recognition:
- ```base_video_transformer.py```: Base model for video gesture recognition.
- ```spatio_temporal_transformer.py```: Spatio-temporal transformer model combining both spatial and temporal attention mechanisms.

# Training
The trainers/ directory contains scripts to train the models:

- ```gesture_trainer.py```: This class provides a base structure for training models and supports multiple architectures.
- ```gesture_tuner.py```: This file provides functionality for training models with federation datasets.

To train a model, configure your dataset and model settings, and then run the appropriate training script.

# Testing and Evaluation
Testing scripts and utilities are available in the test/ directory. The testing framework is designed to evaluate the model’s performance and report metrics.

# Utilities
## Logging and Error Handling
The ```utils/common/``` directory contains utilities for logging, error handling, and JSON/CSV manipulation.
## Data Augmentation
The ```utils/gesture_recognition/data_augmentation.py``` file provides augmentation techniques to enhance the gesture data.
## Tokenizers
Utilities for tokenization are provided in ```utils/gesture_recognition/char_tokenizer.py``` and ```utils/gesture_recognition/tokenizer_factory.py```.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Contact
For further inquiries, please contact the project maintainers at ```info@perennityai.com```.
