import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pyrsdameraulevenshtein import similarity_str,normalized_distance_str

class CallbackManager(tf.keras.callbacks.Callback):
    def __init__(
        self, tokenizer=None, output_path="", model_saver=None, logger=None
    ):
        self.tokenizer = tokenizer    
        self.output_path = output_path
        self.logger = logger
        self.model_saver = model_saver
        self.is_tggmt = False if model_saver is None else True

    @staticmethod
    def create_model_saver(gesture_transformer, gpt2_wrapper, gesture_weights_path,  gpt2_model_path, logger, monitor="val_loss"):
        return  ModelSaverCallback(
            gesture_transformer=gesture_transformer,
            gpt2_wrapper=gpt2_wrapper,
            gesture_weights_path=gesture_weights_path,
            gpt2_model_path=gpt2_model_path,
            logger=logger, 
            monitor=monitor
        )

    def create_callbacks(self, data=None, 
                         include_checkpoint=True, 
                         include_display=True,
                         include_earlystop=True,
                         include_reducelr=True,
                         include_savemodel=False,
                         include_augmentation=False,
                         pipeline=None,
                         earlystop_monitor='val_loss',
                         reducelr_monitor='val_edit_dist',
                         patience=3,
                         lr_factor=0.5,
                         verbose=0):
        
        callbacks =  []
        if include_earlystop:
            # EarlyStopping
            earlystopping_cb = EarlyStopping(
                        monitor=earlystop_monitor,
                        min_delta=0.00001,
                        patience=patience,
                        verbose=verbose,
                        mode='min',
                        restore_best_weights=True
                    )
            callbacks.append(earlystopping_cb)

        if include_reducelr:
            # Define the ReduceLROnPlateau callback
            reduce_lr_cb = ReduceLROnPlateau(
                monitor=reducelr_monitor,  # The metric to monitor
                factor=lr_factor,               # Factor by which the learning rate will be reduced
                patience=patience,               # Number of epochs with no improvement after which learning rate will be reduced
                min_lr=1e-6,              # Lower bound on the learning rate
                verbose=verbose                 # Display messages when the learning rate is reduced
            )
            callbacks.append(reduce_lr_cb)

        if include_display:
            if data is None:
                raise ValueError("Dataset for callbacks cannot be None for display callback")

            batch = next(iter(data))

            # Create display callbacks
            display_cb = DisplayOutputs(
                batch, self.tokenizer.reverse_token_map, 
                bos_token_idx = self.tokenizer.bos_token_idx, 
                eos_token_idx = self.tokenizer.eos_token_idx, 
                pad_token = self.tokenizer.pad_token, 
                target_maxlen = self.tokenizer.target_maxlen, 
                space_token=self.tokenizer.space_token,
                tggmt=self.is_tggmt,
                logger=self.logger
            )  # set the arguments as per vocabulary index for '<' and '>'
        
            callbacks.append(display_cb)

        if include_checkpoint:
            checkpoint_path = f"{self.output_path}/model.weights.h5" 
            # Define the ModelCheckpoint callback
            model_checkpoint_cb = ModelCheckpoint(
                filepath=checkpoint_path,  # Where to save the model
                save_weights_only=True,  # If True, only the weights will be saved
                monitor='val_loss',  # Metric to monitor
                save_best_only=True,  # Save only the best model
                mode='min',  # Save the model with the minimum validation loss
                verbose=1  # Verbosity mode
            )
            callbacks.append(model_checkpoint_cb)

        if include_savemodel:
            if self.model_saver is not None:
                callbacks.append(self.model_saver)        

        if include_augmentation:
            if pipeline is not None:
                callbacks.append(AugmentationCallback(pipeline, patience))
            
        return callbacks

class DisplayOutputs(tf.keras.callbacks.Callback):
    def __init__(
        self, 
        batch, 
        reverse_token_map, 
        bos_token_idx=60, 
        eos_token_idx=61, 
        pad_token = "P", 
        target_maxlen = 64, 
        space_token=95,
        tggmt=False,
        logger=None
    ):
        """Displays a batch of outputs after every 4 epoch

        Args:
            batch: A test batch
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            bos_token_idx: A start token index in the target vocabulary
            eos_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.bos_token_idx = bos_token_idx
        self.eos_token_idx = eos_token_idx
        self.pad_token = pad_token
        self.target_maxlen = target_maxlen
        self.space_token = space_token
        self.reverse_token_map = reverse_token_map
        self.tggmt = tggmt
        self.logger = logger

    def clean_text(self, text, space_token=" "):
        # Remove the specified space_token
        text = text.replace(space_token, " ")
        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)
        # Strip leading and trailing spaces
        return text.replace(f"{self.pad_token}", "").strip()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 4 != 0:
            return
        source = self.batch[0]
        target = self.batch[1].numpy()
        bs = tf.shape(source)[0]

        print(f"DisplayOutputs::Calling predict_source with source shape: {source.shape}, bos_token_idx: {self.bos_token_idx}")

        # Ensure we're in the right context
        strategy = tf.distribute.get_strategy()

        # Check if we're in a replica context or cross-replica context
        replica_context = tf.distribute.get_replica_context()
        bos_token_idx = self.bos_token_idx

        if replica_context is None:  # Check if we're in a cross-replica context
            # outputs = strategy.run(predict_fn)
            # We're in cross-replica context, so we can safely call strategy.run
            if hasattr(self.model, 'gest_transformer') : # TGGMT vs GMT
                step_fn = lambda args: self.model.gest_transformer.predict_source(args, bos_token_idx)
                
            else:
                step_fn = lambda args: self.model.predict_source(args, bos_token_idx)
            per_replica_result = strategy.run(step_fn, args=(source,))
            preds = strategy.gather(per_replica_result, axis=0)
        else:
            if hasattr(self.model, 'gest_transformer') :  # TGGMT vs GMT
                preds = self.model.gest_transformer.predict_source(source, bos_token_idx) 
            else:
                preds = self.model.predict_source(source, bos_token_idx) 

        self.logger.debug("Callback preds : ", type(preds))
        self.logger.debug("Callback preds : ", preds.shape)

        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.reverse_token_map[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction +=self.reverse_token_map[idx]
                if idx == self.eos_token_idx:
                    break

            # Remove space tokens
            target_text = self.clean_text(target_text, space_token=self.space_token)
            prediction_str = self.clean_text(prediction, space_token=self.space_token)

            # Calculate score
            sim_score = similarity_str(list(prediction_str), list(target_text))
            dist_score = normalized_distance_str(list(prediction_str), list(target_text))

            self.logger.info(f"Epoch:         {epoch}")
            self.logger.info(f"Distance:      {dist_score}")
            self.logger.info(f"Similarity:      {sim_score}")
            self.logger.info(f"target:     {target_text}")
            self.logger.info(f"prediction: {prediction_str}\n")
    
class AugmentationCallback(tf.keras.callbacks.Callback):
    def __init__(self, pipeline, patience):
        super().__init__()
        self.pipeline = pipeline
        self.patience = patience
        self.wait = 0
        self.monitor = 'gest_edit_acc'
        self.best_val_monitor = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_monitor = logs.get(self.monitor) if not None else 0.0
        if val_monitor < self.best_val_monitor:
            self.best_val_monitor = val_monitor
            self.wait = 0
            print("Validation loss improved. Resetting patience.")
        else:
            self.wait += 1
            print(f"No improvement in validation loss. Patience: {self.wait}/{self.patience}")

        if self.wait >= self.patience:
            if self.pipeline.dam_level < 4:
                self.pipeline.dam_level += 1
                print(f"Patience exceeded. Increasing data augmentation to DAM level {self.pipeline.dam_level}.")
                self.pipeline.train_ds = self.pipeline.create_train_dataset(dam_level=self.pipeline.dam_level)
            else:
                print("Maximum DAM level reached. Continuing training with highest augmentation.")
            self.wait = 0
    
class ModelSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, gesture_transformer, gpt2_wrapper, gesture_weights_path, gpt2_model_path, monitor="val_loss", logger=None):
        """
        Callback to save the best model weights based on validation loss.

        Args:
            gesture_transformer: The gesture transformer model.
            gpt2_wrapper: The GPT-2 wrapper model.
            gesture_weights_path: Path to save gesture_transformer weights.
            gpt2_model_path: Path to save the GPT-2 wrapper model.
            logger: Logger to log save events.
            monitor: Metric to monitor for saving the best weights.
        """
        super().__init__()
        self.gesture_transformer = gesture_transformer
        self.gpt2_wrapper = gpt2_wrapper
        self.gesture_weights_path = gesture_weights_path
        self.gpt2_model_path = gpt2_model_path
        self.logger = logger
        self.monitor = monitor
        self.best_val_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        # Get the current validation loss
        current_val_loss = logs.get(self.monitor)
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss

            # Save gesture_transformer weights
            self.gesture_transformer.save_weights(self.gesture_weights_path)
            self.logger.info(f"Saved best gesture_transformer weights to {self.gesture_weights_path}")

            # Save gpt2_wrapper model
            self.gpt2_wrapper._save_model() # self.gpt2_model_path
            self.logger.info(f"Saved best gpt2_wrapper model with validation loss: {self.best_val_loss}")
