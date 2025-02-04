import os
import json
import tensorflow as tf
from .base_tokenizer import BaseTokenizer

class WordTokenizer(BaseTokenizer):
    def __init__(self, token_file=None, default_tokens=[], num_classes=92, target_maxlen=64):
        # Define word-specific tokens and pass them to the base class
        word_special_tokens = default_tokens or ["<pad>", "<start>", "<end>", "<unk>", "<space>"]

        super().__init__(token_file, default_tokens=word_special_tokens, num_classes=num_classes, target_maxlen=target_maxlen)
        self.space_token = "<space>"
        self.space_token_idx = self.token_map[self.space_token]
