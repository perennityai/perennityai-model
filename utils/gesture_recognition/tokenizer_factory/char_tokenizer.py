
import os
import json
import tensorflow as tf
from .base_tokenizer import BaseTokenizer

class CharTokenizer(BaseTokenizer):
    def __init__(self, token_file=None, default_tokens=[], num_classes=92, target_maxlen=64):
        # Define character-specific tokens and pass them to the base class
        char_special_tokens = default_tokens or ["P", "<", ">", "-1", "_"]  # 'P' for pad, '<' for start, '>' for end, '`' for unknown
        super().__init__(token_file, default_tokens=char_special_tokens, num_classes=num_classes, target_maxlen=target_maxlen)

