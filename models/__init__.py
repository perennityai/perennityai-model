# models/__init__.py

"""
Module Name: models
Author: PerennityAI
Date: 2024-11-03
Revision: 1.0.0

Description:
This module contains the definition of models.

"""

# Package-level variables
__version__ = "1.0.0"

# Import all functions
from .model_factory import TransformerModelFactory, TransformerTFLiteModelFactory, Seq2SeqTFLiteModelFactory, TFLiteSeq2SeqInference
from models.gesture_transformer import TransformerHyperModel
from models.gesture_seq2seq import SimpleSeq2Seq
from models.gesture_seq2seq import NoAttenSeq2SeqTranslator
from models.gesture_seq2seq import AttenSeq2SeqTranslator
from models.gesture_seq2seq import Seq2SeqHyperModel

from  models.gesture_gmt import TGGMT
from models.gpt_model import GPT2Model
from .rl_env_search import TGGMTSearchEnv

# public classes that are available at the sub-package level
__all__ = ['TransformerModelFactory', 
           'TransformerTFLiteModelFactory',
           'Seq2SeqTFLiteModelFactory',
           'TransformerHyperModel',
           'TFLiteSeq2SeqInference',
           'SimpleSeq2Seq',
           'NoAttenSeq2SeqTranslator',
           'AttenSeq2SeqTranslator',
           'Seq2SeqHyperModel',
           'TGGMT',
           'GPT2Model',
           'TGGMTSearchEnv'
           ]