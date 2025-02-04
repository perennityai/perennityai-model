from .char_tokenizer import  CharTokenizer 
from .word_tokenizer import WordTokenizer 
from utils.common.json_handler import JSONHandler

class TokenizerFactory:

    # A mapping of domain names to their corresponding processor classes
    # 'Tokenizer' has a defined class, while other domains are placeholders
    _domain_map = {
        'char_level': CharTokenizer,
        'word_level': WordTokenizer
    }

    @classmethod
    def create_tokenizer(cls, config):
        """Creates and configures a tokenizer instance based on the specified domain.
        
        Args:
            config (dict): Configuration dictionary containing:
                - 'domain' (str): The domain type for the tokenizer.

        Returns:
            An instance of the tokenizer class for the specified domain, configured according to the provided config.

        Raises:
            ValueError: If the specified domain is not in the _domain_map.
        """
        
        # Retrieve the domain and optional configuration details from config
        _level = config.get("token_level")

        # Check if the domain has a corresponding tokenizer class in the _domain_map
        if _level not in cls._domain_map:
            raise ValueError(f"Unknown domain: {_level}")

        # Retrieve the tokenizer class for the domain
        tokenizer_class = cls._domain_map[_level]
        
        # Create a tokenizer instance using a domain-specific method (e.g., from_pretrained)
        tokenizer = tokenizer_class.from_config(config)   
        if tokenizer is None:
            raise ValueError(f"Tokenizer is None, config is {config}")     
        
        return tokenizer  # Return the configured tokenizer instance