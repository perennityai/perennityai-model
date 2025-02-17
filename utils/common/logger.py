import tensorflow as tf
import torch
import datetime
import os

class Log:
    LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

    def __init__(self, log_file=None, name="LOG::", verbose="INFO"):
        self.log_file = log_file
        self.name = ""
        self.set_logname(name)
        self.verbose = self.LEVELS.get(verbose.upper(), 20)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def _log(self, level, *args):
        """
        Log a message with specified severity level, handling lists and tensors in *args.

        Parameters:
        - level (str): The severity level of the log (e.g., DEBUG, INFO, WARNING).
        - *args: Additional arguments for the log message, which can include lists, tf.Tensors, or torch.Tensors.
        """

        # Function to convert various data types to strings, handling lists and tensors recursively
        def to_string(arg):
            if isinstance(arg, list):
                return '[' + ', '.join(to_string(item) for item in arg) + ']'
            elif isinstance(arg, tf.Tensor):
                if tf.executing_eagerly():
                    return str(arg.numpy())
                else:
                    return str(arg)
            elif isinstance(arg, torch.Tensor):
                return str(arg.detach().cpu().numpy())
            return str(arg)

        # Construct the log message by formatting all arguments
        message = ' '.join(to_string(arg) for arg in args)

        # Check if log level meets the minimum verbose level
        if self.LEVELS.get(level, 20) >= self.verbose:
            # Format timestamp and log level
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] [{self.name + level}] {message}"

            # Print to console
            print(log_message)

            # Write to file if log_file is set
            if self.log_file:
                with open(self.log_file, "a") as file:
                    file.write(log_message + "\n")

    def set_logname(self, classobj):
        # Check if classobj is a string
        if isinstance(classobj, str):
            if "::" in classobj:
                self.name = classobj
            else:
                self.name = classobj + "::"
        # Check if classobj is an object with __name__ attribute
        elif hasattr(classobj, "__name__"):
            if "::" in classobj.__name__:
                self.name = classobj.__name__
            else:
                self.name = classobj.__name__ + "::"
        else:
            raise TypeError("classobj must be either a string or an object with a __name__ attribute")
        return self


    def debug(self, *args):
        """Log a debug message."""
        self._log(f"DEBUG", *args)

    def info(self, *args):
        """Log an informational message."""
        self._log(f"INFO", *args)

    def warning(self, *args):
        """Log a warning message."""
        self._log(f"WARNING", *args)

    def error(self, *args):
        """Log an error message."""
        self._log(f"ERROR", *args)

    def critical(self, *args):
        """Log a critical error message."""
        self._log(f"CRITICAL", *args)



# Log messages with different levels
# logger.debug("Debugging info")        # Will not log (below WARNING)
# logger.info("General information")    # Will not log (below WARNING)
# logger.warning("This is a warning")   # Will log
# logger.error("This is an error")      # Will log
# logger.critical("Critical issue!")    # Will log