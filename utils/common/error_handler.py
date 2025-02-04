from utils.common.logger import Log
import traceback
from typing import Optional, Callable, Any

class ErrorHandler:
    """A utility class for handling errors with options to log, retry, and display custom messages."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the error handler with a Log instance for error logging.
        
        Args:
            log_file (Optional[str]): Optional file path for logging errors. 
        """
        self.logger = Log(log_file or "error_log.txt")  # Use the Log class from logger module

    def handle_error(self, error: Exception, custom_message: Optional[str] = None) -> None:
        """Handles an error by logging it and optionally displaying a custom message.
        
        Args:
            error (Exception): The exception that occurred.
            custom_message (Optional[str]): A custom message to display along with the error.
        """
        error_message = f"Error: {str(error)}"
        detailed_message = f"{error_message}\nTraceback:\n{traceback.format_exc()}"

        # Log the error details using the external Log instance
        self.logger.error(detailed_message)

        # Display custom message if provided
        if custom_message:
            print(custom_message)
        else:
            print(error_message)

    def retry_on_error(self, func: Callable, retries: int = 3, *args, **kwargs) -> Optional[Any]:
        """Retries a function call if an error occurs, up to a specified number of attempts.
        
        Args:
            func (Callable): The function to call.
            retries (int): Number of retry attempts in case of failure.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            Optional[Any]: The return value of the function if successful, or None if all retries failed.
        """
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except Exception as error:
                attempts += 1
                self.handle_error(error, f"Attempt {attempts} failed. Retrying...")
        print(f"All {retries} attempts failed.")
        return None

    def raise_custom_error(self, message: str, error_type: Optional[Exception] = Exception) -> None:
        """Raises a custom error with a specified message and error type.
        
        Args:
            message (str): The error message to display.
            error_type (Optional[Exception]): The type of exception to raise.
        
        Raises:
            Exception: Raises the specified exception type with the given message.
        """
        self.logger.error(f"Raising custom error: {message}")
        raise error_type(message)

# Usage example
# if __name__ == "__main__":
#     # Initialize the error handler with a specified log file
#     error_handler = ErrorHandler("custom_error_log.txt")
    
#     # Example of handling an error
#     try:
#         1 / 0  # This will raise a ZeroDivisionError
#     except Exception as e:
#         error_handler.handle_error(e, custom_message="A division error occurred.")

#     # Example of retrying a function with error handling
#     def faulty_function():
#         raise ValueError("An intentional error.")

#     error_handler.retry_on_error(faulty_function, retries=2)
