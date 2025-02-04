import json
from typing import Any, Dict

class JSONHandler:
    """A utility class for handling JSON files without requiring instantiation."""

    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        """Reads JSON data from a file and returns it as a dictionary.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Raises:
            FileNotFoundError: If the JSON file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
        
        Returns:
            dict: The JSON data as a dictionary.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, str):
                    data = json.loads(data)
                return data
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
            raise
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {file_path}.")
            raise

    @staticmethod
    def write_json(file_path: str, data: Dict[str, Any]) -> None:
        """Writes a dictionary to a JSON file.
        
        Args:
            file_path (str): Path to the JSON file.
            data (dict): The dictionary to be written to the JSON file.
        
        Raises:
            TypeError: If the data provided is not serializable to JSON.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            print(f"Data successfully written to {file_path}")
        except TypeError:
            print("Error: Provided data is not serializable to JSON.")
            raise

    @staticmethod
    def update_json(file_path: str, new_data: Dict[str, Any]) -> None:
        """Updates the JSON file with new data, merging it with existing content.
        
        Args:
            file_path (str): Path to the JSON file.
            new_data (dict): The dictionary with new data to add or update.
        
        Raises:
            json.JSONDecodeError: If the existing file content is not valid JSON.
            FileNotFoundError: If the JSON file does not exist.
        """
        try:
            data = JSONHandler.read_json(file_path)
            data.update(new_data)
            JSONHandler.write_json(file_path, data)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Error: Unable to update JSON file.")
            raise

