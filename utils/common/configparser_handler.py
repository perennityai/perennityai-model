import configparser
from pathlib import Path

class ConfigParserHandler:
    def __init__(self, filename):
        """
        Initializes the ConfigParserHandler with the specified configuration file.
        
        Parameters:
            filename (str): The path to the configuration file. Defaults to "config.ini".
        """
        self.config = configparser.ConfigParser()
        self.file_path = Path(filename)
        
        # Load existing configuration if file exists
        if self.file_path.exists():
            self.load()

    def load(self):
        """Loads the configuration from the file."""
        self.config.read(self.file_path)

    def get(self, section, option, fallback=None):
        """
        Retrieves a value from the configuration.
        
        Parameters:
            section (str): The section of the configuration.
            option (str): The option within the section to retrieve.
            fallback: The fallback value if the option is not found. Defaults to None.
        
        Returns:
            str: The value from the configuration, or the fallback if not found.
        
        Raises:
            ValueError: If the specified section does not exist in the configuration.
        """
        if not self.config.has_section(section):
            raise ValueError(f"The section '{section}' does not exist in the configuration {self.config.sections()}.")
        return self.config.get(section, option, fallback=fallback)

    def get_int(self, section, option, fallback=None):
        """Retrieves an integer value from the configuration."""
        return self.config.getint(section, option, fallback=fallback)

    def get_float(self, section, option, fallback=None):
        """Retrieves a float value from the configuration."""
        return self.config.getfloat(section, option, fallback=fallback)

    def get_boolean(self, section, option, fallback=None):
        """Retrieves a boolean value from the configuration."""
        return self.config.getboolean(section, option, fallback=fallback)

    def set(self, section, option, value):
        """
        Sets a value in the configuration.
        
        Parameters:
            section (str): The section of the configuration.
            option (str): The option within the section to set.
            value (str): The value to set.
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, str(value))

    def save(self):
        """Saves the current configuration to the file."""
        with self.file_path.open("w") as configfile:
            self.config.write(configfile)

    def remove_section(self, section):
        """
        Removes a section from the configuration.
        
        Parameters:
            section (str): The section to remove.
        """
        if self.config.has_section(section):
            self.config.remove_section(section)

    def remove_option(self, section, option):
        """
        Removes an option from a section in the configuration.
        
        Parameters:
            section (str): The section of the configuration.
            option (str): The option within the section to remove.
        """
        if self.config.has_section(section):
            self.config.remove_option(section, option)
