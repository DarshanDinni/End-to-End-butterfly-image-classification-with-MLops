import sys


# Function to create a custom error message
def create_custom_error_message(error, error_details: sys):
    # Get information about the traceback
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_frame.f_lineno

    # Construct a custom error message
    custom_message = f"The error occurred in [{file_name}], at line number [{line_number}], with error message: {str(error)}"
    return custom_message


class CustomException(Exception):
    def __init__(self, error, error_details: sys):
        # Initialize the base class with the error message
        super().__init__(str(error))

        # Create a custom error message using the provided details
        self.error_message = create_custom_error_message(error, error_details)

    def __str__(self) -> str:
        # Override the __str__ method to return the custom error message
        return self.error_message
