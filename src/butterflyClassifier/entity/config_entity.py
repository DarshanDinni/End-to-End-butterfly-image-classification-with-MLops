from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Data class representing the configuration for data ingestion.

    Attributes:
        root_dir (Path): The root directory where data will be stored.
        source_url (str): The URL from which the data will be downloaded.
        local_data_file (Path): The local path where the downloaded data will be stored.
    """
    root_dir: Path
    source_url: str
    local_data_file: Path
