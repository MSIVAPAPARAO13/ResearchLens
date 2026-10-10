"""File system and safe temporary file utilities."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union


def validate_file_extension(file_path_or_name: Union[str, Path], allowed_extensions: list[str]) -> bool:
    """Validate that the given filename or path has an allowed extension."""
    suffix = Path(file_path_or_name).suffix.lower()
    return suffix in [ext.lower() for ext in allowed_extensions]


@contextmanager
def safe_temp_pdf(content: bytes) -> Generator[str, None, None]:
    """Context manager to securely write bytes to a temp file and guarantee cleanup."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()
        yield temp_file.name
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except OSError:
                pass
