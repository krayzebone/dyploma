import sys
import os

def resource_path(relative_path):
    """
    Get the absolute path to a resource. Works for dev (normal) and PyInstaller .exe.
    """
    try:
        # When running in a PyInstaller bundle, _MEIPASS is the temp folder.
        base_path = sys._MEIPASS
    except AttributeError:
        # If not running as a bundle, just use the current directory.
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
