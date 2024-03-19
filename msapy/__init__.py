from msapy import msa, utils, checks
import toml
import os

package_dir = os.path.abspath(os.path.dirname(__file__))
pyproject_path = os.path.join(package_dir, "..", "pyproject.toml")

try:
    __version__ = toml.load(pyproject_path)["tool"]["poetry"]["version"]
except Exception as e:
    __version__ = "unknown"
    print(f"Warning: Could not load version from pyproject.toml: {e}")