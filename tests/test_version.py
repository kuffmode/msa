import toml
import msapy

def test_version():
    assert msapy.__version__ == toml.load("pyproject.toml")["tool"]["poetry"]["version"]