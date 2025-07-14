import toml
import msapy


def test_version():
    pyproject = toml.load("pyproject.toml")
    assert msapy.__version__ == pyproject["project"]["version"]