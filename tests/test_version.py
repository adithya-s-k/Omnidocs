"""Test omnidocs version."""

import omnidocs


def test_version_exists():
    assert hasattr(omnidocs, "__version__")


def test_version_format():
    version = omnidocs.__version__
    assert isinstance(version, str)
    assert len(version.split(".")) >= 2
