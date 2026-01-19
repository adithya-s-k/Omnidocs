"""Test omnidocs version management."""

import omnidocs
from omnidocs import __version__
from omnidocs._version import __version__ as version_from_file


def test_version_exists():
    """Test that version attribute exists."""
    assert hasattr(omnidocs, "__version__")


def test_version_format():
    """Test version follows semantic versioning format."""
    version = omnidocs.__version__
    assert isinstance(version, str)

    # Should have at least major.minor format
    parts = version.split(".")
    assert len(parts) >= 2, f"Version should have at least major.minor: {version}"

    # First two parts should be integers
    major, minor = parts[0], parts[1]
    assert major.isdigit(), f"Major version should be a number: {major}"
    assert minor.isdigit(), f"Minor version should be a number: {minor}"


def test_version_consistency():
    """Test that version is consistent across imports."""
    # All these should be the same
    import omnidocs as od
    from omnidocs import __version__ as v1
    from omnidocs._version import __version__ as v2

    assert v1 == v2, "Version mismatch between omnidocs and omnidocs._version"
    assert v1 == od.__version__, "Version mismatch in omnidocs module"


def test_version_accessible():
    """Test version is accessible from package."""
    # Should work from main import
    assert __version__ is not None
    assert len(__version__) > 0

    # Should work from _version module
    assert version_from_file is not None
    assert __version__ == version_from_file


def test_current_version():
    """Test current version is 0.2.0 or higher."""
    version = omnidocs.__version__

    # Parse major.minor
    parts = version.split(".")
    major = int(parts[0])
    minor = int(parts[1])

    # Should be at least 0.2.x
    assert major == 0, f"Expected major version 0, got {major}"
    assert minor >= 2, f"Expected minor version >= 2, got {minor}"
