import pytest
from pydantic import ValidationError

from segpaste.config import CopyPasteConfig


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_paste_probability(self) -> None:
        """Test invalid paste probability raises ValidationError."""
        with pytest.raises(ValidationError):
            CopyPasteConfig(paste_probability=1.5)

        with pytest.raises(ValidationError):
            CopyPasteConfig(paste_probability=-0.1)

    def test_invalid_paste_objects_count(self) -> None:
        """Test invalid paste objects count raises ValidationError."""
        # Min > Max
        with pytest.raises(ValidationError):
            CopyPasteConfig(min_paste_objects=5, max_paste_objects=4)

        # Negative values (caught by Field(ge=0))
        with pytest.raises(ValidationError):
            CopyPasteConfig(min_paste_objects=-1)

        with pytest.raises(ValidationError):
            CopyPasteConfig(max_paste_objects=-1)

    def test_invalid_scale_range(self) -> None:
        """Test invalid scale range raises ValidationError."""
        with pytest.raises(ValidationError):
            CopyPasteConfig(scale_range=(2.0, 0.5))

    def test_invalid_blend_mode(self) -> None:
        """Test invalid blend mode raises ValidationError."""
        with pytest.raises(ValidationError):
            CopyPasteConfig(blend_mode="invalid")

    def test_valid_config(self) -> None:
        """Test valid configuration."""
        config = CopyPasteConfig(
            paste_probability=0.8,
            min_paste_objects=2,
            max_paste_objects=5,
            scale_range=(0.8, 1.2),
            blend_mode="gaussian",
        )
        assert config.paste_probability == 0.8
        assert config.min_paste_objects == 2
        assert config.max_paste_objects == 5
        assert config.scale_range == (0.8, 1.2)
        assert config.blend_mode == "gaussian"

    def test_forbid_extra_fields(self) -> None:
        """Test that unknown arguments raise ValidationError."""
        with pytest.raises(ValidationError):
            CopyPasteConfig(unknown_field=123)  # type: ignore
