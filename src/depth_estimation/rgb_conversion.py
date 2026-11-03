from typing import ClassVar

from PIL import Image


class RgbConversion:
    """
    Turns images of any 8-bit mode into RGB, the input of every model.
    """

    HIGH_BIT_DEPTH_MODES: ClassVar[frozenset[str]] = frozenset({"I", "F", "I;16", "I;16L", "I;16B", "I;16N"})
    ALPHA_MODES: ClassVar[frozenset[str]] = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})
    BACKGROUND: ClassVar[tuple[int, int, int, int]] = (255, 255, 255, 255)

    @classmethod
    def convert(cls, image: Image.Image) -> Image.Image:
        """
        Convert an image to RGB, compositing transparency over white.

        Colors under fully transparent pixels are arbitrary, so they must not reach the model.

        Parameters
        ----------
        image : Image.Image
            Image of any 8-bit mode.

        Returns
        -------
        Image.Image
            RGB image; the input itself when it is already RGB without transparency.

        Raises
        ------
        ValueError
            If the image has a high bit depth, which PIL would clip instead of scale.
        """
        if image.mode in cls.HIGH_BIT_DEPTH_MODES:
            raise ValueError(f"high-bit-depth images are not supported; scale them to 8 bits. got mode {image.mode}")
        if image.mode in cls.ALPHA_MODES or "transparency" in image.info:
            rgba: Image.Image = image.convert("RGBA")
            return Image.alpha_composite(Image.new("RGBA", rgba.size, cls.BACKGROUND), rgba).convert("RGB")
        return image if image.mode == "RGB" else image.convert("RGB")
