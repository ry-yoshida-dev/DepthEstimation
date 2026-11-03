from enum import StrEnum


class DepthKind(StrEnum):
    """
    What the values of a depth map mean.

    Attributes
    ----------
    METRIC : str
        Distance in meters; larger is farther.
    RELATIVE_INVERSE : str
        Inverse depth (disparity) known only up to an unknown scale and shift; larger is nearer, and values of
        different images cannot be compared.
    """

    METRIC = "metric"
    RELATIVE_INVERSE = "relative_inverse"
