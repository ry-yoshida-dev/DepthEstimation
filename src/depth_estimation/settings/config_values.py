from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ConfigValues:
    """
    Typed, validating access to the plain values of a configuration section, e.g. parsed YAML.

    Every accessor raises ``KeyError`` for a missing required key and ``TypeError`` for a value of the wrong type, so
    a malformed configuration fails before a model is loaded.

    Attributes
    ----------
    values : Mapping[str, object]
        Values keyed by field name.
    """

    values: Mapping[str, object]

    def validate_keys(self, known_keys: frozenset[str]) -> None:
        """
        Reject keys that no field reads, e.g. a misspelled field name.

        Parameters
        ----------
        known_keys : frozenset[str]
            Keys that may be given.

        Raises
        ------
        KeyError
            If a key is not one of ``known_keys``.
        """
        unknown_keys: list[str] = sorted(set(self.values) - known_keys)
        if unknown_keys:
            raise KeyError(f"unknown configuration keys {unknown_keys}. known: {sorted(known_keys)}")

    def text(self, key: str) -> str:
        """
        Read a required string.

        Parameters
        ----------
        key : str
            Field name.

        Returns
        -------
        str
            Value of the field.

        Raises
        ------
        KeyError
            If the key is missing.
        TypeError
            If the value is not a string.
        """
        value: object = self._required(key)
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a string. got {type(value).__name__}")
        return value

    def integer(self, key: str, default: int) -> int:
        """
        Read an optional integer.

        Parameters
        ----------
        key : str
            Field name.
        default : int
            Value when the key is missing.

        Returns
        -------
        int
            Value of the field.

        Raises
        ------
        TypeError
            If the value is not an integer.
        """
        value: object = self.values.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{key} must be an integer. got {type(value).__name__}")
        return value

    def member[EnumT: Enum](self, key: str, enum_type: type[EnumT]) -> EnumT:
        """
        Read a required enum member given by value.

        Parameters
        ----------
        key : str
            Field name.
        enum_type : type[EnumT]
            Enum the value belongs to.

        Returns
        -------
        EnumT
            Member with the given value.

        Raises
        ------
        KeyError
            If the key is missing.
        ValueError
            If no member has the value.
        """
        return enum_type(self._required(key))

    def optional_member[EnumT: Enum](self, key: str, enum_type: type[EnumT], default: EnumT) -> EnumT:
        """
        Read an optional enum member given by value.

        Parameters
        ----------
        key : str
            Field name.
        enum_type : type[EnumT]
            Enum the value belongs to.
        default : EnumT
            Member when the key is missing.

        Returns
        -------
        EnumT
            Member with the given value.

        Raises
        ------
        ValueError
            If no member has the value.
        """
        return enum_type(self.values[key]) if key in self.values else default

    def _required(self, key: str) -> object:
        if key not in self.values:
            raise KeyError(f"missing configuration key {key!r}.")
        return self.values[key]
