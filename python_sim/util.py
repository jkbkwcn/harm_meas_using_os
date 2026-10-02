import numpy as np


def round_sig(x: float, sig: int = 6):
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)


def si_formatter(value: float, unit: str, sig_digits=4, bracket: str = "(") -> str:
    """Format a number using SI units and prefixes.

    Examples:
        >>> si_formatter(1234, "m")
        '1.23 (km)'
        >>> si_formatter(0.001234, "m")
        '1.23 (mm)'

    Args:
        value (float): The value to format.
        unit (str): The unit of the value.
        precision (int, optional): The number of decimal places to include. Defaults to 2.
        bracket (str, optional): The type of bracket to use. Defaults to "(".

    Returns:
        str: The formatted value.
    """
    prefix, mult = get_prefix_multiplier(value)

    closing = {
        "(": ")",
        "[": "]",
        "{": "}",
    }

    value_scaled = value * mult

    return f"{round_sig(value_scaled, sig_digits)}" + " " + bracket + prefix + unit + closing[bracket]


def get_prefix_multiplier(value: float) -> tuple[str, float]:
    """Get the SI prefix and multiplier for a given value.

    Args:
        value (float): The value to get the prefix and multiplier for.

    Raises:
        ValueError: If the value is not a number.

    Returns:
        tuple[str, float]: The SI prefix and multiplier.
    """
    prefixes = {
        15: "P",  # peta
        12: "T",  # tera
        9: "G",  # giga
        6: "M",  # mega
        3: "k",  # kilo
        0: "",  # no prefix
        -3: "m",  # milli
        -6: "u",  # micro
        -9: "n",  # nano
        -12: "p",  # pico
        -15: "f",  # femto
    }

    if not isinstance(value, (int, float)):
        raise ValueError(f"Value must be a number! Got: '{value}'")

    exponent = int("{:.0e}".format(value).split("e")[1])
    exponent = 3 * (exponent // 3)

    exponent = max(min(exponent, 15), -15)
    multiplier = 1 / (10**exponent)

    prefix = prefixes[exponent]

    return prefix, multiplier
