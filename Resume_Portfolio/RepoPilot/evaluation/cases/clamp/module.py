def clamp(value: int, lower: int, upper: int) -> int:
    return min(lower, max(value, upper))
