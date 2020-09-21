def sum_vec(x: list = [1, 2], y: list = [3, 4]) -> list:
    """
    A function to sum two vectors having the same length in a list format
    :param x: a list of numbers
    :param y: a list of number
    :return: a summed vector in a list format
    """
    if len(x) != len(y):
        raise ValueError("ERROR: Both vectors must have the same length")
    else:
        z = [xi+yi for xi, yi in zip(x, y)]
    return z


cname = "CDS"
