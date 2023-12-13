"""Functions to handle data arrays

"""

import numpy as np

def round_nonzero_decimal(num, precision=1, method="round"):
    """
    Round an input decimal number starting from its first non-zero value.

    For instance, with precision of 1, we have:
    0.09324 -> 0.09
    0.00246 -> 0.002

    Parameters
    ----------
    num : float
        Float number.
    precision : int
        Number of decimals to keep. Defaults to 1.
    method : str
        Method for rounding a number. Currently supports
        np.round(), np.floor(), and np.ceil().

    Returns
    -------
    round_num : float
        Rounded number.
    """
    # Validation
    if num > 1:
        raise ValueError("num should be less than 1.")
    if num == 0: return 0
    
    # Identify the number of zero decimals
    decimals = int(abs(np.floor(np.log10(abs(num)))))
    precision = decimals + precision - 1
    
    # Round decimal number
    if method == "round":
        round_num = np.round(num, precision)
    elif method == "floor":
        round_num = np.true_divide(np.floor(num * 10 ** precision), 10 ** precision)
    elif method == "ceil":
        round_num = np.true_divide(np.ceil(num * 10 ** precision), 10 ** precision)
    
    return round_num

def split_half(array):
    """
    Split an array into half after a random shuffle. The random 
    shuffling uses the Fisher-Yates shuffle based on a uniform 
    distribution.

    Parameters
    ----------
    array : np.ndarray
        A data array to be split.

    Returns
    -------
    first_half : np.ndarray
        A first half of the split data array.
    second_half: np.ndarray
        A second half of the split data array.
    """

    # Split an array into half
    np.random.shuffle(array)
    first_half = array[:len(array) // 2]
    second_half = array[len(array) // 2:]
    
    return first_half, second_half