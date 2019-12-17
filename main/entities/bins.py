import numpy as np


class Bins:
    def __init__(self, bin_boundaries: np.ndarray):
        assert (np.all(bin_boundaries[:-1] <= bin_boundaries[1:]))  # are boundaries sorted?
        self.bin_boundaries = bin_boundaries

    def get_num_of_bins(self):
        return len(self.bin_boundaries) - 1
