# RandomSeed class @Author Michael Miller 2205885M
# Class houses random seed objects, only function to store randomly generated integers for use in experiments

class RandomSeed:
    # constructor takes only single int as an argument
    def __init__(self, size):
        self.size = size

    # return int value of its size for use in experiment
    def get_size(self):
        return self.size

    # return string representation of size for use in file writing
    def get_string(self):
        return str(self.size)
