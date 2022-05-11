# BootStraps class, @Author Michael Miller 2205885M
# class allows creation of bootstrap objects used for calculation of bootstrap samples used in experiments


class BootStraps:
    # Constructor size: numeric value correspondng to its percentage, i.e. 70 = 70% sample
    #             bootstrap_percentage: adjusted percentage so that the correct size resample
    #                                   may be taken from previous sample
    def __init__(self, size, bootstrap_percentage):
        self.size = size
        self.sample = None
        self.ratio_split = None
        self.bootstrap_percentage = bootstrap_percentage

    # return string representation of size for use in file writing
    def get_name(self):
        return str(self.size) + "%"

    # return numeric value of size for use in calculations
    def get_size(self):
        return self.size

    # return data-sample associated with this object
    def get_sample(self):
        return self.sample

    # update data-sample associated with this object
    def set_sample(self, sample_new):
        self.sample = sample_new

    # return bootstrap_percentage associated with this sample for use in calculation
    def get_bootstrap_percentage(self):
        return self.bootstrap_percentage



