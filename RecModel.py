from abc import ABC, abstractmethod
# RecModel class @Author Michael Miller 2205885M
# class serves as a template to abstract common functions of all model subclasses,
# class specific implementation details of each method can be found within their respective class

class RecModel(ABC):
    @abstractmethod
    def get_name(self):
        pass
    @abstractmethod
    def get_k(self):
        pass
    @abstractmethod
    def get_max_iter(self):
        pass
    @abstractmethod
    def get_learning_rate(self):
        pass
    @abstractmethod
    def get_lambda_reg(self):
        pass
    @abstractmethod
    def get_use_bias(self):
        pass
    @abstractmethod
    def get_early_stop(self):
        pass
    @abstractmethod
    def get_verbose(self):
        pass
    @abstractmethod
    def get_layers(self):
        pass
    @abstractmethod
    def get_act_fn(self):
        pass
    @abstractmethod
    def get_learner(self):
        pass
    @abstractmethod
    def get_batch_size(self):
        pass
    @abstractmethod
    def get_num_neg(self):
        pass
    @abstractmethod
    def get_model_type(self):
        pass
