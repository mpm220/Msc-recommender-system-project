from RecModel import RecModel
# MF class @Author Michael Miller 2205885M
# class houses input parameters for matrix factorisation models


class MF(RecModel):
    def __init__(self, name, k, max_iter, learning_rate, lambda_reg, use_bias, early_stop, verbose):
        self.name = name
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.use_bias = use_bias
        self.early_stop = early_stop
        self.verbose = verbose
        self.model_type = 1

    # return model param differing from default configuration as "name"
    def get_name(self):
        return self.name

    # return k-models input parameter
    def get_k(self):
        return self.k

    # return max_iterations input parameter
    def get_max_iter(self):
        return self.max_iter

    # return learning rate input parameter
    def get_learning_rate(self):
        return self.learning_rate

    # return lambda regulation input parameter
    def get_lambda_reg(self):
        return self.lambda_reg

    # return use bias input parameter
    def get_use_bias(self):
        return self.use_bias

    # return early stop input parameter
    def get_early_stop(self):
        return self.early_stop

    # return verbose input parameter
    def get_verbose(self):
        return self.verbose

    # return model_type for distinguishing between models
    def get_model_type(self):
        return self.model_type

    def get_layers(self):
        pass

    def get_act_fn(self):
        pass

    def get_learner(self):
        pass

    def get_batch_size(self):
        pass

    def get_num_neg(self):
        pass
