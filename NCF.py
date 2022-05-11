from RecModel import RecModel

# NCF class @Author Michael Miller 2205885M
# class houses input parameters for neural collaborative filtering models


class NCF(RecModel):
    def __init__(self, name, num_factors, layers, act_fn, learner, num_epochs, batch_size, learn_rate, num_neg):
        self.name = name
        self.num_factors = num_factors
        self.layers = layers
        self.act_fn = act_fn
        self.learner = learner
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.num_neg = num_neg
        self.model_type = 3

    # return model param differing from default configuration as "name"
    def get_name(self):
        return self.name

    # return num_factors input parameter
    def get_k(self):
        return self.num_factors

    # return layers input parameter
    def get_layers(self):
        return self.layers

    # return activation function input parameter
    def get_act_fn(self):
        return self.act_fn

    # return learning rate function input parameter
    def get_learner(self):
        return self.learner

    # return number of epochs input parameter
    def get_max_iter(self):
        return self.num_epochs

    # return batch size input parameter
    def get_batch_size(self):
        return self.batch_size

    # return learning rate inut parameter
    def get_learning_rate(self):
        return self.learn_rate

    # return negative pairing number input parameter
    def get_num_neg(self):
        return self.num_neg

    # return model_type for distinguishing between models
    def get_model_type(self):
        return self.model_type

    def get_lambda_reg(self):
        pass

    def get_use_bias(self):
        pass

    def get_early_stop(self):
        pass

    def get_verbose(self):
        pass







