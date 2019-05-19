import numpy
import layer


class Dense(layer.Base):
    size = 0
    input_dim = 0
    activation = ""
    before_activation = ""
    w = ""
    d = ""

    def __init__(self, size, input_dim=0, activation=None):
        self.size = size
        self.input_dim = input_dim
        if activation:
            module = __import__("activation")
            activation_class = getattr(module, activation)
            self.activation = activation_class(self)

    def init(self):
        if self.layerID == 0:
            self.w = 0.01 * numpy.random.randn(self.input_dim, self.size)
        else:
            self.w = 0.01 * numpy.random.randn(self.parent_model.now_model[self.layerID - 1].get_output_size(),
                                               self.size)

    def forward(self, first_input):
        if self.layerID == 0:
            self.output = numpy.dot(first_input, self.w)
        else:
            self.output = numpy.dot(self.parent_model.now_model[self.layerID - 1].output, self.w)
        if self.activation:
            self.before_activation = self.output
            self.output = self.activation.forward()

    def backward(self):
        if self.layerID == len(self.parent_model.now_model) - 1:
            self.d = numpy.dot(self.parent_model.loss_function.getz(), self.w.T)
            if self.layerID == 0:
                last_output = self.parent_model.now_batch["input"]
            else:
                last_output = self.parent_model.now_model[self.layerID - 1].output

            dw = numpy.dot(last_output.T, self.parent_model.loss_function.getz())
            # dw = self.output - self.parent_model.loss_function.getz()
            dz = dw / self.output.shape[0]
        else:
            self.d = numpy.dot(self.parent_model.now_model[self.layerID + 1].d, self.w.T)
            if self.layerID == 0:
                last_output = self.parent_model.now_batch["input"]
            else:
                last_output = self.parent_model.now_model[self.layerID - 1].output
            dw = numpy.dot(last_output.T, self.parent_model.now_model[self.layerID + 1].d)
            # dw = self.output - self.parent_model[self.layerID + 1].d
            dz = dw / self.output.shape[0]
        if self.activation:
            self.activation.backward()
        return dz

    def get_output_size(self):
        return self.size
