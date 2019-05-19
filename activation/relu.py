import activation
import numpy


class Relu(activation.Base):
    def forward(self):
        return numpy.maximum(0, self.now_layer.output)

    def backward(self):
        self.now_layer.parent_model.now_model[self.now_layer.layerID + 1].d = numpy.where(
            numpy.greater(self.now_layer.before_activation, 0),
            self.now_layer.parent_model.now_model[self.now_layer.layerID + 1].d, 0)
