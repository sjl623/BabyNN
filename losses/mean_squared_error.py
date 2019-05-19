import numpy


class Mean_squared_error():
    parent_model = ""

    def __init__(self, parent_model):
        self.parent_model = parent_model

    def calculate(self):
        #print(numpy.square(self.parent_model.now_batch["output"] - self.parent_model.now_model[-1].output))
        tmp=numpy.sum(numpy.square(self.parent_model.now_batch["output"] - self.parent_model.now_model[-1].output))
        return numpy.sum(numpy.mean((numpy.square(self.parent_model.now_batch["output"] - self.parent_model.now_model[-1].output)),
                          axis=0))

    def getz(self):
        return self.parent_model.now_model[-1].output-self.parent_model.now_batch["output"]
        #return self.parent_model.now_batch["output"]-self.parent_model.now_model[-1].output
