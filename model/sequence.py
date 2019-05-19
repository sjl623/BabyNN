import model
import time


class Sequence(model.Base):
    def __init__(self):
        pass

    def add(self, new_layer, activation=None):
        new_layer.parent_model = self
        new_layer.layerID = len(self.now_model)
        self.now_model.append(new_layer)

    def compile(self, optimizer="", loss=""):
        module = __import__("losses")
        losses_class = getattr(module, loss)
        self.loss_function = losses_class(self)
        self.optimizer = optimizer

    def predict(self, first_input):
        for now_layer in self.now_model:
            now_layer.forward(first_input)
        return self.now_model[-1].output

    def train(self):
        for count in range(0, self.epoch):
            if count % 50 == 0 and count != 0:
                time.sleep(60)
            for i in range(0, self.batch_num):
                self.now_batch = self.get_next_batch()
                self.predict(self.now_batch["input"])
                loss = self.loss_function.calculate()
                if i == self.batch_num - 1:
                    print(count, loss)
                self.optimizer.optimize()
