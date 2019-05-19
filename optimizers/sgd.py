class SGD:
    speed = 0
    parent_model = ""

    def __init__(self, parent_model, speed):
        self.parent_model = parent_model
        self.speed = speed

    def optimize(self):
        for now_layer in reversed(self.parent_model.now_model):
            dw = now_layer.backward()
            now_layer.w -= self.speed * dw
