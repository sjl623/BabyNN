class Base:
    now_model = []
    all_input = []
    all_output = []
    epoch = 0
    batch_num = 0
    now_batch_num = 0
    loss_function = ""
    optimizer = ""

    def __init__(self):
        pass

    def init(self):
        for now_layer in self.now_model:
            now_layer.init()

    def fit(self, input_data, output_data, epoch, batch_num):
        self.epoch = epoch
        self.batch_num = batch_num
        self.all_input = input_data
        self.all_output = output_data

    def get_next_batch(self):
        batch_size = int(self.all_input.shape[0] / self.batch_num)
        now_batch_input = self.all_input[self.now_batch_num * batch_size:self.now_batch_num * batch_size + batch_size]
        now_batch_output = self.all_output[self.now_batch_num * batch_size:self.now_batch_num * batch_size + batch_size]
        now_batch = {"input": now_batch_input, "output": now_batch_output}
        self.now_batch_num += 1
        if self.now_batch_num == self.batch_num:
            self.now_batch_num = 0
        return now_batch
