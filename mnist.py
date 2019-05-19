import model
import layer
import optimizers
import pickle
import util
import numpy
import matplotlib.pyplot as plt

train_set, val_set, test_set = pickle.load(open("mnist.pkl", "rb"), encoding='latin1')

model = model.Sequence()
model.add(layer.Dense(300, input_dim=28 * 28, activation="Relu"))
#model.add(layer.Dense(300, activation="Relu"))
model.add(layer.Dense(10))

train_y = util.to_categorical(train_set[1])
idx = numpy.random.choice(train_set[0].shape[0], 50000)
train_set = train_set[0][idx]
train_y = train_y[idx]

model.init()
model.fit(input_data=train_set, output_data=train_y, epoch=500, batch_num=10)
model.compile(optimizer=optimizers.SGD(model, 0.1), loss="Mean_squared_error")
model.train()

id = 0
rightnum = 0
for now in val_set[0]:
    # plt.imshow(numpy.reshape(now,(28,28)))
    # plt.show()
    ans = val_set[1][id]
    res = model.predict(now)
    ansnum = numpy.argmax(res)
    if (ansnum == ans):
        rightnum += 1
    else:
        print(ans, ansnum)
    id += 1
print(rightnum)
