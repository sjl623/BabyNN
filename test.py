import model
import layer
import optimizers
import numpy
import pickle

model = model.Sequence()
model.add(layer.Dense(1, input_dim=2))
model.add(layer.Dense(2))
model.add(layer.Dense(5))

w = numpy.array([[1], [9]])
w2 = numpy.array([[5, 4]])
w3 = numpy.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

text_x = numpy.random.randn(1000, 2)
text_y = numpy.dot(text_x, w)
text_y = numpy.dot(text_y, w2)
text_y = numpy.dot(text_y, w3)
text_y = text_y

model.init()
model.fit(text_x, text_y, epoch=10000, batch_num=100)
model.compile(loss="Mean_squared_error", optimizer=optimizers.SGD(model, speed=0.000001))
model.train()

t = ""
isfirst = True
for now in model.now_model:
    print(now.w)
    if isfirst:
        isfirst = False
        t = now.w
    else:
        t = numpy.dot(t, now.w)
print(numpy.dot(numpy.dot(w, w2), w3))
print(w)
print(t)
