import numpy as np
import matplotlib.pyplot as plt
from chainer import  Variable, optimizers, serializers,initializers,cuda
import pickle
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, cuda


def generate_input(output_path):
    x = 30*np.random.rand(1000).astype(np.float32)
    # y = 7*(x**2)+5*x + 9
    y = 5*x + 9
    y += 10*np.random.randn(1000).astype(np.float32)
    with open(output_path, "wb") as fp:
        pickle.dump({'x':x,'y':y}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def get_input(pickle_object_path):
    with open(pickle_object_path, "rb") as fp:
        pickle_data = pickle.load(fp)
    return pickle_data['x'],pickle_data['y']

class MyClassifier(Chain):
    prior = 0

    def __call__(self, x, t, loss_func):
        self.clear()
        h = self.calculate(x)
        print("h",h)
        self.loss = loss_func(h, t)
        print("loss_func",loss_func,"t",t,"h",h,"self.loss",self.loss)
        chainer.reporter.report({'loss': self.loss}, self)
        return self.loss

    def clear(self):
        self.loss = None

    def calculate(self, x):
        return None

    def error(self, x, t):
        xp = cuda.get_array_module(x, False)
        size = len(t)
        with chainer.no_backprop_mode():
            with chainer.using_config("train", False):
                h = xp.reshape(xp.sign(self.calculate(x).data), size)
        if isinstance(h, chainer.Variable):
            h = h.data
        if isinstance(t, chainer.Variable):
            t = t.data
        result = (h != t).sum() / size
        chainer.reporter.report({'error': result}, self)
        return cuda.to_cpu(result) if xp != np else result


class QuadLayer(chainer.Link):

    def __init__(self, n_in, n_out):
        super(QuadLayer, self).__init__()
        with self.init_scope():
            # self.a = chainer.Parameter(
            #     initializers.Normal(), (n_in,1))
            # self.b = chainer.Parameter(
            #     initializers.Normal(), (n_in,1))
            # self.c = chainer.Parameter(
            #     initializers.Zero(), (n_in,1))
            self.W = chainer.Parameter(
                initializers.Normal(), (n_out, n_in))
            self.bias = chainer.Parameter(
                initializers.Normal(), (n_out,))

    def __call__(self, x):

        # inputs = x, self.a, self.b, self.c
        # input_vars = [chainer.as_variable(x) for x in inputs]
        # in_data = tuple([x.data for x in input_vars])
        #
        # x = in_data[0]
        # a = in_data[1]
        # b = in_data[2]
        # c = in_data[3]
        # print("a",a,"b",b,"c",c)
        # x_sq = x * x
        # var1 = x_sq.dot(a.T).astype(x_sq.dtype, copy=False)
        # var2 = x.dot(b.T).astype(x.dtype, copy=False)
        # value = var1 + var2 + c
        # x1 = x.data
        # x_sq = x*x
        # x_sq = x_sq.data
        # value = x_sq.dot(self.a.T) + x.dot(self.b.T) + self.c.data
        # value = value.astype(value.dtype, copy=False)
        # # in_data = tuple([x1.data for x1 in x])
        # # # bool a = isinstance(x, np.ndarray)
        # # # print(self.a.shape,self.b.shape,x.shape)
        # # x = in_data[0]
        # # x_new_sq = F.square(x)
        # # if x_new_sq.ndim > 2:
        # #     x_new = x_new_sq.reshape(len(x), -1)
        # #
        #
        # # value = x.dot(self.a.T)
        # # # value = self.a * F.square(x) + self.b * x + self.c
        # # #
        # return value
        print("W",self.W,"bias",self.bias)
        check_val = F.linear(x, self.W, self.bias)
        return check_val

# generate_input("test_data.pickle")

quad_function = QuadLayer(1, 1)
def train_data(data, train_target,n_epochs=200):
    # x, y = get_input("test_data.pickle")
    # # plt.scatter(x,y)
    # # plt.xlabel('x')
    # # plt.ylabel('y')
    # x_var = Variable(x.reshape(1000, -1))
    # y_var = Variable(y.reshape(1000, -1))
    # print(n_epochs)
    optimizer = optimizers.MomentumSGD(lr=0.001)


    optimizer.setup(quad_function)
    for _ in range(n_epochs):
        # Get the result of the forward pass.
        output = quad_function(data)

        # Calculate the loss between the training data and target data.
        loss = F.mean_squared_error(train_target, output)
        print("loss", loss)
        # Zero all gradients before updating them.
        quad_function.cleargrads()

        # Calculate and update all gradients.
        loss.backward()

        # Use the optmizer to move all parameters of the network
        # to values which will reduce the loss.
        optimizer.update()


generate_input("test_data.pickle")
x, y = get_input("test_data.pickle")
x_var = Variable(x.reshape(1000, -1))
y_var = Variable(y.reshape(1000, -1))
train_data(x_var, y_var, n_epochs=555)
y_pred = quad_function(x_var)
# print(y_pred.data)
loss = F.mean_squared_error(y_var, y_pred)
print("loss",loss)

# get_training_data()
