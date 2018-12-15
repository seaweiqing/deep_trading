import matplotlib.pyplot as plt
import tensorflow as tf
import pandas_datareader.data as web
import numpy as np
import pandas as pd


def get_prices(symbol):  # extract data from yahoo finance (daily close price)
    start, end = '2007-05-02', '2018-12-01'
    data = web.DataReader(symbol, 'yahoo', start, end)
    data = pd.DataFrame(data)
    prices = data['Close']
    prices = prices.astype(float)
    return prices


def get_returns(prices):  # calculate daily return of stock
    return (prices - prices.shift(-1)) / prices


def sort_data(rets):  # divide return into training set and testing set
    ins = []
    outs = []
    for i in range(len(rets) - 100):
        ins.append(rets[i:i + 100].tolist())
        outs.append(rets[i + 100])
    return np.array(ins), np.array(outs)


# separate data into inputs and outputs for training and testing
gs = get_prices('GS')  # price of gs
rets = get_returns(gs)  # return of gs
ins, outs = sort_data(rets)  # dataset
div = int(.8 * ins.shape[0])
train_ins, train_outs = ins[:div], outs[:div]
test_ins, test_outs = ins[div:], outs[div:]

print(test_ins)

sess = tf.InteractiveSession()

# we define two placeholders for our input and output
x = tf.placeholder(tf.float32, [None, 100])
y_ = tf.placeholder(tf.float32, [None, 1])

# we define trainable variables for our model
W = tf.Variable(tf.random_normal([100, 1]))
b = tf.Variable(tf.random_normal([1]))

# we define our model: y = W*x + b, predicted value
y = tf.matmul(x, W) + b

# MSE loss with gradient descent:
cost = tf.reduce_sum(tf.pow(y - y_, 2)) / (2 * 1000)
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# initialize variables to random values (random initialization)
init = tf.global_variables_initializer()
sess.run(init)
# run optimizer on entire training data set
for epoch in range(20000):
    sess.run(optimizer, feed_dict={x: train_ins, y_: train_outs.reshape(1, -1).T})
    # every 1000 iterations record progress
    if (epoch + 1) % 1000 == 0:  # for every 1000 epoch, print cost
        c = sess.run(cost, feed_dict={x: train_ins, y_: train_outs.reshape(1, -1).T})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

# train results :)
predict = y
p = sess.run(predict, feed_dict={x: train_ins})
position = 2 * ((p > 0) - .5)
returns = position.reshape(-1) * train_outs
plt.plot(np.cumprod(returns + 1))
plt.show()

# test results :(
predict = y
p = sess.run(predict, feed_dict={x: test_ins})
position = 2 * ((p > 0) - .5)
returns = position.reshape(-1) * test_outs
plt.plot(np.cumprod(returns + 1))
plt.show()

test = 1
for i in range(len(returns)):
    test = test * (returns[i] + 1)

