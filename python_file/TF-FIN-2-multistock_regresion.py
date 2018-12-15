# get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas_datareader.data as web
import numpy as np
import pandas as pd


# we modify this data organizing slightly to get two symbols
def get_prices(symbol):
    start, end = '2007-05-02', '2018-12-01'
    data = web.DataReader(symbol, 'yahoo', start, end)
    data = pd.DataFrame(data)
    prices = data['Close']
    prices = prices.astype(float)
    return prices


def get_returns(prices):
    return ((prices - prices.shift(-1)) / prices)[:-1]


def get_data(list):  # change into list for getting more than one company's data
    l = []
    for symbol in list:
        rets = get_returns(get_prices(symbol))  # get price and returns for each symbol in list
        l.append(rets)
    return np.array(l).T


def sort_data(rets):
    ins = []
    outs = []
    for i in range(len(rets) - 100):
        ins.append(rets[i:i + 100].tolist())
        outs.append(rets[i + 100])
    return np.array(ins), np.array(outs)


symbol_list = ['C', 'GS']
rets = get_data(symbol_list)
ins, outs = sort_data(rets)
ins = ins.transpose([0, 2, 1]).reshape([-1, len(symbol_list) * 100])
div = int(.8 * ins.shape[0])
train_ins, train_outs = ins[:div], outs[:div]
test_ins, test_outs = ins[div:], outs[div:]

sess = tf.InteractiveSession()

# once again I only make slight modifications

# define placeholders 
x = tf.placeholder(tf.float32, [None, len(symbol_list) * 100])
y_ = tf.placeholder(tf.float32, [None, len(symbol_list)])

# define trainable variables
W = tf.Variable(tf.random_normal([len(symbol_list) * 100, len(symbol_list)]))
b = tf.Variable(tf.random_normal([len(symbol_list)]))

# we define our model: y = W*x + b
y = tf.matmul(x, W) + b

# MSE:
cost = tf.reduce_sum(tf.pow(y - y_, 2)) / (2 * 1000)
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# initialize variables to random values
init = tf.global_variables_initializer()
sess.run(init)
# run optimizer on entire training data set many times
for epoch in range(20000):
    sess.run(optimizer, feed_dict={x: train_ins, y_: train_outs})  # .reshape(1,-1).T})
    # every 1000 iterations record progress
    if (epoch + 1) % 1000 == 0:
        c = sess.run(cost, feed_dict={x: train_ins, y_: train_outs})  # .reshape(1,-1).T})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

# train results
predict = y
p = sess.run(predict, feed_dict={x: train_ins})
position = 2 * ((p > 0) - .5)
returns = position * train_outs
# daily_returns = sum(returns, 1)
plt.plot(np.cumprod(returns + 1))
plt.show()

# test results
predict = y
p = sess.run(predict, feed_dict={x: test_ins})
position = 2 * ((p > 0) - .5)
returns = position * test_outs
# daily_returns = sum(returns, 0)
plt.plot(np.cumprod(returns + 1))
plt.show()
