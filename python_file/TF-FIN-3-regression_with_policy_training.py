import matplotlib.pyplot as plt
import tensorflow as tf
import pandas_datareader.data as web
import numpy as np
import pandas as pd


def get_prices(symbol):
    start, end = '2007-05-02', '2018-12-01'
    data = web.DataReader(symbol, 'yahoo', start, end)
    data = pd.DataFrame(data)
    prices = data['Close']
    prices = prices.astype(float)
    return prices


def get_returns(prices):
    return ((prices - prices.shift(-1)) / prices)[:-1]


def get_data(list):
    l = []
    for symbol in list:
        rets = get_returns(get_prices(symbol))
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

# normalize inputs (this is new but not specific to PG; you should always normalize inputs)
train_ins, test_ins = train_ins / np.std(ins), test_ins / np.std(ins)

sess = tf.InteractiveSession()

positions = tf.constant([-1, 0, 1])  # long, neutral or short
num_positions = 3

x = tf.placeholder(tf.float32, [None, len(symbol_list) * 100])
y_ = tf.placeholder(tf.float32, [None, len(symbol_list)])

W = tf.Variable(tf.random_normal([len(symbol_list) * 100, num_positions * len(symbol_list)]))
b = tf.Variable(tf.random_normal([num_positions * len(symbol_list)]))

# we define our model: y = W*x + b
y = tf.matmul(x, W) + b  # y is tensor of shape [num_inputs, num_positions * len(symbol_list)]
# a row of y will look like [prob_symbol_1_short, prob_symbol_1_neutral, prob_symbol_1_long, prob_symbol_2_short, ...]
# note that they are not really probabilities because I did not perform a softmax yet

# loop through symbols, taking the buckets for one symbol at a time
pos = {}
symbol_returns = {}
relevant_target_column = {}
for i in range(len(symbol_list)):
    # ISOLATE the buckets relevant to the symbol and get a softmax as well
    symbol_probs = y[:, i * num_positions:(i + 1) * num_positions]
    symbol_probs_softmax = tf.nn.softmax(symbol_probs)  # softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
    # SAMPLE probability to chose our policy's action
    sample = tf.multinomial(tf.log(symbol_probs_softmax), 1)
    pos[i] = tf.reshape(sample, [-1]) - 1  # choose(-1,0,1)
    # GET RETURNS by multiplying the policy (position taken) by the target return for that day
    symbol_returns[i] = tf.multiply(tf.cast(pos[i], tf.float32), y_[:, i])
    # isolate the output probability the selected policy (for use in calculating gradient)
    # see https://github.com/tensorflow/tensorflow/issues/206 for TF discussion including my solution
    sample_mask = tf.reshape(tf.one_hot(sample, 3), [-1, 3])
    relevant_target_column[i] = tf.reduce_sum(symbol_probs_softmax * sample_mask, 1)

# calculate the PERFORMANCE METRICS for the data chosen
daily_returns_by_symbol = tf.concat(axis=1, values=[tf.reshape(t, [-1, 1]) for t in symbol_returns.values()])
daily_returns = tf.reduce_sum(daily_returns_by_symbol, 1) / 2
total_return = tf.reduce_prod(daily_returns + 1)
ann_vol = tf.multiply(
    tf.sqrt(tf.reduce_mean(tf.pow((daily_returns - tf.reduce_mean(daily_returns)), 2))),
    np.sqrt(252)
)
sharpe = total_return / ann_vol

# since we only train the sampled classes, we will combine them so that we can feed them into cross entropy
training_target_cols = tf.concat(axis=1, values=[tf.reshape(t, [-1, 1]) for t in relevant_target_column.values()])
# we want to either push the gradient toward our selection or away from it. We use these ones to find the direction
#     of the gradient, which we will then multiply by our fitness function
ones = tf.ones_like(training_target_cols)

# this isnt actually a gradient, but karpathy sort of calls it one. Since it's a tensor it sort of is a gradient anyway
gradient = tf.nn.sigmoid_cross_entropy_with_logits(labels=training_target_cols,
                                                   logits=ones)  # should this be a prob???

# COST
# how should we do this step? it depends how we want to group our results.
# Choose your own adventure here by uncommenting a cost fn
# this is the most obvious: we push each weight to what works or not. Try it out...we're gonna be RICH!!!! oh, wait...
# cost = tf.multiply(gradient , daily_returns_by_symbol)
# this takes the overall daily return and pushes the weights so that the overall day wins. Again, it overfits enormously
cost = tf.multiply(gradient, tf.reshape(daily_returns, [-1, 1]))
# this multiplies every gradient by the overall return.
# If the strategy won for the past ten years, we do more of it and vice versa
# cost = tf.multiply(gradient , total_return)

# minimize the cost (push the weights where we want them to go)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(5000):
    sess.run(optimizer, feed_dict={x: train_ins, y_: train_outs})  # .reshape(1,-1).T})
    if (epoch + 1) % 1000 == 0:
        c, t = sess.run([cost, total_return], feed_dict={x: train_ins, y_: train_outs})  # .reshape(1,-1).T})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(t), "total return=", "{:.9f}".format(t - 1))

# in sample results
d, t = sess.run([daily_returns, gradient], feed_dict={x: train_ins, y_: train_outs})

# equity curve
plt.plot(np.cumprod(d + 1))

# out of sample results
d, t = sess.run([daily_returns, gradient], feed_dict={x: test_ins, y_: test_outs})

# out of sample results
plt.plot(np.cumprod(d + 1))
