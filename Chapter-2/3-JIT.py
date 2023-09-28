import tensorflow as tf
import timeit

cell = tf.keras.layers.LSTMCell(100)


# The tf.function decorator is used to utilize JIT which enables dynamic bindings without
# the need of explicitly defining placeholders and constants

@tf.function
def fn(input_, state_):
    return cell(input_, state_)


input_ = tf.zeros([100, 100])

state = [tf.zeros([100, 100])] * 2

# warmup
cell(input_, state)
fn(input_, state)
graph_time = timeit.timeit(lambda: cell(input_, state), number=100)
auto_graph_time = timeit.timeit(lambda: fn(input_, state), number=100)

print('graph_time:', graph_time)
print('auto_graph_time:', auto_graph_time)
