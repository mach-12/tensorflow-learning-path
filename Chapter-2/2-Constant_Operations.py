import tensorflow as tf

# Let's define a scaler Constant
t_1 = tf.constant(4)

# Vector Constant
t_2 = tf.constant([6, 2, 5])

# [MxN] shaped int32 Vector Constant of Zeros
M = 5
N = 4
t_3 = tf.constant([M, N], tf.int32)

# Creating a ones array like of a previous tensor shape
t_4 = tf.ones_like(t_2, tf.float32)

# Operation Broadcasting
t_5 = t_4 * 5

# Evenly Spaced Sequence of 'num' Vectors between 'start' and 'stop'
t_6 = tf.linspace(start=N, stop=M, num=10)  # Gives 10 evenly spaced numbers b/w 4 and 5

# Range of vectors similar to Pythonic range(start, stop, stride)
t_7 = tf.range(start=N, limit=M, delta=1)

# Random Distributions
tf.random.set_seed(44)
t_8 = tf.random.normal(shape=(4, 6), mean=0, stddev=1)
t_9 = tf.random.truncated_normal(shape=(4, 6), mean=0, stddev=1)
t_10 = tf.random.uniform(shape=[M, N], minval=0, maxval=6)
t_11 = tf.random.shuffle(t_10)


def print_tensor(name, t):
    """
    Prints the name and value of a tensor to the console.
    """
    print(f"{name}: {t}")


print("Output:")

for i in range(11):
    print_tensor(f"t_{i + 1}", eval(f"t_{i + 1}"))
