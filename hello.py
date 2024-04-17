import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0],True)

x =tf.constant([[1,2,3],[4,5,6]], shape=(3,2))
print(x)
y = tf.eye(5,5)  # here its eye for idendity matrix
print(y)

z= tf.range(start =1, limit=100, delta =5) # same as python range
print(z)


x = tf.constant([1,2,3])
y= tf.constant([9,8,7])

z= x+y
print(z)

a= tf.tensordot(x,y, axes =1)
print(a)


x =tf.random.normal((2,3))
y = tf.random.normal((3,4))
b = x @ y
print(b)


x = tf.range(9)

x= tf.reshape(x, (3,3))
print(x)

y = tf.transpose(x, (1,0))  # transposing the matrix
print(y)

