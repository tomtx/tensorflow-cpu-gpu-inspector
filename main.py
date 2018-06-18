from datetime import datetime

import tensorflow as tf
from tensorflow.python.client import device_lib


def main(tf):
    print("++++++++++++++++++++++++++++++")
    print("+++ Simple TensorFlow Test +++")
    print("++++++++++++++++++++++++++++++")
    print("start: :" + str(datetime.now()))

    # verification of simple TF installation
    print("\n... verifying basic installation")
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    print("... basic installation has been verified")

    # list available devices
    print("\n... listing available devices for computations")
    print(device_lib.list_local_devices())

    # list gnu only
    print("\n... checking GPU devices only")
    tf = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf.list_devices()

    # test GPU computations with TF
    print("\n... checking GPU computations")
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    with tf.Session() as sess:
        print(sess.run(c))
    print("... GPU computations has been completed")

    print("end: :" + str(datetime.now()))
    print("++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    main(tf)
