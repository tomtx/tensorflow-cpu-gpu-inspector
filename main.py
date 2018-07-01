from datetime import datetime

import tensorflow as tf
from tensorflow.python.client import device_lib


def verify_tf_installation():
    print("\n++++++++++++++++++++++++++++++")
    # verification of simple TF installation
    print("... verifying basic TF installation")
    # - build a graph (create one constant tensor)
    hello_tf = tf.constant('Hello, a simple TensorFlow installation has been verified!')
    # - launch the graph in a session
    sess = tf.Session()
    # - run the operation
    result = sess.run(hello_tf)
    # - close the session
    sess.close()
    print(result)
    print("++++++++++++++++++++++++++++++")
    
    # listing available devices
    print("\n++++++++++++++++++++++++++++++")
    print("... listing all available devices for computations")
    available_devices = device_lib.list_local_devices()
    count_cpu = 0
    count_gpu = 0
    for device in available_devices:
        if device.device_type == 'CPU':
            count_cpu += 1
        if device.device_type == 'GPU':
            count_gpu += 1
        print("\nDEVICE")
        print(device)
    print("There are " + str(count_cpu) + " available CPU devices!")
    print("There are " + str(count_gpu) + " available GPU devices!")
    print("++++++++++++++++++++++++++++++")
    return available_devices


def test_cpu_gpu_computations(available_devices):
    print("\n++++++++++++++++++++++++++++++")
    print("... placing matrix operations on available CPU/GPU devices")
    # test CPU/GPU computations
    for device in available_devices:
        device_name = str(device.name)
        print("\n" + device_name)
        # - build a graph
        with tf.device(device_name):
            # - create a 2-D tensor of [2, 3] shape
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            # - create a 2-D tensor of [3, 2] shape
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            # - perform a matrix product of those two tensors
            c = tf.matmul(a, b)
        # - launch the graph in a session (use the session as a context manager)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            # - run the operation (evaluate the tensor c)
            result = sess.run(c)
            print(result)
    print("... matrix operations has been completed")
    print("++++++++++++++++++++++++++++++")


def main():
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+++ Simple CPU/GPU Computation Test with TensorFlow +++")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("start: " + str(datetime.now()))
    # verify TF installation
    available_devices = verify_tf_installation()
    # test CPU/GPU computations
    test_cpu_gpu_computations(available_devices)
    print("\nend: " + str(datetime.now()))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    main()
