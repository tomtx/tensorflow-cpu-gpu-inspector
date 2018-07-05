# TensorFlow Inspector
> A Python tool for checking TensorFlow (TF) installations & inspecting TF functionalities on available CPU and GPU devices via basic mathematical operations on tensors. Such operations are implemented on matrices, that are represented with a TF *dataflow graph* composed with *constant tensors*. Then, operations on such matrices are run by TF *session* for selected graph parts across a set of available CPU & GPU devices.

## Instructions
Run the *inspector.py* script with the following command (with an activated TensorFlow environment).
```
python inspector.py
```

## Requirements
* Python 3.6+
* TensorFlow 1.6+ (any CPU or GPU installations)
