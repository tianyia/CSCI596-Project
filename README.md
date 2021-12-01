## CSCI 596 Final Project: Google Cloud TPU

Tianyi An, Yiming Wang

### 1.Introduction

Tranining time of machine learning model is important to the efficiency of related research and business deployment. We are hoping to utilize Google Cloud TPU for training of different machine learning models, and compare their performances with training on GPU.

### 2.Why choose Cloud TPU (Tensor Processing Unit)?

* A new tool for machine learning : easy to learn, no need to setup or download locally
* Higher performance on training models for machine learning 
* Designed as a matrix processor specifically for machine learning

- CPU:

![](https://cloud.google.com/tpu/docs/images/image6.gif)

* GPU: thousands of ALUs

  ![](https://cloud.google.com/tpu/docs/images/image2.gif)

* TPU:  [systolic array](https://en.wikipedia.org/wiki/Systolic_array)

  ![](https://cloud.google.com/tpu/docs/images/image4_5pfb45w.gif)

### 3.Details

#### How to use it?

For TensorFlow
* Set up Cloud Storage bucket, VM, and Cloud TPU resources
* Install dependencies and configure the environment
* Prepare datasets
* Train the model



----------

## More

TBA.
