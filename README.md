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
- Create project and a Cloud Storage bucket
```
export PROJECT_ID=project-id

gcloud config set project ${PROJECT_ID}

gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID

gsutil mb -p ${PROJECT_ID} -c standard -l us-central1 gs://bucket-name
```
- Create a Cloud TPU VM
```
gcloud alpha compute tpus tpu-vm create mnist-tutorial \
--zone=us-central1-b \
--accelerator-type=v3-8 \
--version=tpu-vm-tf-2.7.0
```
- Connect to your Cloud TPU VM
```
gcloud alpha compute tpus tpu-vm ssh mnist-tutorial --zone=us-central1-b
```
- Install dependencies and configure the environment
```
export TPU_NAME=local

pip3 install -r /usr/share/tpu/models/official/requirements.txt
```
- Prepare datasets
```
export STORAGE_BUCKET=gs://bucket-name
export MODEL_DIR=${STORAGE_BUCKET}/mnist
export DATA_DIR=${STORAGE_BUCKET}/data
```
- Train the model
```
export PYTHONPATH="${PYTHONPATH}:/usr/share/tpu/models"

cd /usr/share/tpu/models/official/vision/image_classification

python3 mnist_main.py \
  --tpu=${TPU_NAME} \
  --model_dir=${MODEL_DIR} \
  --data_dir=${DATA_DIR} \
  --train_epochs=10 \
  --distribution_strategy=tpu \
  --download
```

Reference: 

[Cloud TPU TensorFlow Quickstart](https://cloud.google.com/tpu/docs/tensorflow-quickstart-tpu-vm#tpu-vm)

[Running MNIST on Cloud TPU (TF 2.x)](https://cloud.google.com/tpu/docs/tutorials/mnist-2.x)

Result of the MNIST on Cloud TPU:
![result](https://github.com/tianyia/CSCI596-Project/blob/main/3.png)

Result of the ResNet on Cloud TPU:
![result](https://github.com/tianyia/CSCI596-Project/blob/main/ResNet1.JPG)

![result](https://github.com/tianyia/CSCI596-Project/blob/main/ResNet2.JPG)

![result](https://github.com/tianyia/CSCI596-Project/blob/main/ResNet3.JPG)
----------

### 4.Next step

* Train different models on TPU and GPU, gather data like training time and dataset size
* Evaluate training performance based on collected data
