## CSCI 596 Final Project: Google Cloud TPU

### Why choose Cloud TPU (Tensor Processing Unit)?

* A new tool for machine learning : easy to learn, no need to setup or download locally
* Higher performance on training models for machine learning 

### Why TPU can be better?

- CPU:

![](https://cloud.google.com/tpu/docs/images/image6.gif)

* GPU: thousands of ALUs

  ![](https://cloud.google.com/tpu/docs/images/image2.gif)

* TPU:  [systolic array](https://en.wikipedia.org/wiki/Systolic_array)

  ![](https://cloud.google.com/tpu/docs/images/image4_5pfb45w.gif)

### Create a Model with TPU

```python
def model_fn(features, laberls, mode, params):
    input_layer = tf.reshape(features,[-1,28,28,1])
    conv1 = tf.layers.conv2d(inputs=input_layer,...)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],...)
    loss = tf.losses.softmax_cross_entropy(oneshot_labels=onehot_labels, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    ###
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
    ###
    train_op = optimizer.minimize(loss)
    return tpu_estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
```

### How to use it?

* TBA



----------

## The Final Proposal

TBA.
