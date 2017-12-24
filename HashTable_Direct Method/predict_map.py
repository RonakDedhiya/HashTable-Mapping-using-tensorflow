## import library
import pandas as pd
import tensorflow as tf
import numpy as np

## Input tensor to be mapped
input_tensor=tf.constant(["167"])

## Within Session
with tf.Session() as sess:
    ## Loading model
    saver = tf.train.import_meta_graph('model_table/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model_table/'))
    graph = tf.get_default_graph()
    ## Loading Variables (keys and values)
    k=graph.get_tensor_by_name("keys_var:0")
    keys=k.eval()
    v=graph.get_tensor_by_name("value_var:0")
    values=v.eval()
    ## Making HashTable with keys and values
    table =tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(keys, values), "Unknown Error Code")
    ## Return Criteria for corresponding Error_code
    out = table.lookup(input_tensor)
    table.init.run()
    print(out.eval())
