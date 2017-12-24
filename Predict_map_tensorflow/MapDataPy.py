## Import Library
import pandas as pd
import tensorflow as tf
import numpy as np

## Input tensor - Error_code to be mapped
value=tf.constant(["167"])

## Within Session
with tf.Session() as sess:
    ## Loading model
    saver = tf.train.import_meta_graph('model1/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model1/'))
    graph = tf.get_default_graph()
    ## Loading variable (list of Error_code)
    v=graph.get_tensor_by_name("var:0")
    f=v.eval()
    ## Table mapping index to Error_code
    table  = tf.contrib.lookup.index_table_from_tensor(
    mapping=f, num_oov_buckets=1, default_value=-1)
    ## Return index for corresponding Error_code
    ids = table.lookup(value)
    tf.tables_initializer().run()
    d=ids.eval()
    print(d)

## Within Session
with tf.Session() as sess:
    ## Loading model
    saver = tf.train.import_meta_graph('model2/model_criteria.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model2/'))
    graph = tf.get_default_graph()
    ## Loading Variable  (list of Criteria)
    v1=graph.get_tensor_by_name("var1:0")
    tf.global_variables_initializer().run()
    f1=v1.eval()
    indices=tf.constant(d)
    ## Table mapping index to Criteria
    table1 =tf.contrib.lookup.index_to_string_table_from_tensor(mapping=f1, default_value="UNKNOWN")
    ## Return criteria for corresponding index
    values = table1.lookup(indices)
    tf.tables_initializer().run()
    print(values.eval())
