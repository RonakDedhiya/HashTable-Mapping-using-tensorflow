import pandas as pd
import tensorflow as tf
import numpy as np


value=tf.constant(["167"])
with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('model1/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model1/'))
    graph = tf.get_default_graph()
    v=graph.get_tensor_by_name("var:0")
    f=v.eval()
    table  = tf.contrib.lookup.index_table_from_tensor(
    mapping=f, num_oov_buckets=1, default_value=-1)
    ids = table.lookup(value)
    tf.tables_initializer().run()
    d=ids.eval()
    print(d)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model2/model_criteria.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model2/'))
    graph = tf.get_default_graph()
    v1=graph.get_tensor_by_name("var1:0")
    tf.global_variables_initializer().run()
    f1=v1.eval()
    indices=tf.constant(d)
    table1 =tf.contrib.lookup.index_to_string_table_from_tensor(mapping=f1, default_value="UNKNOWN")
    values = table1.lookup(indices)
    tf.tables_initializer().run()
    print(values.eval())

