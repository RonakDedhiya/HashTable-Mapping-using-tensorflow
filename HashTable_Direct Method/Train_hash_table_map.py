## Import Library
import pandas as pd
import tensorflow as tf
import numpy as np

## Loading Data
data=pd.read_csv("MapData.csv",encoding="ISO-8859-1",usecols =["ERROR_CODE","Criteria"])
ls=data.ERROR_CODE.tolist()
Criteria=data.Criteria.tolist()

## Converting Error_code and Criteria into list of Strings
for k in range(len(ls)):
    ls[k]=str(ls[k])

for k in range(len(Criteria)):
    Criteria[k]=str(Criteria[k])

## input_tensor to be mapped
s="130"
x=tf.placeholder(tf.int32)

## Representing Error_code and Criteria as tensor
keys=tf.constant(ls)
values = tf.constant(Criteria)

## Saving tensor into variable
keys_var=tf.Variable(keys, name="keys_var")
value_var=tf.Variable(values, name="value_var")
saver = tf.train.Saver()

## Within Session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    k=keys_var.eval()
    v=value_var.eval()
    ## HashTable with keys as Error_code and values as Criteria
    table =tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(k, v), "Unknown Error Code")
    val=x.eval(feed_dict={x:s})
    input_tensor=tf.constant(val)
    ## Return Criteria for given Error_code
    out = table.lookup(tf.as_string(input_tensor))
    table.init.run()
    print(out.eval())
    ## Save session/model
    save_path = saver.save(sess, "model_table/model.ckpt")
    print("Model saved in file: %s" % save_path)
