## Import Library
import pandas as pd
import tensorflow as tf
import numpy as np

## Loading Data
data=pd.read_csv("MapData.csv",encoding="ISO-8859-1",usecols =["ERROR_CODE","Criteria"])

######## Part1: Mapping Error_code to index #############


## Convert Error_code into list of strings
ls=data.ERROR_CODE.tolist()
for k in range(len(ls)):
    ls[k]=str(ls[k])

## Hashtable can work only with tf.constant
map_string=tf.constant(ls)

## Session can only save variables and models but cannot constant
## Saving constant into variables
var=tf.Variable(map_string, name="var")

## Test Input (It is the error_code whose index we need to find in table)
value=tf.constant(["120"])

## Calling saver object
saver = tf.train.Saver()

## Within tf.Session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    f=var.eval()
    ## It makes a table with error code as values and index as keys
    table  = tf.contrib.lookup.index_table_from_tensor(
    mapping=f, num_oov_buckets=1, default_value=-1)
    ## It returns a index value
    ids = table.lookup(value)
    tf.tables_initializer().run()
    d=ids.eval()
    print(ids.eval() )
    ## Save the session(Will save variables for us)
    save_path = saver.save(sess, "model1/model.ckpt")
    print("Model saved in file: %s" % save_path)


########## Part2: Mapping index to Criteria ########

## Convert Criteria into list of strings
Criteria=data.Criteria.tolist()
for k in range(len(Criteria)):
    Criteria[k]=str(Criteria[k])

## HastTable can work only with constant
mapping_string = tf.constant(Criteria)

## Session can only save variables and models but cannot constant
## Saving constant into variables
var1=tf.Variable(mapping_string, name="var1")

## This is the index value of error_code we got from previous session
indices = tf.constant(d)

## Within tf.Session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    m=var1.eval()
    ## It makes table with Criteria as value and index as keys
    table1 =tf.contrib.lookup.index_to_string_table_from_tensor(mapping=m, default_value="UNKNOWN")
    ## It wil lookup for string value for the corresponding index value
    values = table1.lookup(indices)
    tf.tables_initializer().run()
    print(values.eval())
    ## Save the session(This will save the variabales for us)
    save_path = saver.save(sess, "model2/model_criteria.ckpt")
    print("Model saved in file: %s" % save_path)
