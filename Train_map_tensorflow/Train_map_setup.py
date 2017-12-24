import pandas as pd
import tensorflow as tf
import numpy as np

data=pd.read_csv("MapData.csv",encoding="ISO-8859-1",usecols =["ERROR_CODE","Criteria"])
ls=data.ERROR_CODE.tolist()

for k in range(len(ls)):
    ls[k]=str(ls[k])

map_string=tf.constant(ls)
var=tf.Variable(map_string, name="var")
#var1=tf.constant(var.eval())
value=tf.constant(["120"])
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    f=var.eval()
    table  = tf.contrib.lookup.index_table_from_tensor(
    mapping=f, num_oov_buckets=1, default_value=-1)
    ids = table.lookup(value)
    tf.tables_initializer().run()
    d=ids.eval()
    print(ids.eval() )
    save_path = saver.save(sess, "model1/model.ckpt")
    print("Model saved in file: %s" % save_path)


Criteria=data.Criteria.tolist()
for k in range(len(Criteria)):
    Criteria[k]=str(Criteria[k])
    
mapping_string = tf.constant(Criteria)
var1=tf.Variable(mapping_string, name="var1")
indices = tf.constant(d)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    m=var1.eval()
    table1 =tf.contrib.lookup.index_to_string_table_from_tensor(mapping=m, default_value="UNKNOWN")
    values = table1.lookup(indices)
    tf.tables_initializer().run()
    print(values.eval())
    save_path = saver.save(sess, "model2/model_criteria.ckpt")
    print("Model saved in file: %s" % save_path)
