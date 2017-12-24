# Creating HashTable and Saving the Session

Problem Statement:- I have a dataset where I need to map two columns( Error_code and Criteria) i.e
for a specific error code I must get a Criteria Value.

 Tools:- Python 3.6, Tensorflow 1.4

 Implementation:-
 This is one of the method to do mapping. Here instead of making a 1 Direct Hastable containing
 Error_code as keys and Criteria as values. We have been making two table. !st table Maps
 Error_code to index and 2nd table maps index to Criteria.

Here to work with HashTable I have considered to use everything in string and converted Error_Code
and Criteria in to list of Strings and then finally to tensors.

If we have to use model again to query a Error_code and get Criteria. There is no point to do this
Preprocessing(Converting data into list of strings and then tensor) again and again.
Thus I preferred to save the tensor into variables, save the session and then directly use it by
loading it.

This is a tutorial to create hashtable and saving the session.The code is explained in detail with comments in Train_map_setup.py File

There is another code  to load session and use it directly.(https://github.com/RonakDedhiya/HashTable-Mapping-using-tensorflow/tree/master/Predict_map_tensorflow)
