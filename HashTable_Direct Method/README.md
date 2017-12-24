# MApping using Direct HAshTable Method

Problem Statement:- I have a dataset where I need to map two columns( Error_code and Criteria) i.e
for a specific error code I must get a Criteria Value.

 Tools:- Python 3.6, Tensorflow 1.4

 Implementation:-
This Method does direct mapping between Error_code and Criteria using
simple HashTable.

 We deal with key- value pair in string format here. Thus converted our data
 into list of strings.
 This whole amount of preprocessing results into data which is saved in Variables

 Variables can be saved and can be re used, hence it helps to avoid preprocessing again.

 Code 1: Train_hash_map.py - Preprocess data, saves it and create HashTable for mapping
 Code 2: predict_map.py - Loads Variables and does mapping

 Codes are explained in detail with comments 
