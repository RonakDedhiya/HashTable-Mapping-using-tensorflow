import pickle
from flask import Flask
from flask import request
app = Flask(__name__)


class_names=["Normal",
            "NoNetwork",
            "HighRAM",
            "HighRAM_NoNetwork",
            "HighCPU",
            "HighCPU_NoNetwork",
            "HighCPU_HighRAM",
            "HighCPU_HighRAM_NoNetwork"]

file_name="Classifier_model"
with open(file_name, 'rb') as file:  
    clf= pickle.load(file)

result=clf.predict([[390,45,0]])
print(class_names[int(result)])

@app.route('/',methods=['POST'])
def hello_world():
    content=request.json
    CPU_Value=content["CPU_Value"]
    Available_RAM=content["Available_RAM"]
    Network_Bytes=20
    result=clf.predict([[Available_RAM,CPU_Value,Network_Bytes]])
    return class_names[int(result)]

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=9000)
