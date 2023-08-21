import tflite_runtime.interpreter as tflite
import numpy as np
from time import time
# buffering the last received n frames
n= 50
m= 128
buffer = np.zeros((n,m),dtype="float32")
def message(timestamp_ms, output_data):
    """Constructs message

    Args:
        timestamp_ms (_type_): time stamp of the sent message
        output_data (_type_): output of the model 

    Returns:
        payload (dict): Dictionary, msg to be sent to the cloud
    """
    payload = {
        "t": timestamp_ms,   # unix time, in milliseconds
        "r": 2,              # the Radar Id, 1 == KUL in Leuven, 2 == KUL in Geel, 3 == Televic
        "p": 1,              # person identification from the algorithm
        "c": "m",            # means activity
        "m": output_data     # 10 value vector with probabilities [0-100] 
        }
    return payload
def run_model(model,power):
    """invokes model to make a prediction

    Args:
        model (_type_): loaded tflite model
        power (_type_): input to the model

    Returns:
        payload (dict): dictionary containing message to be sent to the cloud 
    """    # get input and output details
    print("hello")
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.allocate_tensors()
    
    # prepare input array, reshape and transform to float32
    power = np.reshape(power,(1,n,m,1))
    power = power.astype("float32")
    
    # set the tensor as input
    model.set_tensor(input_details[0]['index'], power)
    
    # run feedforward
    model.invoke()
    
    # get output probabilities and trans=form to ints
    output_data = model.get_tensor(output_details[0]['index'])
    output_data = output_data*100
    output_data = np.concatenate((output_data,np.zeros(5)),axis=None)
    output_data = output_data.astype('int').tolist()    
    
    # prepare the payload
    timestamp_ms = int(time() * 1000)
    payload = message(timestamp_ms,output_data)
    
    return payload

model = tflite.Interpreter(model_path='/home/pi/Desktop/demonstrator1/model/model.tflite')
print(run_model(model,buffer))
