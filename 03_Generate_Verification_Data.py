import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential, save_model, load_model
import pickle
import random
import pandas as pd
import time
#import seaborn as sn


n_test_samples = 4912
val_pkl = "Val.pkl"
model_name = "Exponential_UNet_LAB_SGD_MainTrain_latest_best.savedmodel"
random.seed(1)

def import_test_data(val_pickle_filename,n_test_samples):
    model_inputs = []
    model_outputs = []
    with open(val_pickle_filename,"rb") as in_pkl_file:
        for i in range(99999): #I have a fear of while True
            try:
                pickle_item = pickle.load(in_pkl_file)
                insertion_location = (random.randint(0,max(0,len(model_inputs)-1)))
                model_inputs.insert(insertion_location,pickle_item[0])
                model_outputs.insert(insertion_location,pickle_item[1])
            except Exception as reason:
                print(f"Failed for some reason,{reason}, at itt {i}")
                break
    print("Attempted to import data")
    print(len(model_inputs),len(model_outputs))
    return np.array(model_inputs)[0:n_test_samples], np.array(model_outputs)[0:n_test_samples]

def import_and_compile_model(modelname):
    model = load_model(modelname,compile=False)
    return model

def test_model(model,X_Inputs,Y_Outputs):
    predictions = model.predict([X_Input])
    return predictions

def generate_binary_mask(float_seg_map):
    temp_test = list(np.argmax(float_seg_map,axis=2).reshape(256*256))
    binary_seg_map = np.eye(17,dtype = bool)[temp_test].reshape(256,256,17)
    return binary_seg_map

def evaluate_results(model_outputs,true_outputs):
    num_categories = len(model_outputs[0][0][0])
    print(num_categories)
    results_array = np.zeros((3,17,2))#accuracy, precision, recall, f1
    time_start = time.time()
    for single_image in range(len(model_outputs)):
        #for image in dataset
        single_img_results_array = []
        binary_seg_map = generate_binary_mask(model_outputs[single_image])
        for layer in range(len(model_outputs[single_image][0][0])):
            #print(model_outputs[single_image,:,:,layer])
            #cv2.imshow("test",model_outputs[single_image,:,:,layer])
            #cv2.waitKey(0)#hopefully shows a binary image...
            #Truepositives = sum(true*pred)
            true_positives = np.sum(binary_seg_map[:,:,layer]*true_outputs[single_image,:,:,layer])
            false_positives = np.sum(np.logical_and(binary_seg_map[:,:,layer] == 1,true_outputs[single_image,:,:,layer] == 0))
            true_negatives = np.sum(np.logical_and(binary_seg_map[:,:,layer] == 0,true_outputs[single_image,:,:,layer] == 0))
            false_negatives = np.sum(np.logical_and(binary_seg_map[:,:,layer] == 0,true_outputs[single_image,:,:,layer] == 1))
            sum_pred = np.sum(true_outputs[single_image,:,:,layer])
            
            results_array[0,layer,0] = true_positives + results_array[0,layer,0] #acc
            results_array[0,layer,1] = sum_pred + results_array[0,layer,1] #acc
            results_array[1,layer,0] = true_positives + results_array[1,layer,0] #prec
            results_array[1,layer,1] = true_positives + false_positives + results_array[1,layer,1] #prec
            results_array[2,layer,0] = true_positives + results_array[2,layer,0] #recall
            results_array[2,layer,1] = true_positives + false_negatives + results_array[2,layer,1] #recall

        if single_image%100 == 0:
            print(f"On image {single_image}, at t= {round(time.time()-time_start,3)}")
        #time.sleep(999)
    assignment_dict = ["airplane","bare-soil","buildings","cars","chaparral"
                       ,"court","dock","field","grass","mobile-home",
                       "pavement","sand","sea","ship","tanks","trees","water"]
    
    with open("results_output.csv","w",encoding="utf-8") as outfile:
        line_text = f"category, accuracy, precision, recall, f1\n"
        outfile.write(line_text)
        for i in range(17):#print test accuracies
            acc = round(results_array[0,i,0]/results_array[0,i,1],10)
            precision = round(results_array[1,i,0]/results_array[1,i,1],10)
            recall = round(results_array[2,i,0]/results_array[2,i,1],10)
            f1 = 2*(precision*recall)/(precision+recall)
            line_text = f"{assignment_dict[i]} , {acc}, {precision}, {recall}, {f1}\n"
            outfile.write(line_text)
            #print(f"truepos {results_array[0,i,0]}, sumpred {results_array[0,i,1]}")
            print(f"{i} class acc = {acc}")
            print(f"{i} class precision = {precision}")
            print(f"{i} class recall = {recall}")

if __name__ == "__main__":
    X_Input, Y_Output = import_test_data(val_pkl,n_test_samples)
    model = import_and_compile_model(model_name)
    model_test_outputs = test_model(model,X_Input,Y_Output)
    metrics = evaluate_results(model_test_outputs, Y_Output)















    
