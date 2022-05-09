import numpy as np
import cv2
from os import walk, path
from tensorflow.keras.models import Sequential, save_model, load_model

#Images must be of size 256x256, otherwise they will be resized
input_folder = "Input_Images"
model_name = "Exponential_UNet_LAB_SGD_MainTrain_latest_best.savedmodel"

def load_images(input_folder):
    filenames = []
    for d, s, f in walk(input_folder):
        for file in f:
            if file.endswith((".png",".tif",".jpg",".jpeg")):
                if "model_result" not in file:
                    filenames.append(path.join(input_folder, file))
    file_data = []
    for filename in filenames:
        cv2_BGR = cv2.imread(filename)
        cv2_LAB = cv2.cvtColor(cv2_BGR, cv2.COLOR_BGR2LAB)
        if cv2_LAB.shape!= (256,256,3):
            cv2_LAB = cv2.resize(cv2_LAB,(256,256))
        cv2_LAB = np.array(cv2_LAB/127.5-1,dtype=np.float16)
        
        file_data.append(cv2_LAB)
    return file_data,filenames

def import_and_compile_model(modelname):
    model = load_model(modelname,compile=False)
    return model

def test_model(model,input_images):
    predictions = model.predict([input_images])
    return predictions

def color_results(model_outputs,filenames):
    colored_images = []
    assignment_dict = [[[240,202,166],"airplane"],
                       [[0,128,128],"bare-soil"],
                       [[128,0,0],"buildings"],
                       [[0,0,255],"cars"],
                       [[0,128,0],"chaparral"],
                       [[0,0,128],"court"],
                       [[233,233,255],"dock"],
                       [[164,160,160],"field"],
                       [[128,128,0],"grass"],
                       [[255,87,90],"mobile-home"],
                       [[0,255,255],"pavement"],
                       [[0,192,255],"sand"],
                       [[255,0,0],"sea"],
                       [[192,0,255],"ship"],
                       [[128,0,128],"tanks"],
                       [[0,255,0],"trees"],
                       [[255,255,0],"water"]]
    for single_image in range(len(model_outputs)):
        print("Image processed")
        colored_img = np.zeros((256,256,3),dtype=np.uint8)
        temp_test = list(np.argmax(model_outputs[single_image],axis=2).reshape(256*256))
        binary_seg_map = np.eye(17,dtype = bool)[temp_test].reshape(256,256,17)
        for layer in range(len(model_outputs[single_image][0][0])):#Should be 17
            m1 = np.zeros((256,256,3))
            m1[:,:,0] = binary_seg_map[:,:,layer]
            m1[:,:,1] = binary_seg_map[:,:,layer]
            m1[:,:,2] = binary_seg_map[:,:,layer]
            m2 = np.zeros((256,256,3),dtype=int)
            m2[:,:,0].fill(assignment_dict[layer][0][0])
            m2[:,:,1].fill(assignment_dict[layer][0][1])
            m2[:,:,2].fill(assignment_dict[layer][0][2])
            color_to_add = np.multiply(m1,m2)
            colored_img = colored_img + color_to_add
            #cv2.imshow(f"Test_{layer}",np.array(binary_seg_map[:,:,layer]*255,dtype=np.uint8))
            #cv2.waitKey(0)
        #print(colored_img.shape)
        #print(colored_img[0])
        #cv2.imshow("output",colored_img)
        #cv2.waitKey(0)
        cv2.imwrite(f"{filenames[single_image].split('.')[0]}_model_result.png",colored_img)
            
            

if __name__ == "__main__":
    input_images,filenames = load_images(input_folder)
    print(f"{len(input_images)} Images to be processed")
    model = import_and_compile_model(model_name)
    model_outputs = test_model(model,input_images)
    color_results(model_outputs,filenames)
