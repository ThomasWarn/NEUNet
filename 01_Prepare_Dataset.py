import numpy as np
import cv2
from os import walk,path
import pickle
import random
import time

Train_Frac = 0.7
Image_Dim = 256
sat_dataset_folder = "DLRSD"
seg_dataset_folder = "DLRSD_Segmentations"
Required_In_Validation = ["airplane27","harbor00",
                          "storagetanks03","storagetanks88.png",
                          "airplane24","storagetanks92"]
random.seed(18)

def grab_rel_filepaths(sat_dataset_folder):
    master_train_filenames = []
    master_val_filenames = []
    
    for subf, f, files in walk(sat_dataset_folder):
        filepaths = []
        for item in files:
            if ".txt" not in item:
                filepaths.append(path.join(subf,item))
        randomly_shuffled = filepaths.copy()
        #print(subf,f,randomly_shuffled)
        #time.sleep(9)
        random.shuffle(randomly_shuffled)
        train_cutoff = int(Train_Frac*len(randomly_shuffled))
        train_filenames = randomly_shuffled[0:train_cutoff]
        val_filenames = randomly_shuffled[train_cutoff:]
        
        train_filenames = [temp.replace(".png","") for temp in train_filenames]
        val_filenames = [temp.replace(".png","") for temp in val_filenames]

        #Moves over items in Required_In_Validation if present.
        for item in Required_In_Validation:
            if item in train_filenames:
                train_filenames.pop(train_filenames.index(item))
                val_filenames.append(item)
        #print(train_filenames)
        master_train_filenames.extend(train_filenames)
        master_val_filenames.extend(val_filenames)
    #print(master_train_filenames)
    random.shuffle(master_train_filenames)
    random.shuffle(master_val_filenames)
    return master_train_filenames, master_val_filenames

def perform_segmentation(BGR_image,assignment_array):
    segmentation_array = np.zeros((256,256,len(assignment_array)),dtype=bool)
    for class_int in range(len(assignment_array)):
        color, name = assignment_array[class_int]
        #print(name,color)
        #print(BGR_image[0])
        temp_mask = cv2.inRange(np.array(BGR_image),np.array(color),np.array(color))
        temp_mask = np.array(temp_mask,dtype=bool)
        segmentation_array[:,:,class_int] = temp_mask
        
    #for line in temp_mask:
    #    print(line)
    #    time.sleep(999)
    #print(temp_mask)
    #print(f"amax = {np.amin(temp_mask)}, amin = {np.amax(temp_mask)}")
    #cv2.imshow("Mask",temp_mask)
    #cv2.waitKey(1)
    #time.sleep(999)
    #time.sleep(999)
    return segmentation_array
def rotate_and_mirror(in_sat_img,in_seg_img,file_obj):
    pickle.dump([in_sat_img,in_seg_img],file_obj)
    in_sat_img = np.rot90(in_sat_img,k=1,axes=(0,1))
    in_seg_img = np.rot90(in_seg_img,k=1,axes=(0,1))
    pickle.dump([in_sat_img,in_seg_img],file_obj)
    in_sat_img = np.rot90(in_sat_img,k=1,axes=(0,1))
    in_seg_img = np.rot90(in_seg_img,k=1,axes=(0,1))
    pickle.dump([in_sat_img,in_seg_img],file_obj)
    in_sat_img = np.rot90(in_sat_img,k=1,axes=(0,1))
    in_seg_img = np.rot90(in_seg_img,k=1,axes=(0,1))
    pickle.dump([in_sat_img,in_seg_img],file_obj)

    in_sat_img = np.rot90(in_sat_img,k=1,axes=(0,1))#rotates back to initial
    in_seg_img = np.rot90(in_seg_img,k=1,axes=(0,1))
    in_sat_img = np.flip(in_sat_img,axis=0)
    in_seg_img = np.flip(in_seg_img,axis=0)
    
    pickle.dump([in_sat_img,in_seg_img],file_obj)
    in_sat_img = np.rot90(in_sat_img,k=1,axes=(0,1))
    in_seg_img = np.rot90(in_seg_img,k=1,axes=(0,1))
    pickle.dump([in_sat_img,in_seg_img],file_obj)
    in_sat_img = np.rot90(in_sat_img,k=1,axes=(0,1))
    in_seg_img = np.rot90(in_seg_img,k=1,axes=(0,1))
    pickle.dump([in_sat_img,in_seg_img],file_obj)
    in_sat_img = np.rot90(in_sat_img,k=1,axes=(0,1))
    in_seg_img = np.rot90(in_seg_img,k=1,axes=(0,1))
    pickle.dump([in_sat_img,in_seg_img],file_obj)
    
def write_data_to_pickle(filenames,out_pkl_name):
    assignment_array = [[[240,202,166],"airplane"],
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
    ##print(train_filenames)
    counter = 0
    with open(out_pkl_name,"wb") as pkl_out:
        for image_filename in filenames:
            counter += 1
            #print(image_filename)
            try:
                BGR_image = cv2.imread(image_filename)
                LAB_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2LAB)
                LAB_image = np.array(LAB_image/127.5-1,dtype=np.float16)
                BGR_segmented_image = cv2.imread(image_filename.replace("DLRSD","DLRSD_Segmented\Images").replace(".tif",".png"))
                BGR_segmented_image = np.array(BGR_segmented_image,dtype=np.uint8)
                
                if counter%100 == 0:
                    print(f"counter @ {counter}")
                if BGR_image.shape == (256,256,3) and BGR_segmented_image.shape == (256,256,3):
                    segmentation_image = perform_segmentation(BGR_segmented_image,assignment_array)
                    rotate_and_mirror(LAB_image,segmentation_image,pkl_out)
            except Exception as e:
                print(f"Error on image {image_filename}. Probably issue with BGR2LAB")

if __name__ == "__main__":
    train_filenames, val_filenames = grab_rel_filepaths(sat_dataset_folder)
    #print(train_filenames)
    print(f"{len(train_filenames)} training images")
    write_data_to_pickle(train_filenames,"Train.pkl")
    print(f"{len(val_filenames)} validation images")
    write_data_to_pickle(val_filenames,"Val.pkl")
    #write_data_to_pickle(, val_filenames)
