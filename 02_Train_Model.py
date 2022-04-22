import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Conv2D,Flatten,MaxPooling2D,BatchNormalization,LeakyReLU,Activation,Conv2DTranspose,AveragePooling2D,SpatialDropout2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import metrics,optimizers
from keras.optimizers import SGD
import numpy as np
import pickle
import random
import time
import math
from os import mkdir


Train_Pkl = "Train.pkl"
Val_Pkl = "Val.pkl"
max_train_len = 10750
max_test_len = 4000


def import_data(Train_Pkl,Val_Pkl):
    train = []
    #import train data
    with open(Train_Pkl,"rb") as out_file_train:
        for i in range(max_train_len):
            try:
                train.append(pickle.load(out_file_train))
            except Exception as e:
                print(e)
                break
            
    #import test data
    test = []
    try:
        with open(Val_Pkl,"rb") as out_file_train:
            for i in range(max_test_len):
                try:
                    test.append(pickle.load(out_file_train))
                except Exception as e:
                    print(e)
                    break
    except:
        pass
    

    test_out = []
    test_in = []
    for item in test:
        test_in.append(item[0])
        test_out.append(item[1])
    
    test = []
    train_out =[]
    train_in = []
    for item in train:
        train_in.append(item[0])
        train_out.append(item[1])
    train = []

    test_out = np.array(test_out)#*2-1#,dtype = np.float16
    test_in = np.array(test_in)
    train_out = np.array(train_out)#*2-1#,dtype = np.float16
    train_in = np.array(train_in)
    print("##############################################")
    return train_in,train_out,test_in,test_out

def define_model(threshold_size,filter_size,pool_size,starting_filters,encoder_exponent,decoder_exponent,dropout,max_filters,LOG_DIR_ADAM):
    model = Sequential()
    #Try maxpool / avg pooling, with batch norm & without batch norm.
    #Encoder
    inputs = keras.layers.Input(shape=(256,256,3)) #https://stackoverflow.com/questions/50888221/valueerror-layer-leaky-re-lu-1-was-called-with-an-input-that-isnt-a-symbolic-t
    num_encoding_layers = int(math.log(256/threshold_size[0],pool_size[0]))
    print(f"creating network structure...")
    print(f"num endode layers = {round(math.log(256/threshold_size[0],pool_size[0]),2)} with a bottleneck size of {int(256/((pool_size[0])**num_encoding_layers))},{int(256/((pool_size[0])**num_encoding_layers))}")#
    model_segment = []
    for i in range(num_encoding_layers):
        print(i,num_encoding_layers)
        filters_in_layer = min(max_filters,max(starting_filters,int(starting_filters * (encoder_exponent)**i)))
        if i == 0:
            model_seg = Conv2D(kernel_size=filter_size,filters=(filters_in_layer),padding="SAME")(inputs)
            model_segment.append(model_seg)
        else:
            model_seg = Conv2D(kernel_size=filter_size,filters=(filters_in_layer),padding="SAME")(model)
            model_segment.append(model_seg)
        if i!= num_encoding_layers - 1:
            model = LeakyReLU(alpha=0.1)(model_seg)
            if dropout>0:
                #model = SpatialDropout2D(dropout)(model)
                model = Dropout(dropout)(model)
            model = AveragePooling2D(pool_size=pool_size)(model)
        elif i == num_encoding_layers - 1: #Allows for stuff to be changed if needed in bottleneck layer
            model = LeakyReLU(alpha=0.1)(model_seg)
            if dropout>0:
                #model = SpatialDropout2D(dropout)(model)
                model = Dropout(dropout)(model)
            model = AveragePooling2D(pool_size=pool_size)(model)
    
    for j in range(num_encoding_layers):
        filters_in_layer = min(max_filters,max(starting_filters,int(starting_filters * (encoder_exponent)**(num_encoding_layers-1)/(decoder_exponent**j))))
        model = keras.layers.Concatenate()([Conv2DTranspose(filters_in_layer, pool_size, strides=pool_size, padding='SAME')(model),(model_segment[-1-j])])
        model = LeakyReLU(alpha=0.1)(model)
        if dropout>0:
            #model = SpatialDropout2D(dropout)(model)
            model = Dropout(dropout)(model)
    









    
    outputs = Conv2D(kernel_size=(3,3),filters=(17),padding="SAME", activation = "tanh")(model)#Output layer, shape = (256,256,17)?
    model = keras.models.Model(inputs,outputs)
    opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07)
    #opt = SGD(learning_rate=0.1)

    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['mae','mse',"categorical_accuracy"])
    
    with open(f"logs\\{LOG_DIR_ADAM}.txt","w") as txtfile:
        model.summary(print_fn=lambda x: txtfile.write(x + '\n'))
        #txtfile.write(text)
    #print(model.summary())
    time.sleep(10)
    #time.sleep(99999)
    return model

def import_and_compile_model(modelname):
    model = load_model(modelname,compile=False)
    opt=SGD(learning_rate=0.1, momentum=0.1, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['mae','mse',"categorical_accuracy"])
    return model

def train_model(model,tr_x,tr_y,val_x,val_y,N_Epochs,tensorboard, LOG_DIR):
    checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=f'{LOG_DIR}_latest_Best_Val_Best.savedmodel', save_weights_only=False,save_best_only=True,monitor='categorical_accuracy', mode='max')
    #checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=f'{LOG_DIR}_latest_Best_Val.savedmodel', save_weights_only=False)
    model.fit(tr_x,tr_y,epochs=N_Epochs,callbacks=[tensorboard,checkpoint_cb],validation_data=(val_x,val_y),batch_size=16)
    #model.save(LOG_DIR+'.model')
    
if __name__ == "__main__":
    #Testing various configurations; input dataformat.
    Num_Epochs = 100
    LOG_DIR_ADAM = f"Exponential_UNet_LAB_Adam_MainTrain_02"
    LOG_DIR_SGD = f"Exponential_UNet_LAB_SGD_MainTrain_0"
    try: 
        mkdir("logs") 
    except: 
        pass
    tensorboard = keras.callbacks.TensorBoard(log_dir=f'logs/{LOG_DIR_ADAM}')
    train_in,train_out,test_in,test_out = import_data(Train_Pkl,Val_Pkl)

    #Important parameters:
    threshold_size = (1,1)#Dimensions for bottleneck
    filter_size = (3,3) #filter size
    pool_size = (2,2) #max / min / avg pool size
    starting_filters = 30 #starting filters, also acts as min limit for number of filters
    encoder_exponent = 1.4 #Encoder exponent size, (1,~2)
    decoder_exponent = 1.36 #Decoder exponent size, (1,~2), slightly smaller than encoder exponent
    dropout = 0.2 #Dropout ratio, (0,~.5)
    max_filters = 768 # hard cap for max number of filters.

    #First ADAM model train.
    model = define_model(threshold_size,filter_size,pool_size,starting_filters,encoder_exponent,decoder_exponent,dropout,max_filters,LOG_DIR_ADAM)
    train_model(model,train_in,train_out,test_in,test_out,Num_Epochs,tensorboard, LOG_DIR_ADAM)

    #Second SGD model train.
    SGD_model = import_and_compile_model(f"{LOG_DIR_ADAM}_latest_Best_Val_Best.savedmodel")
    tensorboard = keras.callbacks.TensorBoard(log_dir=f'logs/{LOG_DIR_SGD}')
    train_model(SGD_model,train_in,train_out,test_in,test_out,Num_Epochs,tensorboard, LOG_DIR_SGD)

    train_in = []
    train_out = []
    test_in = []
    test_out = []

