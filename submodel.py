import numpy as np
import pandas as pd
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.utils.np_utils as kutils
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback,History
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras import backend as K
import keras.metrics
import math
import matplotlib.pyplot as plt
from LossCheckPoint import LossModelCheckpoint
from keras.models import load_model

def OnehotNetwork(train_oneofkeyX,trainY,val_oneofkeyX,valY,
				train_time=None,compilemodels=None):
    
	Oneofkey_input = Input(shape= (train_oneofkeyX.shape[1],train_oneofkeyX.shape[2]))  #49*21=1029
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=7)
	
	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l1(0),border_mode="same",name="0")(Oneofkey_input)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="1")(x)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="2")(x)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)
		
		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		
		x = Dense(256,init='glorot_normal',activation='relu',name="3")(x)
		x = Dropout(0.298224)(x)
		
		x = Dense(128,init='glorot_normal',activation="relu",name="4")(x)
		x = Dropout(0)(x)

		Oneofkey_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer= l2(0.001))(x)
		
		OnehotNetwork = Model(Oneofkey_input,Oneofkey_output)
		
		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		OnehotNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
		'''
		pre_train_seq_path = 'model/pretrain/OnehotNetwork.h5'
		seq_model = load_model(pre_train_seq_path) 
		for l in range(0, 5):
			OnehotNetwork.get_layer(name = str(l)).set_weights(seq_model.get_layer(name = str(l)).get_weights()) 
			OnehotNetwork.get_layer(name = str(l)).trainable = False
		'''
	else:
		OnehotNetwork = load_model("model/"+str(train_time-1)+'model/OnehotNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/OnehotNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/OnehotNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/OnehotNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"onehotloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#onehotfitHistory = OnehotNetwork.fit(train_oneofkeyX,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_oneofkeyX,valY))
		onehotfitHistory = OnehotNetwork.fit(train_oneofkeyX,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_oneofkeyX,valY))
		OnehotNetwork = load_model("model/"+str(train_time)+'model/OnehotNetwork.h5')
		#plt.subplot(2,4,1)
		#plt.plot(onehotfitHistory.history["binary_accuracy"], label="OnehotNetwork_TrainAcc")
		#plt.plot(onehotfitHistory.history["loss"], label="OnehotNetwork_Trainloss")
		#plt.plot(onehotfitHistory.history["val_binary_accuracy"], label="OnehotNetwork_ValAcc")
		#plt.plot(onehotfitHistory.history["val_loss"], label="OnehotNetwork_Valloss")
		#plt.legend()
		#plt.savefig('model/'+str(train_time)+'model/onehotpic.jpg')
		
	return OnehotNetwork

def OtherNetwork(train_physicalXo,trainY,val_physicalXo,valY,
				train_time=None,compilemodels=None):
	
	physical_O_input = Input(shape=(train_physicalXo.shape[1],train_physicalXo.shape[2]))  #49*28=1372
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

	if (train_time==0):         
		x = core.Flatten()(physical_O_input)
		x = BatchNormalization()(x)
		
		x = Dense(256,init='glorot_normal',activation='relu',name="6")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		x = Dense(128,init='glorot_normal',activation='relu',name="7")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.1)(x)

		physical_O_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer= l2(0.001))(x)
		
		OtherNetwork = Model(physical_O_input,physical_O_output)
		
		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		OtherNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
		'''
		pre_train_O_path = 'model/pretrain/OtherNetwork.h5'
		O_model = load_model(pre_train_O_path) 
		for l in range(6, 8):
			OtherNetwork.get_layer(name = str(l)).set_weights(O_model.get_layer(name = str(l)).get_weights()) 
			OtherNetwork.get_layer(name = str(l)).trainable = False
		'''
	else:
		OtherNetwork = load_model("model/"+str(train_time-1)+'model/OtherNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/OtherNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/OtherNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/OtherNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Onetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#OfitHistory = OtherNetwork.fit(train_physicalXo,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer], class_weight={0:0.1,1:1},validation_data=(val_physicalXo,valY))                
		OfitHistory = OtherNetwork.fit(train_physicalXo,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer], class_weight={0:0.1,1:1},validation_data=(val_physicalXo,valY))                
		OtherNetwork = load_model("model/"+str(train_time)+'model/OtherNetwork.h5')
		#plt.subplot(2,4,2)
		#plt.plot(OfitHistory.history["binary_accuracy"], label="OtherNetwork_TrainAcc")
		#plt.plot(OfitHistory.history["loss"], label="OtherNetwork_Trainloss")
		#plt.plot(OfitHistory.history["val_binary_accuracy"], label="OtherNetwork_ValAcc")
		#plt.plot(OfitHistory.history["val_loss"], label="OtherNetwork_Valloss")
		#plt.legend()
		#plt.savefig('model/'+str(train_time)+'model/Opic.jpg')
	return OtherNetwork

def PhysicochemicalNetwork(train_physicalXp,trainY,val_physicalXp,valY,
						train_time=None,compilemodels=None):

	physical_P_input = Input(shape=(train_physicalXp.shape[1],train_physicalXp.shape[2]))  #49*46=2254 	
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)
	
	if (train_time==0): 
		x = core.Flatten()(physical_P_input)
		x = BatchNormalization()(x)

		x = Dense(512,init='glorot_normal',activation='relu',name="9")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		x = Dense(256,init='glorot_normal',activation='relu',name="10")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		x = Dense(128,init='glorot_normal',activation='relu',name="11")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.1)(x)
		
		physical_P_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)
		
		PhysicochemicalNetwork = Model(physical_P_input,physical_P_output)
		
		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		PhysicochemicalNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
		'''
		pre_train_P_path = 'model/pretrain/PhysicochemicalNetwork.h5'
		P_model = load_model(pre_train_P_path) 
		for l in range(9, 12):
			PhysicochemicalNetwork.get_layer(name = str(l)).set_weights(P_model.get_layer(name = str(l)).get_weights()) 
			PhysicochemicalNetwork.get_layer(name = str(l)).trainable = False
			print("wr")
		'''
	else:
		PhysicochemicalNetwork = load_model("model/"+str(train_time-1)+'model/PhysicochemicalNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/PhysicochemicalNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/PhysicochemicalNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/PhysicochemicalNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Pnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#PfitHistory = PhysicochemicalNetwork.fit(train_physicalXp,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXp,valY))                
		PfitHistory = PhysicochemicalNetwork.fit(train_physicalXp,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXp,valY))                
		PhysicochemicalNetwork = load_model("model/"+str(train_time)+'model/PhysicochemicalNetwork.h5')
		#plt.subplot(2,4,3)
		#plt.plot(PfitHistory.history["binary_accuracy"], label="PhysicochemicalNetwork_TrainAcc")
		#plt.plot(PfitHistory.history["loss"], label="PhysicochemicalNetwork_Trainloss")
		#plt.plot(PfitHistory.history["val_binary_accuracy"], label="PhysicochemicalNetwork_ValAcc")
		#plt.plot(PfitHistory.history["val_loss"], label="PhysicochemicalNetwork_Valloss")
		#plt.legend()    	
		#plt.savefig('model/'+str(train_time)+'model/Ppic.jpg')
	return PhysicochemicalNetwork

def HydrophobicityNetwork(train_physicalXh,trainY,val_physicalXh,valY,
						train_time=None,compilemodels=None):

	physical_H_input = Input(shape=(train_physicalXh.shape[1],train_physicalXh.shape[2]))  #49*149=7301
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

	if (train_time==0):
		x = core.Flatten()(physical_H_input)
		x = BatchNormalization()(x)

		x = Dense(1024,init='glorot_normal',activation='relu',name="13")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		x = Dense(512,init='glorot_normal',activation='relu',name="14")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		x = Dense(256,init='glorot_normal',activation='relu',name="15")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)

		x = Dense(128,init='glorot_normal',activation='relu',name="16")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.1)(x)

		physical_H_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)
		
		HydrophobicityNetwork = Model(physical_H_input,physical_H_output)
		
		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		HydrophobicityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
		'''
		pre_train_H_path = 'model/pretrain/HydrophobicityNetwork.h5'
		H_model = load_model(pre_train_H_path) 
		for l in range(13, 17):
			HydrophobicityNetwork.get_layer(name = str(l)).set_weights(H_model.get_layer(name = str(l)).get_weights()) 
			HydrophobicityNetwork.get_layer(name = str(l)).trainable = False
		'''
	else:
		HydrophobicityNetwork = load_model("model/"+str(train_time-1)+'model/HydrophobicityNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/HydrophobicityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/HydrophobicityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/HydrophobicityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Hnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#HfitHistory = HydrophobicityNetwork.fit(train_physicalXh,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXh,valY))                
		HfitHistory = HydrophobicityNetwork.fit(train_physicalXh,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXh,valY))                
		HydrophobicityNetwork = load_model("model/"+str(train_time)+'model/HydrophobicityNetwork.h5')
		#plt.subplot(2,4,4)
		#plt.plot(HfitHistory.history["binary_accuracy"], label="HydrophobicityNetwork_TrainAcc")
		#plt.plot(HfitHistory.history["loss"], label="HydrophobicityNetwork_Trainloss")
		#plt.plot(HfitHistory.history["val_binary_accuracy"], label="HydrophobicityNetwork_ValAcc")
		#plt.plot(HfitHistory.history["val_loss"], label="HydrophobicityNetwork_Valloss")
		#plt.legend()    	
		#plt.savefig('model/'+str(train_time)+'model/Hpic.jpg')              	
	return HydrophobicityNetwork

def CompositionNetwork(train_physicalXc,trainY,val_physicalXc,valY,
					train_time=None,compilemodels=None):

	physical_C_input = Input(shape=(train_physicalXc.shape[1],train_physicalXc.shape[2]))  #49*24=1176
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="18")(physical_C_input)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)
		
		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="19")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)

		physical_C_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		CompositionNetwork = Model(physical_C_input,physical_C_output)
		
		optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		#optimization='Nadam'
		CompositionNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
		
		pre_train_C_path = 'model/pretrain/CompositionNetwork.h5'
		C_model = load_model(pre_train_C_path) 
		for l in range(18, 20):
			CompositionNetwork.get_layer(name = str(l)).set_weights(C_model.get_layer(name = str(l)).get_weights()) 
			CompositionNetwork.get_layer(name = str(l)).trainable = False
		
	else:
		CompositionNetwork = load_model("model/"+str(train_time-1)+'model/CompositionNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/CompositionNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/CompositionNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/CompositionNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Cnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#CfitHistory = CompositionNetwork.fit(train_physicalXc,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXc,valY))                
		CfitHistory = CompositionNetwork.fit(train_physicalXc,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXc,valY))                
		CompositionNetwork = load_model("model/"+str(train_time)+'model/CompositionNetwork.h5')
		#plt.subplot(2,4,5)
		#plt.plot(CfitHistory.history["binary_accuracy"], label="CompositionNetwork_TrainAcc")
		#plt.plot(CfitHistory.history["loss"], label="CompositionNetwork_Trainloss")
		#plt.plot(CfitHistory.history["val_binary_accuracy"], label="CompositionNetwork_ValAcc")
		#plt.plot(CfitHistory.history["val_loss"], label="CompositionNetwork_Valloss")
		#plt.legend()    	
		#plt.savefig('model/'+str(train_time)+'model/Cpic.jpg')      	
	return CompositionNetwork

def BetapropensityNetwork(train_physicalXb,trainY,val_physicalXb,valY,
						train_time=None,compilemodels=None):

	physical_B_input = Input(shape=(train_physicalXb.shape[1],train_physicalXb.shape[2]))  #49*37=1813
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="21")(physical_B_input)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="22")(x)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="23")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)

		physical_B_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		BetapropensityNetwork = Model(physical_B_input,physical_B_output)
		
		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		BetapropensityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
		'''
		pre_train_B_path = 'model/pretrain/BetapropensityNetwork.h5'
		B_model = load_model(pre_train_B_path) 
		for l in range(21, 24):
			BetapropensityNetwork.get_layer(name = str(l)).set_weights(B_model.get_layer(name = str(l)).get_weights()) 
			BetapropensityNetwork.get_layer(name = str(l)).trainable = False
		'''
	else:
		BetapropensityNetwork = load_model("model/"+str(train_time-1)+'model/BetapropensityNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/BetapropensityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/BetapropensityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/BetapropensityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Bnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#BfitHistory = BetapropensityNetwork.fit(train_physicalXb,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXb,valY))                
		BfitHistory = BetapropensityNetwork.fit(train_physicalXb,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXb,valY))                
		BetapropensityNetwork = load_model("model/"+str(train_time)+'model/BetapropensityNetwork.h5')
		#plt.subplot(2,4,6)
		#plt.plot(BfitHistory.history["binary_accuracy"], label="BetapropensityNetwork_TrainAcc")
		#plt.plot(BfitHistory.history["loss"], label="BetapropensityNetwork_Trainloss")
		#plt.plot(BfitHistory.history["val_binary_accuracy"], label="BetapropensityNetwork_ValAcc")
		#plt.plot(BfitHistory.history["val_loss"], label="BetapropensityNetwork_Valloss")
		#plt.legend()    	
		#plt.savefig('model/'+str(train_time)+'model/Bpic.jpg')      	
	return BetapropensityNetwork

def AlphaturnpropensityNetwork(train_physicalXa,trainY,val_physicalXa,valY,
							train_time=None,compilemodels=None):

	physical_A_input = Input(shape=(train_physicalXa.shape[1],train_physicalXa.shape[2]))  #49*118=5782
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="25")(physical_A_input)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="26")(x)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="27")(x)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(51,7,init='glorot_normal',W_regularizer= l2(0),border_mode="same",name="28")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_A_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		AlphaturnpropensityNetwork = Model(physical_A_input,physical_A_output)
		
		#optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
		optimization='Nadam'
		AlphaturnpropensityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
		'''
		pre_train_A_path = 'model/pretrain/AlphaturnpropensityNetwork.h5'
		A_model = load_model(pre_train_A_path) 
		for l in range(25, 29):
			AlphaturnpropensityNetwork.get_layer(name = str(l)).set_weights(A_model.get_layer(name = str(l)).get_weights()) 
			AlphaturnpropensityNetwork.get_layer(name = str(l)).trainable = False
		'''
	else:
		AlphaturnpropensityNetwork = load_model("model/"+str(train_time-1)+'model/AlphaturnpropensityNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/AlphaturnpropensityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/AlphaturnpropensityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/AlphaturnpropensityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Anetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		#AfitHistory = AlphaturnpropensityNetwork.fit(train_physicalXa,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXa,valY))                
		AfitHistory = AlphaturnpropensityNetwork.fit(train_physicalXa,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight={0:0.1,1:1},validation_data=(val_physicalXa,valY))                
		AlphaturnpropensityNetwork = load_model("model/"+str(train_time)+'model/AlphaturnpropensityNetwork.h5')
		#plt.subplot(2,4,7)
		#plt.plot(AfitHistory.history['binary_accuracy'], label="AlphaturnpropensityNetwork_TrainAcc")
		#plt.plot(AfitHistory.history['loss'], label="AlphaturnpropensityNetwork_Trainloss")
		#plt.plot(AfitHistory.history['val_binary_accuracy'], label="AlphaturnpropensityNetwork_ValAcc")
		#plt.plot(AfitHistory.history['val_loss'], label="AlphaturnpropensityNetwork_Valloss")
		#plt.legend()    	
		#plt.savefig('model/'+str(train_time)+'model/Apic.jpg')       	
	return AlphaturnpropensityNetwork