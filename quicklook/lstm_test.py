
import numpy as np
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Masking, TimeDistributed
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from fetch_columns import genData,fetchFluxTimeseries,fetchPosNegTimeseries


def simpleLSTM():
  X_pre_train,Y_pre_train,X_pre_test,Y_pre_test = genData("old/positive/","old/negative/",0.9)
  #convert Y to one-hot; get dimensions right
  X_train = np.expand_dims(X_pre_train,axis=2)
  X_test = np.expand_dims(X_pre_test,axis=2)
  Y_train = np.where(Y_pre_train>0,[0,1],[1,0])
  Y_test = np.where(Y_pre_test>0,[0,1],[1,0])
  
  
  model = Sequential()
  model.add(Masking(mask_value=0.,input_shape=(X_train.shape[1],1))) #uneven length
  model.add(LSTM(units=20,return_sequences=True))
  model.add(LSTM(units=100,return_sequences=True))
  model.add(LSTM(units=20))
  #model.add(Dropout(rate=0.2))
  model.add(Dense(units=2,activation='softmax'))
  #apparently stateful=True might also help for the LSTM layers?
  #return_sequences=True for every layer except the last (and maybe the last too)
  
  model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

  model.fit(X_train,Y_train,batch_size=10,epochs=30,validation_data=(X_test,Y_test))


#X, Y = fetchPosNegTimeseries("positive/","negative/")
#frac_train = 0.9
#size = int(frac_train*X.shape[0])
#X_train = np.expand_dims(X[:size,:],axis=2)
#X_test = np.expand_dims(X[size:,:],axis=2)
#Y_train = Y[:size,:,:]
#Y_test = Y[size:,:,:]


def timeseriesLSTM(layers):
  fluxes, labels = fetchFluxTimeseries("positive/")
  frac_train = 0.9
  size = int(frac_train*fluxes.shape[0])
  X_train = np.expand_dims(fluxes[:size,:],axis=2)
  X_test = np.expand_dims(fluxes[size:,:],axis=2)
  Y_train = np.expand_dims(labels[:size,:],axis=2)
  Y_test = np.expand_dims(labels[size:,:],axis=2)

  model = Sequential()
  model.add(Masking(mask_value=0.,input_shape=(fluxes.shape[1],1))) #uneven length
  for i in range(len(layers)):
    model.add(LSTM(units=layers[i],return_sequences=True))
  model.add(TimeDistributed(Dense(units=1,activation="sigmoid")))

  model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

  #should batch size be 1?
  model.fit(X_train,Y_train,batch_size=10,epochs=5,validation_data=(X_test,Y_test))
  return model


def classifierLSTM(layers):


  model = Sequential()
  model.add(Masking(mask_value=0.,input_shape=(X.shape[1],1))) #uneven length
  for i in range(len(layers)):
    model.add(LSTM(units=layers[i],return_sequences=True))
  model.add(TimeDistributed(Dense(units=2,activation="sigmoid")))

  model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

  #should batch size be 1?
  model.fit(X_train,Y_train,batch_size=1,epochs=5,validation_data=(X_test,Y_test))
  return model

def plot(model,tbl):
  x = tbl["SAP_FLUX"]
  y = tbl["IN_TRANSIT"]
  time = tbl["TIME"]
  pad = np.zeros((1,130,1)) # whatever length of timeseries the model accepts (there doesn't seem to be a way to get this except trial/error)
  pad[0,:x.shape[0],0]=x
  output = model.predict(pad)
  p_transit = output[0,:x.shape[0],0]
  plt.gcf().clear()
  plt.scatter(time,x,c=y,cmap=plt.get_cmap("bwr"))
  plt.title("Light Curve")
  plt.savefig("lightcurve.png")
  plt.gcf().clear()
  plt.scatter(time,p_transit,c=y,cmap=plt.get_cmap("bwr"))
  plt.title("LSTM Output")
  plt.savefig("p_transit.png")





