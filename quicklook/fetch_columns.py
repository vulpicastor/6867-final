import astropy.io.ascii
import numpy as np
import sklearn
import os




def fetchFluxVectors(path): #returns numpy array where rows are data points; 
    #each point is a vector of fluxes, currently padded with zeros to the length of the longest one
    fluxes = []
    for f in os.listdir(path):
        #print(f)
        if f[-5:] == ".ecsv":
            tbl = astropy.io.ascii.read(path+"/"+f)
            #print(tbl)
            col = np.array(tbl["SAP_FLUX"])
            if not np.isnan(col).any(): #ignore timeseries with NaNs
                fluxes.append(col)
    maxlen = max(len(col) for col in fluxes)
    for i in range(len(fluxes)):
        #"normalize" by subtracting last
        #fluxes[i] -= fluxes[i][-1]
        padded = np.zeros(maxlen)
        padded[:len(fluxes[i])]=fluxes[i]
        fluxes[i] = padded
    return np.array(fluxes)

#TODO:find a better way than zero padding (maybe modify quicklook so all the pieces have ~the same length?)


def fetchFluxTimeseries(path): #returns numpy array where rows are data points; 
    #each point is a vector of fluxes, currently padded with zeros to the length of the longest one
    fluxes = []
    labels = []
    for f in os.listdir(path):
        #print(f)
        if f[-5:] == ".ecsv":
            tbl = astropy.io.ascii.read(path+"/"+f)
            #print(tbl)
            col = np.array(tbl["SAP_FLUX"])
            lab = np.array(tbl["IN_TRANSIT"])
            if len(col)!=0 and not np.isnan(col).any(): #ignore timeseries with NaNs
                fluxes.append(col)
                labels.append(lab)
    maxlen = max(len(col) for col in fluxes)
    for i in range(len(fluxes)):
        #"normalize" by subtracting last
        #fluxes[i] -= fluxes[i][-1]
        padded = np.zeros(maxlen)
        padded2 = np.copy(padded)
        padded[:len(fluxes[i])]=fluxes[i]
        padded2[:len(labels[i])]=labels[i]
        fluxes[i] = padded
        labels[i] = padded2
    return (np.array(fluxes),np.array(labels))

def fetchPosNegTimeseries(pathpos,pathneg): #returns numpy array where rows are data points; 
    #each point is a vector of fluxes, currently padded with zeros to the length of the longest one
    fluxes = []
    labels = []
    falses = []
    for f in os.listdir(pathpos):
        #print(f)
        if f[-5:] == ".ecsv":
            tbl = astropy.io.ascii.read(pathpos+"/"+f)
            #print(tbl)
            col = np.array(tbl["SAP_FLUX"])
            lab = np.array(tbl["IN_TRANSIT"])
            fal = np.zeros(lab.shape)
            if not np.isnan(col).any(): #ignore timeseries with NaNs
                fluxes.append(col)
                labels.append(lab)
                falses.append(fal)
    for f in os.listdir(pathneg):
        #print(f)
        if f[-5:] == ".ecsv":
            tbl = astropy.io.ascii.read(pathneg+"/"+f)
            #print(tbl)
            col = np.array(tbl["SAP_FLUX"])
            lab = np.array(tbl["IN_TRANSIT"])
            fal = np.array(tbl["EB_injection"])
            if not np.isnan(col).any(): #ignore timeseries with NaNs
                fluxes.append(col)
                labels.append(lab)
                falses.append(fal)
    maxlen = max(len(col) for col in fluxes)
    for i in range(len(fluxes)):
        #"normalize" by subtracting last
        #fluxes[i] -= fluxes[i][-1]
        padded = np.zeros(maxlen)
        padded2 = np.copy(padded)
        padded3 = np.copy(padded)
        padded[:len(fluxes[i])]=fluxes[i]
        padded2[:len(labels[i])]=labels[i]
        padded3[:len(falses[i])]=falses[i]
        fluxes[i] = padded
        labels[i] = padded2
        falses[i] = padded3
    combined = np.concatenate((np.array(fluxes),np.array(labels),np.array(falses)),axis=1)
    np.random.shuffle(combined)
    X = combined[:,:maxlen]
    Y = np.dstack((combined[:,maxlen:2*maxlen],combined[:,2*maxlen:]))
    return (X,Y)


def genData(path_pos, path_neg, frac_train=0.8): #uses first frac_train of data as train; rest as test
    flux_pos = fetchFluxVectors(path_pos)
    Y_pos = np.ones((flux_pos.shape[0],1))
    flux_neg = fetchFluxVectors(path_neg)
    Y_neg = np.ones((flux_neg.shape[0],1))*-1

    #pad positive and negative to same length
    maxlen = max(flux_pos.shape[1],flux_neg.shape[1])
    pospad = np.zeros((flux_pos.shape[0],maxlen))
    negpad = np.zeros((flux_neg.shape[0],maxlen))
    pospad[:,:flux_pos.shape[1]] = flux_pos
    negpad[:,:flux_neg.shape[1]] = flux_neg
    flux_pos = pospad
    flux_neg = negpad
    
    size_pos=int(frac_train*flux_pos.shape[0])
    X_pos_train = flux_pos[:size_pos,:]
    X_pos_test = flux_pos[size_pos:,:]
    Y_pos_train = Y_pos[:size_pos,:]
    Y_pos_test = Y_pos[size_pos:,:]

    size_neg=int(frac_train*flux_neg.shape[0])
    X_neg_train = flux_neg[:size_neg,:]
    X_neg_test = flux_neg[size_neg:,:]
    Y_neg_train = Y_neg[:size_neg,:]
    Y_neg_test = Y_neg[size_neg:,:]

    X_pre_train = np.concatenate((X_pos_train,X_neg_train),axis=0)
    Y_pre_train = np.concatenate((Y_pos_train,Y_neg_train),axis=0)
    X_test = np.concatenate((X_pos_test,X_neg_test),axis=0)
    Y_test = np.concatenate((Y_pos_test,Y_neg_test),axis=0)

    #randomize order of training data
    combined = np.concatenate((Y_pre_train,X_pre_train),axis=1)
    np.random.shuffle(combined)
    Y_train = combined[:,[0]]
    X_train = combined[:,1:]

    return (X_train,Y_train,X_test,Y_test)

def runPCA(path_pos,path_neg,dim=2):
    X,Y,_,_ = genData(path_pos,path_neg,1)
    pca = sklearn.decomposition.PCA(n_components=dim)
    X_coll = pca.fit_transform(X)
    return (X_coll,Y)

def svmFluxVectors(path_pos,path_neg,frac_train,type="linear",pca=None):
    X_train,Y_train,X_test,Y_test = genData(path_pos,path_neg,frac_train)
    #prints out mean accuracy
    svm = sklearn.svm.SVC(kernel=type)
    svm.fit(X_train,Y_train)
    print("Mean SVM accuracy on training set: {}".format(svm.score(X_train,Y_train)))
    print("Mean SVM accuracy on test set: {}".format(svm.score(X_test,Y_test)))

