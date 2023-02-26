"""
______________________________________________________________________________________________________________
Functions and utilities to split the data:
______________________________________________________________________________________________________________
- find_all_idx:        
    -> Objective:   Find the ids present inside a .csv separation file. 
    -> Used:        RAPID model.
    -> Input:       oid1 = [all object id], oid2 = [training_id], [validation_id], [test_id]
    -> Return:      [idx]  

- Split-nonMeta_Data:  
    -> Objective:   Splitig data without any metadata in training validation and test with a define number
                    of splits using K-fold.
    -> Used:        RAPID model.
    -> Input:       num_splits, X, y, timesX, object_ids, labels, save =""
    -> Return:      X_Train, X_Val, y_Train, y_Val, X_test, y_test, timesX_test

- Split_Meta_Data:  
    -> Objective:   Splitig data with metadata in training validation and test with a define number of
                    splits.
    -> Used:        RAPID model.
    -> Input:       num_splits, metadata, X, y, timesX, object_ids, labels ,save ="" 
    -> Return:      X_train, X_val, y_train, y_val, X_test, y_test, timesX_test
______________________________________________________________________________________________________________
"""

# Modulos ####################################################################################################
import pandas as pd
import numpy  as np
import os     as o
import torch

# Functions ##################################################################################################
from sklearn.model_selection import KFold

##############################################################################################################
def find_all_idxs(oid1,oid2):

    idxs=np.where(np.isin(oid1,oid2))[0]
    
    return idxs

############################################################################################################## 
def Split_nonMeta_Data(num_splits, X, y, timesX, object_ids, labels, save =""):

    # Lists to save the data
    X_Train        = []          
    X_Val          = []          
    y_Train        = []          
    y_Val          = []          
    
    # Length for the files
    total_lengh_oid        = len(object_ids)                         
    full_training_lenght   = int(total_lengh_oid * 0.9)              
    test_length            = total_lengh_oid-full_training_lenght    

    # Summary
    print(100*"_")
    print(f"Preliminary sizes:")
    print(100*"_")
    print(f"Trainig (train+val) | {full_training_lenght}")
    print(f"Test    (just test) | {test_length}"   )
    print(100*"_")

    # Ramdon index selection
    ramdon_index = np.random.choice(np.arange(0, total_lengh_oid), size=test_length, replace = False)

    # Test data
    timesX_test          = timesX[ramdon_index, :]
    
    X_test               = X[ramdon_index, :, :]
    X_test               = X_test.transpose(0, 2, 1)

    y_test               = y[ramdon_index, :]
    y_test = np.array(y_test, dtype="int64")
    
    labels_test          = labels[ramdon_index]
    object_ids_test      = object_ids[ramdon_index]

    # Onehot
    tensor         = torch.from_numpy(y_test)
    num_categories = int(tensor.max().item() +1) 
    one_hot        = torch.nn.functional.one_hot(tensor,num_categories).type(torch.float64)
    y_test         = one_hot.numpy()

    # path test
    test_save = save + "/data_test/"

    # Save data
    np.save(test_save+"timesX.npy", timesX_test)
    np.save(test_save+"X.npy", X_test)
    np.save(test_save+"y.npy",y_test)
    np.save(test_save+"labels.npy", labels_test)
    np.save(test_save+"object_ids.npy", object_ids_test)

    # Sumary
    print(100*"_")
    print("Shape Test:")
    print(100*"_")
    print(f"Xtime:        | {timesX_test.shape}")
    print(f"X:            | {X_test.shape}")
    print(f"y:            | {y_test.shape}")
    print(f"labels:       | {labels_test.shape}")
    print(100*"_")

    # New sets without test data
    timesX_new          = np.delete(timesX,ramdon_index, axis = 0)
    X_new               = np.delete(X,ramdon_index, axis = 0)
    y_new               = np.delete(y,ramdon_index, axis = 0)
    labels_new          = np.delete(labels,ramdon_index, axis = 0)
    object_ids_new      = np.delete(object_ids,ramdon_index, axis = 0)

    # Kfold
    kf = KFold(n_splits=num_splits, random_state = 0,shuffle=True)

    index_train_fold = []
    index_val_fold   = [] 
    
    # Split
    print(100*"_")
    print(kf)
    print(100*"_")
    for i, (train_ids, val_ids) in enumerate(kf.split(X_new)):
        # Info
        print(f"Fold {i}:")
        print(f"  Train: index={train_ids}")
        print(f"  Val:   index={val_ids}")

        # Create fold directory
        folder = f"fold_{i}"
        p_dirt = f"./{save}data_train/"
        if not o.path.exists(p_dirt+folder):
            o.makedirs(p_dirt+folder)
        else:
            print(f"Folder {p_dirt} already exists")

        # Training Data
        timesX_train          = timesX_new[train_ids, :]
        
        X_train               = X_new[train_ids, :, :]
        X_train               = X_train.transpose(0, 2, 1)
        
        y_train               = y_new[train_ids, :]
        y_train               = np.array(y_train, dtype="int64")
        
        labels_train          = labels_new[train_ids]
        object_ids_train      = object_ids_new[train_ids]
        
        # Onehot
        tensor         = torch.from_numpy(y_train)
        num_categories = int(tensor.max().item() +1) 
        one_hot        = torch.nn.functional.one_hot(tensor,num_categories).type(torch.float64)
        y_train        = one_hot.numpy()

        # Summary
        print(100*"_")
        print(f"Shape Training")
        print(100*"_")
        print(f"Xtime:        | {timesX_train.shape}")
        print(f"X:            | {X_train.shape}")
        print(f"y:            | {y_train.shape}")
        print(f"labels:       | {labels_train.shape}")
        print(100*"_")

        # Create data directory
        folder = "train/"
        p_dirt = f"./{save}data_train/fold_{i}/"
        if not o.path.exists(p_dirt+folder):
            o.makedirs(p_dirt+folder)
        else:
            print(f"Folder {p_dirt} already exists")

        # Save data
        full_path = p_dirt+folder
        np.save(full_path+"timesX.npy", timesX_train)
        np.save(full_path+"X.npy", X_train)
        np.save(full_path+"y.npy",y_train)
        np.save(full_path+"labels.npy", labels_train)
        np.save(full_path+"object_ids.npy", object_ids_train)

        # Append on the list
        X_Train.append(X_train)
        y_Train.append(y_train)

        # Validation Data
        timesX_val          = timesX_new[val_ids, :]
        
        X_val               = X_new[val_ids, :, :]
        X_val               = X_val.transpose(0, 2, 1)
        
        y_val               = y_new[val_ids, :]
        y_val               = np.array(y_val, dtype="int64")

        labels_val          = labels_new[val_ids]
        object_ids_val      = object_ids_new[val_ids]

        # Onehot
        tensor         = torch.from_numpy(y_val)
        num_categories = int(tensor.max().item() +1) 
        one_hot        = torch.nn.functional.one_hot(tensor,num_categories).type(torch.float64)
        y_val          = one_hot.numpy()

        # Imprimimos
        print(100*"_")
        print("Shape Validation:")
        print(100*"_")
        print(f"Xtime:        | {timesX_val.shape}")
        print(f"X:            | {X_val.shape}")
        print(f"y:            | {y_val.shape}")
        print(f"labels:       | {labels_val.shape}")
        print(100*"_")

        # Create data folder
        folder = "validation/"
        p_dirt = f"./{save}data_train/fold_{i}/"
        if not o.path.exists(p_dirt+folder):
            o.makedirs(p_dirt+folder)
        else:
            print(f"Folder {p_dirt} already exists")

        # Save data
        full_path = p_dirt+folder
        np.save(full_path+"timesX.npy", timesX_val)
        np.save(full_path+"X.npy", X_val)
        np.save(full_path+"y.npy",y_val)
        np.save(full_path+"labels.npy", labels_val)
        np.save(full_path+"object_ids.npy", object_ids_val)

        # Append on the list
        X_Val.append(X_val)
        y_Val.append(y_val)
    
    # Full split training data
    X_Train = np.array(X_Train)
    y_Train = np.array(y_Train, dtype="float64")

    # Full split validation data
    X_Val = np.array(X_Val)
    y_Val = np.array(y_Val, dtype="float64")
    
    return X_Train, X_Val, y_Train, y_Val, X_test, y_test, timesX_test

##############################################################################################################
def Split_Meta_Data(num_splits, metadata, X, y, timesX, object_ids, labels ,save =""):
    
    # We define the number of folds
    n_splits       = num_splits  

    # List to save data
    train_oid      = []
    valid_oid      = []
    X_train        = []
    X_val          = []
    y_train        = []
    y_val          = []

    # Re-shape X data
    Xt = X.transpose(0, 2, 1)

    # We extract the partition and the fold respectively
    for i in range(n_splits):
        
        train_oid.append(metadata[metadata["partition"] == "training_"+str(i)]["oid"].unique())
        valid_oid.append(metadata[metadata["partition"] == "validation_"+str(i)]["oid"].unique())

    # We extract test id
    test_oid = metadata[metadata["partition"] == "test"]["oid"].unique()

    # Find the wanted ids
    for i in range(n_splits):

        idxs_train      = find_all_idxs(object_ids,train_oid[i])
        idxs_validation = find_all_idxs(object_ids,valid_oid[i])
        
        # Save training data
        X_train.append(Xt[idxs_train])
        y_train.append(y[idxs_train])
        
        # Save validation data
        X_val.append(Xt[idxs_validation])
        y_val.append(y[idxs_validation])

        # Create directory
        if (save != ""):
            path_train = save+f"data_train/fold_{i}/train/"
            path_val   = save+f"data_train/fold_{i}/validation/"
            
            if not o.path.exists(path_train):
                o.makedirs(path_train)
            else:
                print(f"Folder {path_train} already exist.")
            
            np.save(path_train + "timesX.npy"    , timesX[idxs_train])
            np.save(path_train + "X.npy"         , Xt[idxs_train])
            np.save(path_train + "y.npy"         , y[idxs_train])
            np.save(path_train + "labels.npy"    , labels[idxs_train])
            np.save(path_train + "object_ids.npy", object_ids[idxs_train])

            if not o.path.exists(path_val):
                o.makedirs(path_val)
            else:
                print(f"Folder {path_val} already exist.")
            
            np.save(path_val + "timesX.npy"    , timesX[idxs_validation])
            np.save(path_val + "X.npy"         , Xt[idxs_validation])
            np.save(path_val + "y.npy"         , y[idxs_validation])
            np.save(path_val + "labels.npy"    , labels[idxs_validation])
            np.save(path_val + "object_ids.npy", object_ids[idxs_validation])

    # Append training data on list
    X_train = np.array([X_train[0],X_train[1],X_train[2],X_train[3],X_train[4]], dtype="float64")
    y_train = np.array([y_train[0],y_train[1],y_train[2],y_train[3],y_train[4]], dtype="float64")

    # Append validation data on list
    X_val = np.array([X_val[0],X_val[1],X_val[2],X_val[3],X_val[4]], dtype="float64")
    y_val = np.array([y_val[0],y_val[1],y_val[2],y_val[3],y_val[4]], dtype="float64")
        
    # Find test data ids
    idxs_test = find_all_idxs(object_ids,test_oid)

    # Extract test data
    X_test            = Xt[idxs_test]
    y_test            = y[idxs_test]
    timesX_test       = timesX[idxs_test]
    labels_test       = labels[idxs_test]
    object_ids_test   = object_ids[idxs_test]

    X_test = np.array(X_test, dtype = "float64")
    y_test = np.array(y_test, dtype = "float64")

    # Create directory
    if (save != ""):
        
        path_test  = save+"data_test/static/"

        if not o.path.exists(path_test):
            o.makedirs(path_test)
        else:
            print(f"El directorio {path_test} ya existe")
        
        # Save
        np.save(path_test+"timesX.npy", timesX_test)
        np.save(path_test+"X.npy", X_test)
        np.save(path_test+"y.npy",y_test)
        np.save(path_test+"labels.npy", labels_test)
        np.save(path_test+"object_ids.npy", object_ids_test)
        
    # Summary
    print(100*"_")
    print("Shape Training:")
    print(100*"_")
    print(f"X:                  | {X_train.shape}")
    print(f"Y:                  | {y_train.shape}")
    print(100*"_")

    print(100*"_")
    print("Shape Validation:")
    print(100*"_")
    print(f"X:                  | {X_val.shape}")
    print(f"y:                  | {y_val.shape}")
    print(100*"_")

    print(100*"_")
    print("Shape Test:")
    print(100*"_")
    print(f"X:                  | {X_test.shape}")
    print(f"Y:                  | {y_test.shape}")
    print(100*"_")

    return X_train, X_val, y_train, y_val, X_test, y_test, timesX_test

##############################################################################################################