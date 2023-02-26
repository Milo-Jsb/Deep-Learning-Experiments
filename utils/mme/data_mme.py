import pandas as pd
import numpy  as np
import random as rn
import torch
import csv


import os

import utils.data_split    as data_split
import utils.data_handling
import utils.data_loaders  as data_loaders
import utils.preprocess    as preprocess


def split_mme_data(data_type, data_subtype, ramdon_idx ,X, y, timesX, obj_id, labels, path):
    # Source
    # -> All data labeled for  training
    if (data_type == "source"):
        # Creamos el directorio para guardarlo
        path = path+"data_train/"

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f"El directorio {path} ya existe")

        # Cambios en la forma de los datos
        X_n = X.transpose(0, 2, 1)
        y_n = y

        # Pasamos a one hot
        y_n = np.array(y_n, dtype="int64")

        # Generamos formato onehot
        tensor         = torch.from_numpy(y_n)
        num_categories = int(tensor.max().item() +1) 
        one_hot        = torch.nn.functional.one_hot(tensor,num_categories).type(torch.float64)
        y_n            = one_hot.numpy()

        # Guardamos los datos 
        np.save(path+data_type+"_timesX.npy", timesX)
        np.save(path+data_type+"_X.npy", X_n)
        np.save(path+data_type+"_y.npy",y_n)
        np.save(path+data_type+"_labels.npy", labels)
        np.save(path+data_type+"_object_ids.npy", obj_id)
        np.save(path+"ramdon_ids.npy", ramdon_idx)

        # Contabilizamos
        num_sne = len(obj_id)
        print(100*"_")
        print(f"Numero de SNs en {data_type} : {num_sne}") 
        print(100*"_")
        return X_n, y_n, labels 
    
    
    elif (data_type == "target"):
        
        path = path+f"{data_subtype}/"
        # Creamos el directorio para guardarlo        
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f"El directorio {path} ya existe")

        # Extraccion de los datos
        timesX_n = timesX[ramdon_idx, :]
        X_n      = X[ramdon_idx, :, :]
        X_n      = X_n.transpose(0, 2, 1)
        y_n      = y[ramdon_idx, :]
        obj_id_n = obj_id[ramdon_idx]
        labels_n = labels[ramdon_idx]

        # Pasamos a one hot
        y_n = np.array(y_n, dtype="int64")

        # Generamos formato onehot
        tensor         = torch.from_numpy(y_n)
        num_categories = int(tensor.max().item() +1) 
        one_hot        = torch.nn.functional.one_hot(tensor,num_categories).type(torch.float64)
        y_n            = one_hot.numpy()

        # Guardamos los datos 
        np.save(path+data_type+"_timesX.npy", timesX_n)
        np.save(path+data_type+"_X.npy", X_n)
        np.save(path+data_type+"_y.npy",y_n)
        np.save(path+data_type+"_labels.npy", labels_n)
        np.save(path+data_type+"_object_ids.npy", obj_id_n)
        np.save(path+"ramdon_ids.npy", ramdon_idx)


        # Contabilizamos
        num_sne = len(obj_id_n)
        print(100*"_")
        print(f"Numero de SNs en {data_type} {data_subtype}: {num_sne}")
        print(100*"_")
        return X_n, y_n, labels_n 

       
def get_dataset(CONFIG):
    
    # Source path
    path_source   = './data/{}/semisupervised_mme/data_train/'.format(CONFIG.source)

    # Target paths
    path_lab      = './data/{}/data_model_alerce/semisupervised_mme/label_train/'.format(CONFIG.target)
    path_unl      = './data/{}/data_model_alerce/semisupervised_mme/unlabel/'.format(CONFIG.target)
    path_val      = './data/{}/data_model_alerce/semisupervised_mme/label_val/'.format(CONFIG.target)
    path_test     = './data/{}/data_model_alerce/semisupervised_mme/test/'.format(CONFIG.target)
    
    # Open source and target data
    source_data      = utils.data_handling.load_npy_files(path_source,
                                                        "source_X.npy",
                                                        "source_y.npy")
    
    target_data_lab  = utils.data_handling.load_npy_files(path_lab, 
                                                        "target_X.npy",
                                                        "target_y.npy")
    
    target_data_unl  = utils.data_handling.load_npy_files(path_unl, 
                                                        "target_X.npy",
                                                        "target_y.npy")
    
    target_data_val  = utils.data_handling.load_npy_files(path_val, 
                                                        "target_X.npy",
                                                        "target_y.npy")
    
    target_data_test = utils.data_handling.load_npy_files(path_test,
                                                        "target_X.npy",
                                                        "target_y.npy")
    
    # Create data loaders   
    data_loader = data_loaders.PhotometryDataLoaders(source_data, 
                                                     target_data_lab, 
                                                     target_data_unl, 
                                                     target_data_val,
                                                     target_data_test,
                                                     one_hot=False,
                                                     batch_size=CONFIG.batch_size,
                                                     collate_fn=False,
                                                     weight_norm=False
                                                     )
    
    dataloader_source      = data_loader.source_set
    dataloader_target_lab  = data_loader.target_set_lab
    dataloader_target_unl  = data_loader.target_set_unl
    dataloader_target_val  = data_loader.target_set_val
    dataloader_target_test = data_loader.target_data_test

    # Asegura el input y num classes desde la data
    CONFIG.input_size  = dataloader_source.dataset.data[0].size(1)
    CONFIG.num_classes = data_loader.nb_classes
    print(100*"_")
    print(data_loader)
    print(f"Input-size:        {CONFIG.input_size}")
    print(f"Numero de clases:  {CONFIG.num_classes}")
    print(100*"_")

    return dataloader_source, dataloader_target_lab, dataloader_target_unl, dataloader_target_val, dataloader_target_test