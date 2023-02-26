""" 
Funciones que limpian y normalizan los datos.
Jorge: Agregue algunas funciones para preparar los programas de clasificacion
"""

import pandas as pd
import numpy as np
import random
import glob

import torch
pd.options.mode.chained_assignment = None  # default='warn'


def normalize_band(df, tpo_name, flux_name, error_name):
    df[tpo_name] = (df[tpo_name] - df[tpo_name].min()) / (df[tpo_name].max() - df[tpo_name].min())

    for flux, error in zip(flux_name, error_name):
        df[flux] = (df[flux] - df[flux].min()) / (df[flux].max() - df[flux].min())
        df[error] = (df[error] - df[error].min()) / (df[error].max() - df[error].min())

    return df


def cleaning_data(df_ligthcurves, n_det_min, tpo_name, name_col_flux, name_col_error, normalize=False):
    
    """Limpia los datos y normaliza los flujos""" 
    
    # data
    df_ligthcurves = df_ligthcurves.dropna() # Eliminación de datos NAN
    list_ids = df_ligthcurves.oid.unique() 

    # outuput
    lc_pos = []
    id_sn_clean = []

    for id_sn in list_ids:
        filter_lc = df_ligthcurves[df_ligthcurves.oid == id_sn]
        bands_name = filter_lc.fid.unique()
        bands_name = ['g', 'r'] # En este caso solo se esta utilizando estas dos bandas (eliminar si se generaliza)

        bands = []
        flag = True
        for band in bands_name:
            band_x = filter_lc[filter_lc.fid == band]

            # Para normalizar
            if normalize:
                band_x = normalize_band(band_x, tpo_name, name_col_flux, name_col_error)
            bands.append(band_x)

            # BBDD con puntos desde la explosión con y sin norm
                # with 'and' --> 1713 supernovas
                # 'or' --> 1926 supernovas (En este caso se utiliza 'or')
            if len(band_x) >= n_det_min and flag:
                id_sn_clean.append(id_sn)
                flag = False

        filter_lc_pos = pd.concat(bands, axis=0).sort_values(by=['mjd'])
        lc_pos.append(filter_lc_pos)

    df_lc_pos = pd.concat(lc_pos, axis=0)

    if normalize:
        df_lc_pos = df_lc_pos.dropna() # # Eliminación de datos NAN post normalization
        print(f"Numero de supernovas despues de la limpieza y normalización: {len(id_sn_clean)}")
    else:
        print(f"Numero de supernovas despues de la limpieza: {len(id_sn_clean)}")
        
    df_lc_pos_clean = df_lc_pos[df_lc_pos.oid.isin(id_sn_clean)]
    
    return df_lc_pos_clean


def read_txt(file_path):
    file_labeled_ids = open(file_path, "r")
    labeled_ids = file_labeled_ids.read()
    file_labeled_ids.close()

    object_label = labeled_ids.split("\n")
    id_file = []
    for object in object_label:
        id = object.split(" ")
        id_file.append(id[0])

    return id_file[0:-1] # no considera el espacio al final del txt


def open_file(file_path, col_data, num_augments):
    files_data = glob.glob(file_path + '*.csv')
    data_augments = []

    for i in range(num_augments):
        reader = pd.read_csv(files_data[i], header=None, sep=",")
        reader.columns = col_data

        data_augments.append(reader)

    return data_augments

def load_npy_files(file_path, X_name, y_name):

    X      = np.load(file_path+X_name, allow_pickle=True).astype(np.float64)
    y      = np.load(file_path+y_name, allow_pickle=True).astype(np.float64)

    return [X, y]



def data_normalization(data_augments, tpo_name, flux_name, error_name):
    ids_sn = data_augments[0].oid.unique()
    data_augments_norm = []
    it = 0
    for data_i in data_augments:
        data_augments_norm.append([])
        print(f"Normalizando data {it+1}")

        for id_sn in ids_sn:
            df_sn = data_i[data_i.oid == id_sn]
            df_sn = normalize_band(df_sn, tpo_name, flux_name, error_name)
            data_augments_norm[it].append(df_sn)

        data_augments_norm[it] = pd.concat(data_augments_norm[it], axis=0)
        it += 1

    # La normalización puede generar N/A en bandas en donde solo haya un solo valor
    for it in range(len(data_augments_norm)):
        data_augments_norm[it] = data_augments_norm[it].dropna()

    return data_augments_norm

def input_model_charnock(data_augments):
    
    # Separa los datos
    ids_sne = data_augments[0].oid.unique()

    data, labels = [], []

    for data_i in data_augments:
        for id in ids_sne:
            filter_sn = data_i[data_i.oid == id]
            data_sequence = filter_sn.iloc[0:, 1:-1:].values
            label = filter_sn.iloc[0,-1]

            labels.append(label)
            data.append(torch.tensor(data_sequence))

    labels_torch = torch.tensor(labels)

    # shuffle in the same order
    dataset = list(zip(data, labels_torch))
    random.shuffle(dataset)
    data, labels_torch = zip(*dataset)

    return list(data), torch.stack(list(labels_torch))

def input_model_rapid(data):
    
    Atributes      = torch.tensor(data[0])
    labels_onehot  = torch.tensor(data[1]) 
    
    Labels         = torch.argmax(labels_onehot, dim=-1)

    return Atributes, Labels


def get_clf_for_RAPID(y_pred_list):

    y_pred_class = []
    pre_exp      = []

    for j in range(0, len(y_pred_list), 1):
        # Analizamos la curva de luz 
        lc_j        = y_pred_list[j]
        # Extraemos los valores y cantidad ded repeticiones
        values, counts = np.unique(lc_j, return_counts=True)
        # sorteamos los 3 primeros indices            
        most_common_indices = np.argsort(counts)[-2:][::-1]
        most_common         = values[most_common_indices]
        # empezamos a filtrar
        if (most_common[0] != 0):
            clf         = most_common[0]
            y_pred_class.append(clf)

        elif((len(most_common) == 1) and (most_common[0] == 0)):
            pre_exp.append(j)
        else:
            clf         = most_common[1]
            y_pred_class.append(clf)
    
    return np.array(y_pred_class), np.array(pre_exp)

