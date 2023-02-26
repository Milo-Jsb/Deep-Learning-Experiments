##############################################################################################################
#         Programa para realizar experimentos de Adaptacion de Dominio utilizando MME                        #
##############################################################################################################
# Por Daniel Moreno y Jorge Saavedra.                                                                        #                                                                          #
##############################################################################################################


# Definiciones iniciales #####################################################################################

# Sobre el modelo y el experimento
model       = "RAPID"                          # Modelo a utilizar
experiment  = "Sandia"                         # Nombre del experimento que realizaremos.

# Sobre los datos fuente
source      = "PLAsTiCC"                       # Conjunto de datos fuente
sub_source  = ""                               # Si los datos se encuentran en un subfolder agregar aqui
meta_source = False                            # Si se posee un csv con metadatos, utilizar.
train_s     = "data_train"                     # Folder con los datos de entrenamiento                  
sub_train_s = ""                                
test_s      = "data_test"                      # Folder con los datos de testeo
sub_test_s  = ""

# Sobre los datos objetivo
target       = "ZTF"                           # Conjunto de datos objetivo
sub_target   = "data_model_alerce/"             # Si los datos se encuentran en un subfolder agregar aqui     
meta_target  = True                            # Si se posee un csv con metadatos, utilizar.
train_t      = "data_train"                    # Folder con los datos de entrenamiento                  
sub_train_t  = ""
test_t       = "data_test"                     # Folder con los datos de testeo
sub_test_t   = "static/"

print(100*"_")
print(f"Experimento de MME: {model}, con datos de {source} a {target}")
print(100*"_")
print(f"Nombre: {experiment}")

# Modulos a Importar #########################################################################################
import torch
import itertools
import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# Programas y funciones creadas por dani #####################################################################
import utils.data_split          as ds   
import utils.plots               as up   
import utils.data_loaders        as dl   
import utils.metrics             as um   
import utils.data_handling       as dh  
import utils.mme.data_mme        as umme
import CONF.RAPID.conf           as confi 
import models.base_model
import utils.mme.data_mme        

# Configuracion del modelo ##################################################################################
CONFIG = confi.get_args()

print(100*"_")
print("Hiperparametros y configuracion del modelo (conf file)             ")
print(100*"_")
print(CONFIG)
print(100*"_")

# Path Generales ############################################################################################
path_source  = f"./data/{source}/{sub_source}"
path_target  = f"./data/{target}/{sub_target}"
path_fig     = f"./figures/{model}/{experiment}/" 
path_mdata   = f"./data/metadatos/"
 
# Taxonomia PLAsTiCC ########################################################################################
"""
Taxonomía = [    [nombres de las clases]
                 [nuevas clases]
                 [definicion numerica]  ]
"""
Tax = [ 
        ["SNIbc", 'CaRT', 'SNIa-normal', 'PISN', 'SNIax', 'SNIa-91bg', 'TDE', 'SNII',
        'point-Ia', 'KN', 'SLSN-I', 'ILOT'],
        
        ["non-SNIa", "non-SNIa", "SNIa", "non-SNIa", "SNIa", "SNIa","non-SNIa",
        "non-SNIa","SNIa","non-SNIa","non-SNIa","non-SNIa"],

        [2,2,1,2,1,1,2,2,1,2,2,2]
        
        ]

# Seleccion de procesos ######################################################################################
load_data        = True   # Cargar datos originales.
split_data       = True   # Dividir datos.
load_split       = False  # Cargar los archivos ya divididos
dom_adap         = True   # Utilizar Adaptacion de Dominio
upload_model     = False  # Cargar modelo ya entrenado.
test_model       = True   # Predecir con un conjunto de test

# Imprimimos en la terminal
print(100*"_")
print("Resumen de ejecucion                                ")
print(100*"_")
print(f"Cargar datos Originales:     | {load_data}")
print(f"Dividir datos Originales:    | {split_data}")
print(f"Cargar datos divididos:      | {load_split}")
print(f"Adaptacion de dominio:       | {dom_adap}")
print(f"Cargar modelo:               | {upload_model}")
print(f"Testear modelo               | {test_model}")
print(100*"_")

# Cargamos datos sin dividir ######################################################################################
if load_data:
    """
    Este proceso lo he hecho varias veces, creo que podríamos hacerlo una funcion en dataloaders.py
    """
    # Source #####################################################################################################

    timesX_s          = np.load(path_source+"timesX.npy",allow_pickle=True)         # Tiempo de las observaciones
    X_s               = np.load(path_source+"X.npy",allow_pickle=True)              # FLujo Bandas g y r 
    y_s               = np.load(path_source+"y.npy",allow_pickle=True)              # Clases por punto de observación
    labels_s          = np.load(path_source+"labels.npy",allow_pickle=True)         # Clase por supernova
    object_ids_s      = np.load(path_source+"object_ids.npy",allow_pickle=True)     # IDs de cada supernova 
    general_labels_s  = np.load(path_source+"general_labels.npy",allow_pickle=True) # Nombre de las clases generales

    if (len(X_s[0]) != 0):
        # Los datos de PLAsTiCC tiene features adicionales a las bandas, este paso las remueve del programa
        X_s = X_s[:,:2,:]
    
    # Imprimimos resumen de los datos
    print(100*"_")
    print(f"Resumen de datos Source: {source}")
    print(100*"_")
    print(f"Cantidad de bandas fotométricas:         | {len(X_s[0])}")
    print(f"Tipos de bandas                          | [g, r]")
    print(f"Cantidad de SNs:                         | {len(object_ids_s)}")
    print(f"Cantidad de observaciones por SNs:       | {len(X_s[0][0])}")
    print(f"Clases generales de clasificacion:       | {general_labels_s}")
    print(f"Clase adicional:                         | Pre-Explosion -> [0]")
    print(100*"_")

    # Imprimimos dimensiones de los datos
    print(100*"_")
    print("Shape de los datos originales sin dividir ")
    print(100*"_")
    print(f"Tiempo:                     | {timesX_s.shape}")
    print(f"Atributos:                  | {X_s.shape}")
    print(f"Clases por observacion:     | {y_s.shape}")
    print(f"Clases por supernova:       | {labels_s.shape}")
    print(f"Clases por supernova:       | {np.unique(labels_s)}")
    print(100*"_")


    # Target #####################################################################################################

    timesX_t          = np.load(path_target+"timesX.npy",allow_pickle=True)         # Tiempo de las observaciones
    X_t               = np.load(path_target+"X.npy",allow_pickle=True)              # FLujo Bandas g y r 
    y_t               = np.load(path_target+"y.npy",allow_pickle=True)              # Clases por punto de observación
    labels_t          = np.load(path_target+"labels.npy",allow_pickle=True)         # Clase por supernova
    object_ids_t      = np.load(path_target+"object_ids.npy",allow_pickle=True)     # IDs de cada supernova 
    general_labels_t  = np.load(path_target+"general_labels.npy",allow_pickle=True) # Nombre de las clases generales

    # Imprimimos resumen de los datos
    print(100*"_")
    print(f"Resumen de datos Target: {target}")
    print(100*"_")
    print(f"Cantidad de bandas fotométricas:         | {len(X_t[0])}")
    print(f"Tipos de bandas                          | [g, r]")
    print(f"Cantidad de SNs:                         | {len(object_ids_t)}")
    print(f"Cantidad de observaciones por SNs:       | {len(X_t[0][0])}")
    print(f"Clases generales de clasificacion:       | {general_labels_t}")
    print(f"Clase adicional:                         | Pre-Explosion -> [0]")
    print(100*"_")

    # Creamos un dataframe con los metadatos disponibles
    metadata_df     = pd.read_csv(path_mdata+"metadata.csv")
    metadata        = metadata_df.loc[metadata_df['oid'].isin(object_ids_t)]

    # Imprimimos metadatos
    print(100*"_")
    print("Metadatos disponibles:") 
    print(100*"_")
    print(print(metadata))
    print(100*"_")

    # Imprimimos dimensiones de los datos
    print(100*"_")
    print("Shape de los datos originales sin dividir ")
    print(100*"_")
    print(f"Tiempo:                     | {timesX_t.shape}")
    print(f"Atributos:                  | {X_t.shape}")
    print(f"Clases por observacion:     | {y_t.shape}")
    print(f"Clases por supernova:       | {labels_t.shape}")
    print(f"Clases por supernova:       | {np.unique(labels_t)}")
    print(100*"_")

# Dividimos datos para mme ########################################################################################
if split_data:
    
    # Revision de datos para PLAsTiCC
    if (len(np.unique(labels_s)) != 2):
        
        # Tipos de supernova y cantidad
        unique, counter = np.unique(labels_s, return_counts= True)

        # Desplegamos la informacion en un dataframe
        FullClasses = pd.DataFrame({"Object": Tax[0], "General-Labels": general_labels_s, "frecuency": counter, 
                                "y-Labels": unique, "Binary-Classes": Tax[1], "New-Labels": Tax[2]})

        print(100*"_")
        print(f"Clases entregadas en los datos {source} y sus valores")
        print(100*"_")
        print(FullClasses)
        print(100*"_")

        # Buscamos reemplazar las etiquetas para crear una clasificacion binaria
        replace_nonsn    = [1,2,4,7,8,10,11,12]
        replace_sn       = [3,5,6,9]
        
        # Removemos alguna columna de datos si es necesario
        remove_classes = True

        if remove_classes:

            index_class  = np.where(labels_s == 10)
            timesX_s     = np.delete(timesX_s, index_class, axis = 0)
            X_s          = np.delete(X_s, index_class, axis = 0)
            y_s          = np.delete(y_s, index_class, axis = 0)
            labels_s     = np.delete(labels_s, index_class, axis = 0)
            object_ids_s = np.delete(object_ids_s, index_class, axis = 0)
            
            # Imprimimos 
            print(100*"_")
            print(f"Se ha retirado la siguiente clase de los datos")
            print(100*"_")
            print(FullClasses.loc[FullClasses["y-Labels"]==10])
            print(100*"_")

            # Redefinimos la nueva clasificacion
            replace_nonsn  = [1,2,4,7,8,11,12]
        

        # Reemplazamos la clasificacion para las no sns en y
        mask_nonsn_1      = np.isin(y_s, replace_nonsn)
        y_s[mask_nonsn_1] = 2

        # Reemplazamos la clasificacion para las no sns en labels
        mask_nonsn_2           = np.isin(labels_s, replace_nonsn)
        labels_s[mask_nonsn_2] = 2

        # Reemplazamos la clasificacion para las sn en y 
        mask_sn_1      = np.isin(y_s, replace_sn)
        y_s[mask_sn_1] = 1

        # Reemplazamos la clasificacion para las sn en labels 
        mask_sn_2           = np.isin(labels_s, replace_sn)
        labels_s[mask_sn_2] = 1

    # Cantidad de elementos por cada subconjunto del target
    total_length       = len(object_ids_t)
    test_lenght        = int(total_length * 0.1)    # 10% ~ 200 supernovas
    label_train_lenght = 10
    label_val_lenght   = 25
    unlabeled_lenght   = total_length - test_lenght -label_train_lenght - label_val_lenght

    # Seleccion random de elementos para cada subconjunto del target
    ramdon_index = np.random.choice(np.arange(0,total_length ), size=total_length, replace = False)
    
    # Index position
    test_idx     = ramdon_index[0 : test_lenght]
    lb_train_idx = ramdon_index[test_lenght : (test_lenght + label_train_lenght) ]
    lb_val_idx   = ramdon_index[(test_lenght + label_train_lenght) : (test_lenght + label_train_lenght +label_val_lenght )]
    unlb_idx     = ramdon_index[(test_lenght + label_train_lenght + label_val_lenght ) : total_length ]

    # Paths
    path_data_source  = path_source+"semisupervised_mme/" 
    path_data_target  = path_target+"semisupervised_mme/"
    
    # Generamos los splits
    X_st , y_st , labels_st  = umme.split_mme_data("source", "train", None,
                                                   X_s, y_s, timesX_s, object_ids_s, labels_s,
                                                   path_data_source)
    data_source         = [X_st, y_st]    
    
    X_tt , y_tt , labels_tt  = umme.split_mme_data("target", "test", test_idx,
                                                   X_t, y_t, timesX_t, object_ids_t, labels_t,
                                                   path_data_target)
    data_target_test    = [X_tt, y_tt]


    X_tlt, y_tlt, labels_tlt = umme.split_mme_data("target", "label_train", lb_train_idx, 
                                                   X_t, y_t, timesX_t, object_ids_t, labels_t,
                                                   path_data_target)
    data_label_train    = [X_tlt, y_tlt]
    
    X_tlv, y_tlv, labels_tlv = umme.split_mme_data("target", "label_val" , lb_val_idx,
                                                   X_t, y_t, timesX_t, object_ids_t, labels_t,
                                                   path_data_target)
    data_label_val      = [X_tlv, y_tlv]

    X_tul, y_tul, labels_tul = umme.split_mme_data("target", "unlabel", unlb_idx,
                                                   X_t, y_t, timesX_t, object_ids_t, labels_t,
                                                   path_data_target)
    data_unlabel        = [X_tul, y_tul]

    # Dataloader
    batch_size = 256
    one_hot    = False

    dataloader  = dl.PhotometryDataLoaders(data_source, data_label_train, data_unlabel, data_label_val, data_target_test, 
                            one_hot,batch_size, num_workers=0, shuffle=False, collate_fn=None, 
                            normalize=False, n_quantiles=1000, weight_norm=False)
    
    dataloader_source      = dataloader.source_set
    dataloader_target_lab  = dataloader.target_set_lab
    dataloader_target_unl  = dataloader.target_set_unl
    dataloader_target_val  = dataloader.target_set_val
    dataloader_target_test = dataloader.target_data_test

    # Atributos y numero de clases en el set de datos
    input_size  = dataloader_source.dataset.data[0].size(1)
    num_classes = dataloader.nb_classes

    print(100*"_")
    print(dataloader)
    print(f"Input-size:        {input_size}")
    print(f"Numero de clases:  {num_classes}")
    print(100*"_")

# Cargamos datos previamente divididos ############################################################################
if load_split:

    dataloader_source, \
        dataloader_target_lab, \
            dataloader_target_unl, \
                dataloader_target_val, \
                    dataloader_target_test = umme.get_dataset(CONFIG)
    
# Realizamos Adaptacino de Dominio ################################################################################
if dom_adap:
    
    # Definimos su configuracion
    def model_type(CONFIG):
    
        if CONFIG.load_model:
            model = models.base_model.Model(CONFIG)
            model.summary()

        else:
            model = models.base_model.Model(CONFIG)
            model.summary()
            model.compile(CONFIG.loss_list, CONFIG.optimizer, CONFIG.lr, CONFIG.metric_eval)

        return model
    
    # Creamos el modelo
    rapid_model = model_type(CONFIG)

    print(100*"_")
    print(rapid_model)
    print(100*"_")

    # Comienza el entrenamiento n

    rapid_model.fit(source_dl = dataloader_source, 
                   target_lab_dl = dataloader_target_lab,
                   target_unl_dl = dataloader_target_unl,
                   val_dl = dataloader_target_val,
                   train_type = CONFIG.type_train, 
                   num_epochs = CONFIG.num_epochs, 
                   verbose = CONFIG.verbose)
    
    

    # ----------------------------------------------------------------------------------------
    metric        = 'Loss'
    labels_curves = ['Training', 'Validation']
    color_curves  = ['y', 'r']

    train_loss = rapid_model.checkpoint.train_loss
    val_loss   = rapid_model.checkpoint.val_loss
    up.plot_learning_curves(train_loss, val_loss, metric, labels_curves, color_curves, loc='lower right', path_save=path_fig)

    # ----------------------------------------------------------------------------------------
    metric = 'Accuracy'
    labels_curves = ['Training', 'Validation']
    color_curves = ['y', 'r']


    train_acc = rapid_model.checkpoint.train_acc
    val_acc   = rapid_model.checkpoint.val_acc
    up.plot_learning_curves(train_acc, val_acc, metric, labels_curves, color_curves, loc='lower right', path_save=path_fig)

# Cargamos un modelo previamente entrenado ########################################################################
if upload_model:
    
    # Definimos su configuracion
    def model_type(CONFIG):
    
        if CONFIG.load_model:
            model = models.base_model.Model(CONFIG)
            model.summary()

        else:
            model = models.base_model.Model(CONFIG)
            model.summary()
            model.compile(CONFIG.loss_list, CONFIG.optimizer, CONFIG.lr, CONFIG.metric_eval)

        return model
    
    # Cambiamos la configuracion
    CONFIG.load_model=True
    CONFIG.model_path='/best_result/rapidpt-922-0.8000.pt' 

    # Recreamos el modelo
    rapid_model = model_type(CONFIG)

    print(100*"_")
    print(f"Se cargaron correctamente los pesos:         {CONFIG.model_path}")
    print(100*"_")

if test_model:
    # Evaluamos
    loss, acc = rapid_model.evaluate(dataloader_target_test, dataloader_target_test.batch_size)

    # Realizamos las predicciones con los datos de test
    y_pred_prob, y_pred = rapid_model.predict(dataloader_target_test, dataloader_target_test.batch_size)
    # Definimos los labels a utilizar
    dict_labels = {1: 'SNIa', 2: 'nonSNIa', 0: "Prexplosion"}

    #Cargamos los datos de test
    y_target = np.load(path_target+"y.npy",allow_pickle=True).astype(np.int64)
    ids_test = np.load(path_target+"/semisupervised_mme/test/ramdon_ids.npy",allow_pickle=True)

    y_test = y_target[ids_test, :]

    # Generamos formato onehot
    tensor         = torch.from_numpy(y_test)
    
    # Imprimimos informacion desde las prediciones 
    print(100*"_")
    print("La prediccion de datos fue realizada correctamente")
    print(100*"_")
    print(f"Shape y_pred_prob:              {y_pred_prob.shape}")
    print(f"Shape y_pred:                   {y_pred.shape}")
    print(f"Shape y_test:                   {y_test.shape}")
    print(100*"_")

    # Imprimimos las metricas obtenidas con los datos de test
    print(100*"_")
    print("Resultados del modelo con el conjunto de datos de test")
    print(100*"_")
    
    one_hot = False

    um.print_metrics_pbp_lc(tensor, y_pred, y_pred_prob, loss, dataloader_target_test, dict_labels, "rapid",one_hot)
    print(100*"_")
    
    # Generamos una matriz de confusion
    figsize = (13, 7)

    # Extraemos las predicciones para la curva entera de luz, en este caso es necesario filtrar si hay curvas
    # clasificadas como preexplosiones o no
    
    _, pre_exp_p = dh.get_clf_for_RAPID(y_pred)
    
    if (len(pre_exp_p) != 0):
        # Filtramos aquellas clasificadas como pre-explosiones
        for a in range(0, len(pre_exp_p), 1):
            # indice de la supernova
            pre_idx  = pre_exp_p[a]
            # curva de luz correspondiente
            pre_a    = y_pred[pre_idx]
            # lista con probabilidades
            prob_a   = y_pred_prob[pre_idx]
            # buscamos solo entre las opciones 1 y 2
            new_crit = prob_a[1:]
            
            for i in range(prob_a.shape[1]):
                #el +1 va para que el nuevo index sea consistente con la notacion 0 :Pre_exp, 1: SN1a, 2: nonSN1a 
                y_pred[pre_idx, i] = np.argmax(new_crit[:,i])+1
            
    y_pred_class, _ = dh.get_clf_for_RAPID(y_pred)

    #cargamos los labels
    labels_target = np.load(path_target+"labels.npy",allow_pickle=True)
    ids_test      = np.load(path_target+"/semisupervised_mme/test/ramdon_ids.npy",allow_pickle=True)

    labels = labels_target[ids_test]        

    # Generamos un diccionario de clasificacion binaria     
    dict_labels = {1: 'SNIa', 2: 'nonSNIa'}

    label_order = ['SNIa', 'nonSNIa']

    up.plot_cm(labels, y_pred_class, label_order, dict_labels, figsize, path_fig)

print(100*"_")
print("El programa se ejecutó correctamente")
print(100*"_")
