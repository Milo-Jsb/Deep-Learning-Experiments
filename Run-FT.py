##############################################################################################################
#         Programa para realizar experimentos de Adaptacion de Dominio utilizando Fine Tunning               #
##############################################################################################################
# Por Daniel Moreno y Jorge Saavedra.                                                                        #
# Ultima actualizacion: 06/01/2023                                                                           #
##############################################################################################################

# Definiciones iniciales #####################################################################################

# Sobre el modelo y el experimento
model       = "RAPID"                          # Modelo a utilizar
experiment  = "First-try"                      # Nombre del experimento que realizaremos.

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
sub_target   = ""                              # Si los datos se encuentran en un subfolder agregar aqui     
meta_target  = True                            # Si se posee un csv con metadatos, utilizar.
train_t      = "data_train"                    # Folder con los datos de entrenamiento                  
sub_train_t  = ""
test_t       = "data_test"                     # Folder con los datos de testeo
sub_test_t   = "static/"

# Cantidad de splits
n_splits  = 5


print(100*"_")
print(f"Experimento de fine-tunning: {source} a {target}")
print(100*"_")
print(f"Nombre: {experiment}")


# Modulos a Importar #########################################################################################
import glob
import torch
import itertools
import os
import numpy             as np
import pandas            as pd
import random            as rm
import matplotlib.pyplot as plt

# Programas y funciones creadas por dani (tuneadas por mi) ###################################################
import utils.plots               as up    
import utils.metrics             as um    
import utils.data_loaders        as dl    
import utils.data_split          as ds   
import utils.data_handling       as dh   
import models.base_model  

# Discriminamos la configuracion para los modelos según el modelo a utilizar #################################
if (model == "RAPID"):
    import CONF.RAPID.conf        as confi 

elif (model == "Charnock"):
    import CONF.Charnock.conf     as confi 

# Funciones Varias ###########################################################################################
def model_type(CONFIG):
        
    if CONFIG.load_model:
        model = models.base_model.Model(CONFIG)
        model.summary()

    else:
        model = models.base_model.Model(CONFIG)
        model.summary()
        model.compile(CONFIG.loss_list, CONFIG.optimizer, CONFIG.lr, CONFIG.metric_eval)

    return model 

# Seleccion de procesos ######################################################################################

# Datos
load_split       = True     # Cargar los archivos ya divididos
upload_model     = True     # Cargar modelo previamente entrenado.
dom_adap         = False    # Realizar adaptacion de dominio
test_model       = True     # Predecir con un conjunto de testeo

# Imprimimos en la terminal
print(100*"_")
print("Resumen de ejecucion: ")
print(100*"_")
print(f"Cargar datos divididos:           | {load_split}")
print(f"Cargar modelo:                    | {upload_model}")
print(f"Realizar Adaptacion de Dominio    | {dom_adap}")
print(f"Testear modelo                    | {test_model}")
print(100*"_")

# Path a los archivos de datos ###############################################################################

path_source  = f"./data/{source}/{sub_source}"
path_target  = f"./data/{target}/{sub_target}"
path_fig     = f"./figures/{model}/{experiment}/" 


# Cargamos datos ##############################################################################################

if load_split:
    # Esto es solo para no repetir dos veces el cargo de datos
    Dataset   = [source, target]
    Datapath  = [path_source, path_target]
    Train     = [train_s, train_t]
    sub_Train = [sub_train_s, sub_train_t]
    Test      = [test_s, test_t]
    sub_Test  = [sub_test_s, sub_test_t]
    
    # Creamos listas para guardar
    Source_dataloader = []
    Target_dataloader = []
    
    for a in range(0, len(Dataset), 1):
        # Path a los archivos
        path_test  = Datapath[a]+f"{Test[a]}/{sub_Test[a]}"
        path_train = Datapath[a]+f"{Train[a]}/{sub_Train[a]}"

        # Especificaciones para los numpy array
        if (Dataset[a] == "ZTF"):
            n_obs    = 50         # Nro de observaciones por curva
            n_train  = 1681       # Cantidad de supernovas para entrenar
            n_val    = 187        # Cantidad de supernovas para evaluar
            n_test   = 200        # Cantidad de supernovas para testear
            
        
        elif (Dataset[a] == "PLAsTiCC"):
            n_obs    = 50         # Nro de observaciones por curva
            n_train  = 15792      # Cantidad de supernovas para entrenar
            n_val    = 3948       # Cantidad de supernovas para evaluar
            n_test   = 2194       # Cantidad de supernovas para testear
                

        # Se crean las listas para guardar los datos 

        # (splits de datos, supernovas, observaciones, banda)
        X_train = np.empty(shape=(n_splits, n_train, n_obs, 2), dtype="float64") 
        X_val   = np.empty(shape=(n_splits, n_val, n_obs, 2), dtype="float64")

        # (splits de datos, supernovas, observaciones, clase)
        y_train = np.empty(shape=(n_splits, n_train, n_obs, 3), dtype="float64")
        y_val   = np.empty(shape=(n_splits, n_val, n_obs, 3), dtype="float64")

        # Se cargan los datos de entrenamiento y validacion
        for i in range(n_splits):

            # Entrenamiento
            X_train[i] = np.load(path_train+"fold_"+str(i)+"/train/X.npy", allow_pickle=True)
            y_train[i] = np.load(path_train+"fold_"+str(i)+"/train/y.npy", allow_pickle=True)
            
            # Validacion
            X_val[i]   = np.load(path_train+"fold_"+str(i)+"/validation/X.npy", allow_pickle=True)
            y_val[i]   = np.load(path_train+"fold_"+str(i)+"/validation/y.npy", allow_pickle=True)

        # Se cargan los datos de testeo

        X_test      = np.load(path_test+"X.npy", allow_pickle=True)
        y_test      = np.load(path_test+"y.npy", allow_pickle=True)
        timesX_test = np.load(path_test+"timesX.npy", allow_pickle=True)
        
        print(100*"_")
        print(f"Shape de los datos: Entrenamiento ({Dataset[a]})")
        print(100*"_")
        print(f"Atributos:                  | {X_train.shape}")
        print(f"Clases por observacion:     | {y_train.shape}")
        print(100*"_")

        # Imprimimos info
        print(100*"_")
        print(f"Shape de los datos: Validacion ({Dataset[a]})")
        print(100*"_")
        print(f"Atributos:                  | {X_val.shape}")
        print(f"Clases por observacion:     | {y_val.shape}")
        print(100*"_")
    
        # Imprimimos info
        print(100*"_")
        print(f"Shape de los datos: test ({Dataset[a]})")
        print(100*"_")
        print(f"Atributos:                  | {X_test.shape}")
        print(f"Clases por observacion:     | {y_test.shape}")
        print(100*"_")

        # Transformamos los datos a tensores
        X_train           = torch.tensor(X_train)
        y_train_onehot    = torch.tensor(y_train) 
        X_val             = torch.tensor(X_val) 
        y_val_onehot      = torch.tensor(y_val) 
        X_test            = torch.tensor(X_test) 
        y_test_onehot     = torch.tensor(y_test) 


        # Toma la posicion del mayor valor, por alguna razon esto redimensiona las matrices
        y_train    = torch.argmax(y_train_onehot, dim=-1)
        y_val      = torch.argmax(y_val_onehot, dim=-1)
        y_test     = torch.argmax(y_test_onehot, dim=-1)
        

        # Informacion necesaria
        batch_size = 256
        one_hot    = False


        # Generamos un ciclo para realizar los folds
        if (Dataset[a] == source):
            
            for split in range(n_splits):
                
                Source_dataloader.append(dl.ZTFDataLoaders(X_train[split], 
                                                           y_train[split], 
                                                           X_val[split], 
                                                           y_val[split], 
                                                           X_test,
                                                           y_test, 
                                                           one_hot, 
                                                           model,batch_size=batch_size, 
                                                           num_workers=0,
                                                           shuffle=False, 
                                                           collate_fn=None, 
                                                           weight_norm=False))

            print(100*"_")
            print(f"ZTFDataLoaders se ejecuto exitosamente para {source} con {n_splits}-folds")
            print(100*"_")

        elif(Dataset[a] == target):
            
            for split in range(n_splits):
                Target_dataloader.append(dl.ZTFDataLoaders(X_train[split], 
                                                           y_train[split], 
                                                           X_val[split], 
                                                           y_val[split], 
                                                           X_test,
                                                           y_test, 
                                                           one_hot, 
                                                           model,batch_size=batch_size, 
                                                           num_workers=0,
                                                           shuffle=False, 
                                                           collate_fn=dl.collate_fn, 
                                                           weight_norm=False))
                
            print(100*"_")
            print(f"ZTFDataLoaders se ejecuto exitosamente para {target} con {n_splits}-folds")
            print(100*"_")

# Cargamos el modelo previamente entrenado #########################################################################
if upload_model:
    # Creamos la configuracion del modelo    
    CONFIG = confi.get_args()

    # Nos aseguramos de que se pide cargar el modelo
    CONFIG.load_model = True

    # Creamos una lista para guardar los modelos
    Full_models = []


    # Explicitamos los modelos a cargar
    model_paths = ["rapidpt-514-0.9104.pt",
                   "rapidpt-418-0.9184.pt",
                   "rapidpt-488-0.8952.pt",
                   "rapidpt-964-0.9267.pt",
                   "rapidpt-448-0.9170.pt"]

    for split in range(n_splits):
        
        # Definimos el path donde se guardaron los parametros
        CONFIG.experiment = f"/{experiment}_split_{split}/"
        # Definimos el path donde se guardo el checkpoint del modelo
        CONFIG.model_path = f"/best_result/{model_paths[split]}"

        # Cargamos los checkpoints
        rapid_model = model_type(CONFIG)
        
        print(100*"_")
        print(f"Se cargaron correctamente los pesos:         {CONFIG.model_path}")
        print(100*"_")

        # Guardamos el modelo
        Full_models.append(rapid_model)
    
    print(Full_models)

# Realizamos Fine-Tunning #########################################################################################
if dom_adap:

    print(100*"_")
    print("Comienza el re-entrenamiento")
    print(100*"_")
    rapid_model.fit(Target_dataloader[0].train_set, None, None, None,
                      Target_dataloader[0].val_set, 'ft', CONFIG.num_epochs, CONFIG.verbose)
    print(100*"_")

    # Analizamos la perdida del modelo
    metric        = 'Loss'
    labels_curves = ['Training', 'Validation']
    color_curves  = ['firebrick', 'gold']

    # Extraemos la perdida de entrenamiento y validacion
    train_loss    = rapid_model.checkpoint.train_loss
    val_loss      = rapid_model.checkpoint.val_loss

    # Ploteamos
    up.plot_learning_curves(train_loss, val_loss, metric, labels_curves, color_curves, loc = "best", path_save= path_fig)

    # Analizar la precision del modelo
    metric        = 'Accuracy'
    labels_curves = ['Training', 'Validation']
    color_curves  = ['firebrick', 'gold']

    # Extraemos la precision del entrenamiento y validacion
    train_acc     = rapid_model.checkpoint.train_acc
    val_acc       = rapid_model.checkpoint.val_acc
    
    # Ploteamos
    up.plot_learning_curves(train_acc, val_acc, metric, labels_curves, color_curves, loc = "best", path_save= path_fig)

# Cargamos un modelo ya adaptado #################################################################################



# Testeamos el modelo ############################################################################################
if test_model:

    # Extraemos perdida y precision del modelo durante la validacion
    loss, acc = rapid_model.evaluate(Target_dataloader[0].test_set, Target_dataloader[0].test_set.batch_size)

    # Realizamos las predicciones con los datos de test
    y_pred_prob, y_pred = rapid_model.predict(Target_dataloader[0].test_set, Target_dataloader[0].test_set.batch_size)
    # Definimos los labels a utilizar
    dict_labels = {1: 'SNIa', 2: 'nonSNIa', 0: "Prexplosion"}
    
    # Imprimimos informacion desde las prediciones 
    print(100*"_")
    print("La prediccion de datos fue realizada correctamente")
    print(100*"_")
    print(f"Shape y_pred_prob:              {y_pred_prob.shape}")
    print(f"Shape y_pred:                   {y_pred.shape}")
    
    path_test         = Datapath[1]+f"{Test[1]}/{sub_Test[1]}"
    y_test            = np.load(path_test+"y.npy", allow_pickle=True)
    y_test_onehot     = torch.tensor(y_test)
    y_test            = torch.argmax(y_test_onehot, dim=-1)
     
    print(f"Shape y_test:                   {y_test.numpy().shape}")
    print(100*"_")

    # Imprimimos las metricas obtenidas con los datos de test
    print(100*"_")
    print("Resultados del modelo con el conjunto de datos de test")
    print(100*"_")
    um.print_metrics_pbp_lc(y_test, y_pred, y_pred_prob, loss, Target_dataloader[0], dict_labels, "rapid", one_hot)
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
    y_test_class, _ = dh.get_clf_for_RAPID(y_test.numpy())
        

    # Generamos un diccionario de clasificacion binaria     
    dict_labels = {1: 'SNIa', 2: 'nonSNIa'}

    label_order = ['SNIa', 'nonSNIa']

    # Imprimimos las metricas obtenidas con los datos de test
    print(100*"_")
    print("Resultados del modelo con el conjunto de datos de test full curva")
    print(100*"_")
    um.print_metrics_full_lc(torch.from_numpy(y_test_class), y_pred_class, y_pred_prob, loss, dict_labels , False)
    print(100*"_")


    # Creamos la matriz de confusion
    up.plot_cm(y_test_class, y_pred_class, label_order, dict_labels, figsize, path_fig)

print(100*"_")
print("El programa se ejecutó correctamente")
print(100*"_")

################################################################################################################
