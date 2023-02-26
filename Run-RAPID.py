"""
______________________________________________________________________________________________________________
Supernovae Binary Classification
- Model:              RAPID.
- Original work:      Muthukrishna, et al (2019).
- Modifications:      Daniel Moreno & Milo.

*Notes: This code only accepts data preprocessed with the method suggested by the authors, currently it has 
        been tested with datasets from the Zwicky Transient Facility (ZTF) and Photometric LSST Astronomical 
        Time-series Classification Challenge (PLAsTiCC).
______________________________________________________________________________________________________________
"""

# Initial Definitions ########################################################################################

# Dataset especifications
Dataset    = "ZTF"                 
sub_path   = "data_model_alerce/"  

# Name of the experiment
experiment = "ZTF-Final"

# Data related quantities 
metadatos  = True                  
n_splits   = 5                     
num_class  = 2                     

# Folders 
F_train    = "data_train"          
sF_train   = ""                    
F_test     = "data_test"           
sF_test    = "static/"             

# Taxonomy (if needed)
"""
Tax = [  [original_classes], [new_clases], [labels] ]
"""
Tax = [ 
        ["SNIbc", 'CaRT', 'SNIa-normal', 'PISN', 'SNIax', 'SNIa-91bg', 'TDE', 'SNII',
        'point-Ia', 'KN', 'SLSN-I', 'ILOT'],
        
        ["non-SNIa", "non-SNIa", "SNIa", "non-SNIa", "SNIa", "SNIa","non-SNIa",
        "non-SNIa","SNIa","non-SNIa","non-SNIa","non-SNIa"],

        [2,2,1,2,1,1,2,2,1,2,2,2]
      ]

# Start ######################################################################################################
print(110*"_")
print(f"RAPID with {Dataset}:")
print(110*"_")
print(f"Experiment {experiment}")

# Modulos to import ##########################################################################################
import numpy                     as np
import pandas                    as pd
import matplotlib.pyplot         as plt
import torch
import itertools
import os

# Programs and utilities #####################################################################################
import utils.data_split          as ds   
import utils.plots               as up   
import utils.data_loaders        as dl   
import utils.metrics             as um   
import CONF.RAPID.conf           as conf 
import utils.data_handling       as dh   
import models.base_model         as bm  
print(110*"_")

# Process selection ##########################################################################################
plot_data        = True     
load_data        = True     
split_data       = False    
load_split       = True     
train_model      = True    
upload_model     = False    
test_model       = True    

# Print Summary
print(110*"_")
print("Program Summary")
print(110*"_")

print(f"Lightcurve data plots:            | {plot_data}")
print(f"Load Original Data:               | {load_data}")
print(f"Split Original Data:              | {split_data}")
print(f"Load Spli Data:                   | {load_split}")
print(f"Train Model:                      | {train_model}")
print(f"Load Trained Model:               | {upload_model}")
print(f"Test Trained Model:               | {test_model}")
print(110*"_")

##############################################################################################################
"""
______________________________________________________________________________________________________________
- path_data:   Path to data files. 
- path_mdata:  Path to metadata, particulary a .csv file.
- path_fig:    Path to save the figures created by the program.
______________________________________________________________________________________________________________
* Note: These three directories need to be created before running the program.
"""
path_data  = f"./data/{Dataset}/{sub_path}/"
path_mdata = f"./data/{Dataset}/" 
path_fig   = f"./figures/RAPID/{experiment}/" 

##############################################################################################################
if load_data:
    """
    __________________________________________________________________________________________________________
    Here the main objective is to  load the raw-data (but already preprocessed), use only the desired bands 
    and preprare for split
    __________________________________________________________________________________________________________
    """
    timesX          = np.load(path_data+"timesX.npy",allow_pickle=True)         
    X               = np.load(path_data+"X.npy",allow_pickle=True)              
    y               = np.load(path_data+"y.npy",allow_pickle=True)              
    labels          = np.load(path_data+"labels.npy",allow_pickle=True)         
    object_ids      = np.load(path_data+"object_ids.npy",allow_pickle=True)     
    general_labels  = np.load(path_data+"general_labels.npy",allow_pickle=True) 

    # Only bands as atributes
    if (len(X[0]) != 0):
        X = X[:,:2,:]
    
    # Summary
    print(100*"_")
    print("Resumen de datos                           ")
    print(100*"_")
    print(f"Cantidad de bandas fotométricas:         | {len(X[0])}")
    print(f"Tipos de bandas                          | [g, r]")
    print(f"Cantidad de SNs:                         | {len(object_ids)}")
    print(f"Cantidad de observaciones por SNs:       | {len(X[0][0])}")
    print(f"Clases generales de clasificacion:       | {general_labels}")
    print(f"Clase adicional:                         | Pre-Explosion -> [0]")
    print(100*"_")

    if metadatos:
        
        metadata_df  = pd.read_csv(path_mdata+"metadata.csv")
        metadata     = metadata_df.loc[metadata_df['oid'].isin(object_ids)]

        # Summary
        print(100*"_")
        print("Metadata available:") 
        print(100*"_")
        print(print(metadata))
        print(100*"_")

    # Summary
    print(100*"_")
    print("Shape de los datos originales sin dividir ")
    print(100*"_")
    print(f"Tiempo:                     | {timesX.shape}")
    print(f"Atributos:                  | {X.shape}")
    print(f"Clases por observacion:     | {y.shape}")
    print(f"Clases por supernova:       | {labels.shape}")
    print(f"Clases por supernova:       | {np.unique(labels)}")
    print(100*"_")

##############################################################################################################
if split_data:
    """
    __________________________________________________________________________________________________________
    Here we first check if data its ready for binary classification, and remove any class if needed. Then we
    split data (from scrach or with metadata), transform the atributes and the labels to tensor format, and 
    finally create a dataloader.  
    __________________________________________________________________________________________________________
    """

    # Preparing binary classification
    if (len(np.unique(labels)) != 2):
        
        # Info
        unique, counter = np.unique(labels, return_counts= True)
        FullClasses     = pd.DataFrame({"Object": Tax[0], 
                                        "General-Labels": general_labels, 
                                        "frecuency": counter, 
                                        "y-Labels": unique, 
                                        "Binary-Classes": Tax[1], 
                                        "New-Labels": Tax[2]}
                                      )
        sum_per_class   = FullClasses.groupby("New-Labels")["frecuency"].sum()
        
        # Summary
        print(100*"_")
        print("Information on previous and new classes")
        print(100*"_")
        print(FullClasses)
        print(100*"_")
        print(f"{sum_per_class}")
        print(100*"_")

        # Replacing labels using masks
        replace_nonsn    = [1,2,4,7,8,10,11,12]
        replace_sn       = [3,5,6,9]
        
        # Removing Classes if needed
        remove_classes = True

        if remove_classes:

            index_class = np.where(labels == 10)
            timesX      = np.delete(timesX, index_class, axis = 0)
            X           = np.delete(X, index_class, axis = 0)
            y           = np.delete(y, index_class, axis = 0)
            labels      = np.delete(labels, index_class, axis = 0)
            object_ids  = np.delete(object_ids, index_class, axis = 0)
            
            # Imprimimos 
            print(100*"_")
            print(f"We deleted the following information")
            print(100*"_")
            print(FullClasses.loc[FullClasses["y-Labels"]==10])
            print(100*"_")

            # Updated mask
            replace_nonsn  = [1,2,4,7,8,11,12]
        
        # Masking
        mask_nonsn_1    = np.isin(y, replace_nonsn)
        y[mask_nonsn_1] = 2

        mask_nonsn_2 = np.isin(labels, replace_nonsn)
        labels[mask_nonsn_2] = 2

        mask_sn_1    = np.isin(y, replace_sn)
        y[mask_sn_1] = 1

        mask_sn_2    = np.isin(labels, replace_sn)
        labels[mask_sn_2] = 1

        # Summary
        print(100*"_")
        print(f"The new values on the y-file are:")
        print(100*"_")
        print(f"{np.unique(y)}   |  [Preexplosion, SNIa, non-SNIa]")
        print(100*"_")

        print(100*"_")
        print(f"The new-values on the labels-files are_")
        print(100*"_")
        print(f"{np.unique(labels)}   |  [SNIa, non-SNIa]")
        print(100*"_")

    # Spliting with metadata
    if metadatos:
        X_train, X_val, y_train, y_val, X_test, y_test, timesX_test = ds.Split_Meta_Data(n_splits, 
                                                                                        metadata, 
                                                                                        X, 
                                                                                        y, 
                                                                                        timesX, 
                                                                                        object_ids,
                                                                                        labels,
                                                                                        save = "") 
    # Spliting from scrach
    else: 
        X_train, X_val, y_train, y_val, X_test, y_test, timesX_test = ds.Split_nonMeta_Data(n_splits, 
                                                                                            X, 
                                                                                            y, 
                                                                                            timesX, 
                                                                                            object_ids,
                                                                                            labels, 
                                                                                            save = path_mdata)
        
    # Tensor format (pytorch)
    X_train           = torch.tensor(X_train)
    y_train_onehot    = torch.tensor(y_train) 
    X_val             = torch.tensor(X_val) 
    y_val_onehot      = torch.tensor(y_val) 
    X_test            = torch.tensor(X_test) 
    y_test_onehot     = torch.tensor(y_test) 

    # Resize matrix
    y_train    = torch.argmax(y_train_onehot, dim=-1)
    y_val      = torch.argmax(y_val_onehot, dim=-1)
    y_test     = torch.argmax(y_test_onehot, dim=-1)

    # Summary
    print(100*"_")
    print("Shape: Training")
    print(100*"_")
    print(f"X_train:                  | {X_train.shape}")
    print(f"y_train:                  | {y_train.shape}")
    print(100*"_")

    print(100*"_")
    print("Shape: Validation")
    print(100*"_")
    print(f"X_val:                    | {X_val.shape}")
    print(f"y_val:                    | {y_val.shape}")
    print(100*"_")

    print(100*"_")
    print("Shape : Test")
    print(100*"_")
    print(f"X_test:                   | {X_test.shape}")
    print(f"y_test:                   | {y_test.shape}")
    print(100*"_")

    # Dataloader parameters
    batch_size = 256
    one_hot    = False
    model      = "rapid"

    # List for K-folds
    data_loaders = []

    # Creating dataloaders
    for split in range(n_splits):

        data_loaders.append(dl.ZTFDataLoaders(X_train[split], y_train[split], 
                                              X_val[split]  , y_val[split], 
                                              X_test        , y_test, 
                                              one_hot, 
                                              batch_size  = batch_size, 
                                              num_workers = 0,
                                              shuffle     = False, 
                                              collate_fn  = False, 
                                              normalize   = False, 
                                              n_quantiles = 1000, 
                                              weight_norm = False))
        
    print(100*"_")
    print(f"ZTFDataLoaders was successfully executed using {n_splits}-folds")
    print(100*"_")

    # Checking input size and number of classes
    input_size  = data_loaders[0].train_set.dataset.data[0].size(1)
    num_classes = data_loaders[0].train_set.dataset.labels.unique().size(0)
 
    print(f"Input-size:        {input_size}")
    print(f"Numero de clases:  {num_classes}")
    print(100*"_")

##############################################################################################################
if load_split:
    """
    __________________________________________________________________________________________________________
    Here we load pre-splited data, transform the atributes and the labels to tensor format, and finally create
    a dataloader.  
    __________________________________________________________________________________________________________
    """

    # Paths
    path_test  = path_mdata+f"{F_test}/{sF_test}/"

    path_train = path_mdata+f"{F_train}/"

    # Numpy array specifications per dataset
    if (Dataset == "ZTF"):
        n_obs    = 50         
        n_train  = 1681       
        n_val    = 187        
        n_test   = 200        
    
    elif (Dataset == "PLAsTiCC"):
        n_obs    = 50         
        n_train  = 15792      
        n_val    = 3948       
        n_test   = 2194       
    
    # Empty numpy arrays (n_split, sne, obs, band)
    X_train = np.empty(shape=(n_splits, n_train, n_obs, 2), dtype="float64") 
    X_val   = np.empty(shape=(n_splits, n_val, n_obs, 2), dtype="float64")

    # Empty numpy arrays (n_split, sne, obs, class)
    y_train = np.empty(shape=(n_splits, n_train, n_obs, 3), dtype="float64")
    y_val   = np.empty(shape=(n_splits, n_val, n_obs, 3), dtype="float64")

    # Load training and validation
    for i in range(n_splits):

        X_train[i] = np.load(path_train+"fold_"+str(i)+"/train/X.npy", allow_pickle=True)
        y_train[i] = np.load(path_train+"fold_"+str(i)+"/train/y.npy", allow_pickle=True)
        
        X_val[i]   = np.load(path_train+"fold_"+str(i)+"/validation/X.npy", allow_pickle=True)
        y_val[i]   = np.load(path_train+"fold_"+str(i)+"/validation/y.npy", allow_pickle=True)

    # Loading test
    X_test      = np.load(path_test+"X.npy", allow_pickle=True)
    y_test      = np.load(path_test+"y.npy", allow_pickle=True)
    timesX_test = np.load(path_test+"timesX.npy", allow_pickle=True)

    # Tensor format
    X_train           = torch.tensor(X_train)
    y_train_onehot    = torch.tensor(y_train) 
    X_val             = torch.tensor(X_val) 
    y_val_onehot      = torch.tensor(y_val) 
    X_test            = torch.tensor(X_test) 
    y_test_onehot     = torch.tensor(y_test) 

    # Se-size matrix
    y_train    = torch.argmax(y_train_onehot, dim=-1)
    y_val      = torch.argmax(y_val_onehot, dim=-1)
    y_test     = torch.argmax(y_test_onehot, dim=-1)

    # Summary
    print(100*"_")
    print("Shape: Training")
    print(100*"_")
    print(f"X_train:                  | {X_train.shape}")
    print(f"y_train:                  | {y_train.shape}")
    print(100*"_")

    print(100*"_")
    print("Shape: Validation")
    print(100*"_")
    print(f"X_val:                    | {X_val.shape}")
    print(f"y_val:                    | {y_val.shape}")
    print(100*"_")

    print(100*"_")
    print("Shape: Test")
    print(100*"_")
    print(f"X_test:                   | {X_test.shape}")
    print(f"y_test:                   | {y_test.shape}")
    print(100*"_")

    # Dataloader parameters
    batch_size = 256
    one_hot    = False
    model      = "rapid"

    # List
    data_loaders = []

    # Create dataloaders 
    for split in range(n_splits):
        data_loaders.append(dl.ZTFDataLoaders(X_train[split], y_train[split], 
                                              X_val[split]  , y_val[split], 
                                              X_test        , y_test, 
                                              one_hot, 
                                              batch_size  = batch_size, 
                                              num_workers = 0,
                                              shuffle     = False, 
                                              collate_fn  = False, 
                                              weight_norm = False))

    # Summary
    print(100*"_")
    print(f"ZTFDataLoaders was successfully executed using {n_splits}-folds")
    print(100*"_")

    # Checking input size and number of classes
    input_size  = data_loaders[0].train_set.dataset.data[0].size(1)
    num_classes = data_loaders[0].train_set.dataset.labels.unique().size(0)
 
    print(f"Input-size:        {input_size}")
    print(f"Numero de clases:  {num_classes}")
    print(100*"_")

##############################################################################################################
if plot_data:
    
    # IDs test data
    object_ids_test = np.load(path_mdata+f"{F_test}/{sF_test}/object_ids.npy",allow_pickle=True)

    # Plot the curve directly from the numpy data
    up.plot_lc_from_X(X_test[21],
                      timesX_test[21],
                      object_ids_test[21],
                      y_test.argmax(axis=-1), 
                      path = path_fig)
    
# Entrenamiento del Modelo ###################################################################################
if train_model:

    # Configuration of the model   
    CONFIG = conf.get_args()

    print(100*"_")
    print("Hyperparameters and configuration (conf file)")
    print(100*"_")
    print(CONFIG)
    print(100*"_")
    
    # Lists for models
    Full_models       = []
    
    # Lists for data
    train_loss_splits = []
    val_loss_splits   = []
    train_acc_splits  = []
    val_acc_splits    = []

    for split in range(n_splits):
        
        # Define path
        CONFIG.experiment = f"/{experiment}_split_{split}/"

        # Generate model
        rapid_model = bm.model_type(CONFIG)
        
        # Summary
        print(100*"_")
        print(rapid_model)
        print(100*"_")
        
        print(100*"_")
        print("Start the training")
        print(100*"_")

        rapid_model.fit(data_loaders[split].train_set, 
                        None, 
                        None, 
                        None,
                        data_loaders[split].val_set, 
                        'normal', 
                        CONFIG.num_epochs, 
                        CONFIG.verbose)
    
        print(100*"_")

        # Append trained model
        Full_models.append(rapid_model)

        # Extract loss 
        train_loss    = rapid_model.checkpoint.train_loss
        val_loss      = rapid_model.checkpoint.val_loss

        # Append loss
        train_loss_splits.append(train_loss)
        val_loss_splits.append(val_loss)

        # Extract accuracy
        train_acc     = rapid_model.checkpoint.train_acc
        val_acc       = rapid_model.checkpoint.val_acc

        # Append accuracy
        train_acc_splits.append(train_acc)
        val_acc_splits.append(val_acc)

    if (n_splits == 1):
        # Graficamos los resultados
        labels_curves = ['Training', 'Validation']
        color_curves  = ['firebrick', 'gold']

        # Analizamos la perdida del modelo
        metric        = 'Loss'

        # Extraemos la perdida de entrenamiento y validacion
        train_loss    = rapid_model.checkpoint.train_loss
        val_loss      = rapid_model.checkpoint.val_loss

        # Ploteamos
        up.plot_learning_curves(train_loss, val_loss, metric, labels_curves, color_curves, loc = "best", path_save= path_fig)

        # Analizar la precision del modelo
        metric        = 'Accuracy'
        
        # Extraemos la precision del entrenamiento y validacion
        train_acc     = rapid_model.checkpoint.train_acc
        val_acc       = rapid_model.checkpoint.val_acc

        # Ploteamos
        up.plot_learning_curves(train_acc, val_acc, metric, labels_curves, color_curves, loc = "best", path_save= path_fig)

    else:
        # Graficamos los resultados
        labels_curves = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
        color_curves  = ['chocolate' ,'firebrick', 'orange' ,'gold', 'olivedrab']
        style_curves  = ["solid", "dotted", "dashed", "dashdot", "dashed"]
        
        # Analizamos la perdida del modelo
        metric        = 'Loss'

        up.plot_learning_splits(train_loss_splits,
                                "Training",
                                metric,
                                labels_curves,
                                color_curves,
                                style_curves,
                                "best",
                                path_fig
                                )
        
        up.plot_learning_splits(val_loss_splits,
                                "Validation",
                                metric,
                                labels_curves,
                                color_curves,
                                style_curves,
                                "best",
                                path_fig
                                )

        # Analizar la precision del modelo
        metric        = 'Accuracy'
        
        up.plot_learning_splits(train_acc_splits,
                                "Training",
                                metric,
                                labels_curves,
                                color_curves,
                                style_curves,
                                "best",
                                path_fig
                                )
        
        up.plot_learning_splits(val_acc_splits,
                                "Validation",
                                metric,
                                labels_curves,
                                color_curves,
                                style_curves,
                                "best",
                                path_fig
                                )
    
# Cargamos un modelo previo ##################################################################################
if upload_model:

    # Creamos la configuracion del modelo    
    CONFIG = conf.get_args()

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
        rapid_model = bm.model_type(CONFIG)
        
        print(100*"_")
        print(f"Se cargaron correctamente los pesos:         {CONFIG.model_path}")
        print(100*"_")

        # Guardamos el modelo
        Full_models.append(rapid_model)
    
    print(Full_models)

# Predecimos #################################################################################################
if test_model:
    # Creamos listas para guardar los datos 
    loss_models        = []
    acc_models         = []
    # Predicciones binarias para toda la curva
    y_pred_models      = []
    labels_models      = []
    # Metrics scores
    loss_models        = [] 
    acc_models         = []
    acc_bal_models     = []
    prec_models        = []
    rec_models         = []
    f1_models          = []
    cm_models          = []

    for split in range(n_splits):
        # Extraemos perdida y precision del modelo
        loss        , _   = Full_models[split].evaluate(data_loaders[split].test_set, 
                                                          data_loaders[split].test_set.batch_size)
        # Guardamos
        loss_models.append(loss)
        
        # Realizamos las predicciones con los datos de test
        y_pred_prob, y_pred = Full_models[split].predict(data_loaders[split].test_set, 
                                                         data_loaders[split].test_set.batch_size)

        # Definimos los labels a utilizar
        dict_labels = {1: 'SNIa', 2: 'nonSNIa', 0: "Prexplosion"}
        
        # Imprimimos informacion desde las prediciones 
        print(100*"_")
        print(f"K-fold {split}: La prediccion de datos fue realizada correctamente")
        print(100*"_")
        print(f"Shape y_pred_prob:              {y_pred_prob.shape}")
        print(f"Shape y_pred:                   {y_pred.shape}")
        print(f"Shape y_test:                   {y_test.numpy().shape}")
        print(100*"_")

        # Imprimimos las metricas obtenidas con los datos de test
        print(100*"_")
        print(f"K-fold {split}: Resultados del modelo con el conjunto de datos de test")
        print(100*"_")
        um.print_metrics_pbp_lc(y_test, y_pred, loss, one_hot)
        print(100*"_")
    
        # Definimos las predicciones generales
        _, pre_exp_p = dh.get_clf_for_RAPID(y_pred)
        
        # Filtramos aquellas clasificadas como pre-explosiones
        if (len(pre_exp_p) != 0):
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
                
        # Extraemos las predicciones binarias
        y_pred_class, _   = dh.get_clf_for_RAPID(y_pred)
        y_pred_class = np.array(y_pred_class, dtype="int64")
        
        # Cargamos los labels
        labels_test = np.load(path_test+"labels.npy",allow_pickle=True)
        labels_test = np.array(labels_test, dtype="int64")
        
        # guardamos en listas
        y_pred_models.append(y_pred_class)
        labels_models.append(labels_test)
        
        # Extraemos las metricas
        acc, acc_bal, prec, rec, f1, cm = um.get_metrics(labels_test, y_pred_class)
        
        # Guardamos en las listas
        acc_models.append(acc)    
        acc_bal_models.append(acc_bal)
        prec_models.append(prec)
        rec_models.append(rec)
        f1_models.append(f1)
        cm_models.append(cm) 

    print(acc_models)
    print(acc_bal_models)
    print(prec_models)
    print(rec_models)
    print(f1_models)
    print(cm_models)


    if (n_splits == 1):
        # Imprimimos las metricas
        print(100*"_")
        print("Metricas finales")
        print(100*"_")
              
        print(f"loss:      {loss_models[0]:.3f}")
        print(f"accuracy:  {acc_models[0]:.3f}")
        print(f"acc_bal:   {acc_bal_models[0]:.3f}")
        print(f"precision: {prec_models[0]:.3f}")
        print(f"recall:    {rec_models[0]:.3f}")
        print(f"f1:        {f1_models[0]:.3f}")
        print(100*"_")

        
        # varios
        Title   = f"Confusion Matrix: {Dataset} with RAPID model"
        Figsize = (13, 7)
        Labels  = ['SNIa', 'nonSNIa']
        
        up.plot_confusion_matrix(cm_models[0], Labels, Figsize, normalize=True, title=Title, path=path_fig)    
    
    else:
        # Imprimimos las metricas
        print(100*"_")
        print("Metricas generales y desviacion estandar")
        print(100*"_")
              
        save_losses           = np.array(loss_models)
        save_accuracy         = np.array(acc_models)
        save_accuracy_balance = np.array(acc_bal_models)
        save_precision        = np.array(prec_models)
        save_recall           = np.array(rec_models)
        save_f1               = np.array(f1_models)

        print(f"loss:      {save_losses.mean(axis=0):.3f}  +-  {save_losses.std(axis=0):.3f}")
        print(f"accuracy:  {save_accuracy.mean(axis=0):.3f}  +-  {save_accuracy.std(axis=0):.3f}")
        print(f"acc_bal:   {save_accuracy_balance.mean(axis=0):.3f}  +-  {save_accuracy_balance.std(axis=0):.3f}")
        print(f"precision: {save_precision.mean(axis=0):.3f}  +-  {save_precision.std(axis=0):.3f}")
        print(f"recall:    {save_recall.mean(axis=0):.3f}  +-  {save_recall.std(axis=0):.3f}")
        print(f"f1:        {save_f1.mean(axis=0):.3f}  +-  {save_f1.std(axis=0):.3f}")
        print(100*"_")

        print(f"cm:         {cm}")
        # Creamos la matriz de confusion
        cm_mean = np.mean(cm_models, axis = 0)
        cm_std  = np.std(cm_models, axis = 0)
        
        print(f"cm_mean:    {cm_mean}")
        print(f"cm_std:     {cm_std}")

        # varios
        Title   = f"Confusion Matrix: {Dataset} with RAPID model"
        Figsize = ((cm_mean.shape[1] + 1)*2 , cm_mean.shape[0]*2)
        Labels  = ['SNIa', 'nonSNIa']
        
        up.plot_confusion_matrix_statics(cm_mean, 
                                         cm_std, 
                                         Labels, Labels, 
                                         Figsize, title=Title, 
                                         normalize=False, 
                                         path=path_fig )

print(100*"_")
print("El programa se ejecutó correctamente")
print(100*"_")

################################################################################################################