##############################################################################################################
#                 Programa para preprocesar, dividir y entrenar el modelo de Charnock                        #
##############################################################################################################
# Programa original extraído del trabajo de Charnock y Moss (2017)                                           #
# Con modificaciones por Daniel Moreno y Jorge Saavedra.                                                     #
##############################################################################################################
# Actualmente está creado para aceptar datos preprocesados provenientes desde SPCC                           #
# ############################################################################################################

print("Charnock con SPCC:")

# Modulos a Importar #########################################################################################
import glob
import torch
import numpy             as np
import pandas            as pd
import random            as rm
import matplotlib.pyplot as plt
print("___________________________________________________________________")
# Programas y funciones creadas por dani (tuneadas por mi) ###################################################
import utils.preprocess         as prep  
import utils.plots              as up    
import utils.metrics            as um    
import utils.data_loaders       as dl    
import CONF.Charnock.conf       as conf  
import models.base_model           
print("___________________________________________________________________")
# Seleccion de procesos ######################################################################################
preprocess_data  = False  # Procesamiento de los datos.
aumentation_data = False  # Aumegntacion de los datos.
plot_data        = False  # Graficos ejemplo de datos.
save_split       = False  # Dividir los datos en entrenamiento y testeo.
load_split       = True   # Cargar los archivos ya divididos.
train_model      = False  # Entrenar Modelo.
n_folds          = False  # Si queremos o no realizar fold con el modelo (esta pensado para 5-folds).
upload_model     = True   # Cargar modelo previo (asegurate de revisar el programa conf).
test_model       = True   # Predecir con un conjunto de test.

# Imprimimos en la terminal
print("___________________________________________________________________")
print("Resumen de ejecucion                                ")
print("___________________________________________________________________")
print(f"Preprocesamiento:            | {preprocess_data}")
print(f"Augmentacion:                | {aumentation_data}")
print(f"Plots de Ejemplo:            | {plot_data}")
print(f"Division de datos:           | {save_split}")
print(f"Cargar datos divididos:      | {load_split}")
print(f"Entrenar modelo:             | {train_model}")
print(f"N_folds:                     | {n_folds}")
print(f"Cargar modelo:               | {upload_model}")
print(f"Testear modelo               | {test_model}")
print("___________________________________________________________________")


# Path a los archivos de datos ###############################################################################
path_data = "./data/SPCC/spcc_data_in_format.pkl"
path_save = "./data/SPCC/supervised/paper_reproduce"
path_fig  = "./figures/Charnock"

# Preprocesamiento ###########################################################################################

# Hiperparámetros para procesar los datos 
key_types  = {'Ia':0, 'II':1, 'Ibc':1, 'IIn':1, 'IIP':1, 'IIL':1, 'Ib':1, 'Ic':1}    # Clasificacion binaria
bands_name = ['g', 'r', 'i', 'z']                                                    # Bandas a observar

print("___________________________________________________________________")
print("Clases: [SNIa - Non-SNIa]                               ")
print("___________________________________________________________________")
print(f"Bandas {bands_name}:           ")

# booleanos
normalize          = False                                                                    
online             = False
reproduce_charnock = True

# Hiperparámetros para augmentar los datos
num_augments = 5                                                                 # Cantidad de augmentaciones
mask_path = path_save + '/mask_data.csv'                                         # Path para guadar el mask
data_path = path_save + '/data_preprocessed.csv'                                 # Path a los datos procesados
col_data = ['oid', 'mjd', 'ra', 'dec', 'mwebv', 'photo_z', 'g_flux', 'g_error',  # Columnas de la tabla
            'r_flux', 'r_error', 'i_flux', 'i_error', 'z_flux', 'z_error', 
            'sim_redshift', 'class']
#fil_col_data = ['ra', 'dec', 'mwebv', 'photo_z', 'sim_redshift']                # Columnas para filtrar

# Aqui procesamos los datos
if preprocess_data:
    # Se importa la clase 
    prep.CharnockPreprocess(key_types, path_data, path_save, bands_name, 
                                  normalize, reproduce_charnock, online)

    print("___________________________________________________________________")
    print("El procesamiento fue realizado")
    print("___________________________________________________________________")

# Aqui augmentamos los datos
if aumentation_data:
    # Se importa la funcion 
    prep.charnock_augmentation(num_augments, mask_path, data_path, path_save,
                                 col_data, bands_name)
    print("___________________________________________________________________")
    print("La aumentacion fue realizada")
    print("___________________________________________________________________")

# Plots de ejemplo de las curvas #############################################################################

# Tabla de datos sin augmentacion
Tab_Orig_Data = pd.read_pickle(path_data)

# Ramdon File
augmented = "/Data-Augmented/data_augmented_4.csv"

# Tabla de datos augmentados
Tab_Augm_Data = pd.read_csv(path_save+augmented, names=col_data, header= None)

# Supernova a analizar
obid = rm.choice(Tab_Orig_Data["oid"].unique())

# listas
Colors = ["green", "red", "violet", "black"]

# Ploteamos  
if plot_data:
    # Una curva ramdon en cuatro bandas
    up.plot_example_curve(obid, Tab_Orig_Data, bands_name, Colors, path_fig)
    # Imprimimos
    print("___________________________________________________________________")
    print("Gráficos de ejemplo:")
    print("___________________________________________________________________")
    print("Plot de Prueba 1:        Realizado")
    # Las curvas despues de augmentar los datos
    up.plot_example_augm(obid, Tab_Augm_Data, path_fig)
    # Imprimimos
    print("Plot de Prueba 2:        Realizado")
    print("___________________________________________________________________")

# Separacion de datos #########################################################################################

# Definimos un camino a los csv
path = path_save + "/Data-Augmented/" +"data_augmented_"

# Leemos los archivos
files_unblind_hostz = glob.glob(path + '*.csv')

# Imprimimos
print("___________________________________________________________________")
print("Los Archivos dedatos que se utilizaran son:")
print("___________________________________________________________________")
for a in files_unblind_hostz:
    print(a)
print("___________________________________________________________________")

# Creamos una lista para albergar los datos
data_augments = []

# Leemos y guardamos
for filename in files_unblind_hostz:
    reader = pd.read_csv(filename, header=None, sep=",")
    reader.columns = col_data
    # Si queremos filtar los datos, descomentar esta linea
#    reader = reader.drop(fil_col_data, axis=1)

    data_augments.append(reader)

print("___________________________________________________________________")
print(f"Cantidad de archivos:           {len(data_augments)}")
print(f"Largo de de unblind_nohostz_1:  {len(data_augments[0])}")
print(f"Cantidad de supernovas:         {len(set(data_augments[0].oid))}")
print("___________________________________________________________________")

# Cantidad de supernovas que quedaran para el entrenamiento, validacion y testeo
val_fraction  = 0.2
test_fraction = 0.3

# id's
ids_sne        = data_augments[0].oid.unique()
ids_sne_length = len(ids_sne)
test_length    = int(ids_sne_length * test_fraction)
val_lenght     = int((ids_sne_length-test_length) * val_fraction)

# Indices permutados de los datos de ids_sne_length
indices        = np.random.permutation(ids_sne_length) 

# Indice de los datos que se entrenaran y testearan
training_idx = indices[:ids_sne_length-test_length-val_lenght] 
val_idx      = indices[ids_sne_length-test_length-val_lenght:ids_sne_length-test_length]
test_idx     = indices[ids_sne_length-test_length:]

# ids de supernovas para entrenar y testear
ids_train = ids_sne[training_idx]
ids_val   = ids_sne[val_idx]
ids_test  = ids_sne[test_idx]

print("___________________________________________________________________")
print("Sobre la Separacion de datos")
print("___________________________________________________________________")
print(f'Fraccion de datos test:                                   {test_fraction}')
print(f'Cantidad de supernovas en el conjunto de entrenamiento:   {len(ids_train)}')
print(f'Cantidad de supernovas en el conjunto de validacion:      {len(ids_val)}')
print(f'Cantidad de supernovas en el conjunto de prueba:          {len(ids_test)}')
print("___________________________________________________________________")

# Realizamos el split de datos:
if save_split:
    # Nombres de archivos para entrenamiento y testeo
    name_train_s = path_save+"/Data-split/pytorch_training_1.txt"
    name_val_s   = path_save+"/Data-split/pytorch_validation_1.txt" 
    name_test_s  = path_save+"/Data-split/pytorch_test_1.txt" 
    # Se guardan los archivos
    np.savetxt(name_train_s, ids_train, fmt='%s')
    np.savetxt(name_val_s, ids_val, fmt='%s')    
    np.savetxt(name_test_s, ids_test, fmt='%s')
    # Imprimimos
    print("___________________________________________________________________")
    print(f"Se han creado los archivos  ")
    print(f"Entrenamiento:   {name_train_s}")
    print(f"Validacion:      {name_test_s}")
    print(f"Testeo:          {name_test_s}")
    print("___________________________________________________________________")


# Cargamos los datos si queremos:
if load_split:
    # Nombre de los archivos
    name_train = "pytorch_training_1.txt"
    name_val   = "pytorch_validation_1.txt"
    name_test  = "pytorch_test_1.txt"
    
    # Path a los archivos
    path_split = path_save + "/Data-split/"
    
    # Leemos entrenamiento
    my_file_train      = open(path_split+name_train, "r")
    train_content      = my_file_train.read()
    train_content_list = train_content.split("\n")
    my_file_train.close()
    # Imprimimos
    print("___________________________________________________________________")
    print(f"Se leyo correctamente el archivo:     {name_train}")
    
    # Leemos Validacion
    my_file_val        = open(path_split+name_val, "r")
    val_content        = my_file_val.read()
    val_content_list   = val_content.split("\n")
    my_file_val.close()
    # Imprimimos
    print(f"Se leyo correctamente el archivo:     {name_val}")

    # Leemos testeo
    my_file_test       = open(path_split+name_test, "r")
    test_content       = my_file_test.read()
    test_content_list  = test_content.split("\n")
    my_file_test.close()
    # Imprimimos
    print(f"Se leyo correctamente el archivo:     {name_test}")

    # Contabilizamos
    ids_train = [int(i) for i in train_content_list[0:-1]]
    print(f"Datos de entrenamiento:               {len(ids_train)}")
    # Contabilizamos
    ids_Val   = [int(i) for i in val_content_list[0:-1]]
    print(f"Datos de entrenamiento:               {len(ids_val)}")
    # Contabilizamos
    ids_test  = [int(i) for i in test_content_list[0:-1]]
    print(f"Datos de test:                        {len(ids_test)}")
    print("___________________________________________________________________")

# Creamos las listas para guardar datos y etiquetas
data, labels   = [], []
training_idx   = []
validation_idx = [] 
test_idx       = []

idx = 0
for data_i in data_augments:
    for id in ids_sne:
        filter_sn = data_i[data_i.oid == id]
        data_sequence = filter_sn.iloc[0:, 1:-1:].values
        label = filter_sn.iloc[0,-1]

        labels.append(label)
        data.append(torch.tensor(data_sequence))

        # Indexa en que lugar de la data y labels se encuentran los datos de training validacion y testing
        if id in ids_train:
            training_idx.append(idx)
        elif id in ids_val:
            validation_idx.append(idx)
        elif id in ids_test:
            test_idx.append(idx)

        idx += 1

# Desordena la consecutividad de las supernovas aumentadas
rm.shuffle(training_idx)
rm.shuffle(validation_idx)
rm.shuffle(test_idx)

# Lectura de datos aumentados según los ids obtenidos para entrenamiento y test ############################

# Guardamos las etiquetas en un formato tensor
labels_torch = torch.tensor(labels)

# Creamos una variable que contabiliza las clases
nb_classes = labels_torch.unique().size(0)

# Datos para entrenamiento
X_train = []
for idx in training_idx:
    X_train.append(data[idx])
    
y_train        = labels_torch[training_idx]
y_train_onehot = torch.nn.functional.one_hot(labels_torch[training_idx], nb_classes)

# Datos para validacion
X_val = []
for idx in validation_idx:
    X_val.append(data[idx])

y_val        = labels_torch[validation_idx]
y_val_onehot = torch.nn.functional.one_hot(labels_torch[validation_idx], nb_classes)


# Datos para testeo
X_test = []
for idx in test_idx:
    X_test.append(data[idx])

y_test        = labels_torch[test_idx]
y_test_onehot = torch.nn.functional.one_hot(labels_torch[test_idx], nb_classes)

print("___________________________________________________________________")
print("Se separaron exitosamente los datos y se guardaron en formato", "\n", "de tensor (Pytorch)")
print("___________________________________________________________________")
print(f"Numero de Clases:                       {nb_classes}")
print(f'Cantidad de SNs:                        {len(data)}')
print(f'Cantidad de supernovas para training:   {len(X_train)}')
print(f'Cantidad de supernovas para validacion: {len(X_val)}')
print(f'Cantidad de supernovas para test:       {len(X_test)}')
print("___________________________________________________________________")

# Data loader ################################################################################################

# Bool sobre si queremos realizar un one_hot
one_hot = False

# Generamos un data load cargando la función desde el data loader
data_loader = dl.CharnockDataLoaders(X_train, y_train, X_val, y_val, X_test, y_test, one_hot, batch_size=10, 
                                                num_workers=0, shuffle=False, collate_fn=dl.collate_fn,
                                                weight_norm=True)
print("___________________________________________________________________")
print("CharnockDataLoaders se ejecuto exitosamente")

# Atributos y numero de clases en el set de datos
input_size  = data_loader.train_set.dataset.data[0].size(1)
num_classes = data_loader.train_set.dataset.labels.unique().size(0)

print(f"input_size : {input_size}")
print(f"num_classes: {num_classes}")
print("___________________________________________________________________")

# Entrenamiento del Modelo ####################################################################################

# Creamos la configuracino del modelo
CONFIG = conf.get_args()

print("___________________________________________________________________")
print("Hiperparametros y configuracion del modelo (conf file)             ")
print("___________________________________________________________________")
print(CONFIG)
print("___________________________________________________________________")


def model_type(CONFIG):
    
    if CONFIG.load_model:
        model = models.base_model.Model(CONFIG)
        model.summary()

    else:
        model = models.base_model.Model(CONFIG)
        model.summary()
        model.compile(CONFIG.loss_list, CONFIG.optimizer, CONFIG.lr, CONFIG.metric_eval)

    return model

charnock_model = model_type(CONFIG)

print("___________________________________________________________________")
print(charnock_model)
print("___________________________________________________________________")

if train_model:
    print("___________________________________________________________________")
    print("Comienza el entrenamiento")
    charnock_model.fit(data_loader.train_set, None, None, None,
                      data_loader.val_set, 'normal', CONFIG.num_epochs, CONFIG.verbose)
    print("Se entreno con exito")
    print("___________________________________________________________________")

    # Analizamos la perdida del modelo
    metric        = 'Loss'
    labels_curves = ['Training', 'Validation']
    color_curves  = ['firebrick', 'gold']

    # Extraemos la perdida de entrenamiento y validacion
    train_loss    = charnock_model.checkpoint.train_loss
    val_loss      = charnock_model.checkpoint.val_loss

    # Ploteamos
    up.plot_learning_curves(train_loss, val_loss, metric, labels_curves, color_curves, loc = "best", path_save= path_fig)

    # Analizar la precision del modelo
    metric        = 'Accuracy'
    labels_curves = ['Training', 'Validation']
    color_curves  = ['firebrick', 'gold']

    # Extraemos la precision del entrenamiento y validacion
    train_acc     = charnock_model.checkpoint.train_acc
    val_acc       = charnock_model.checkpoint.val_acc
    
    # Ploteamos
    up.plot_learning_curves(train_acc, val_acc, metric, labels_curves, color_curves, loc = "best", path_save= path_fig)

# Cargamos un modelo previo ###################################################################################
if upload_model:
    # Cargamos un checkpoint (esto hay que cambiarlo manual en el conf y aqui)
    checkpoint     = "charnockpt-002-0.9152.pt"
    charnock_model = model_type(CONFIG)
    print("___________________________________________________________________")
    print(f"Se cargaron correctamente los pesos:       {checkpoint}")
    print("___________________________________________________________________")


# Evaluamos el modelo con datos de testeo generando matrices de confusion #####################################

# Extraemos perdida y precision del modelo durante la validacion
loss, acc  = charnock_model.evaluate(data_loader.test_set, data_loader.test_set.batch_size)

# Predecimos con los datos de testeo
if test_model:
    # Realizamos las predicciones con los datos de test
    y_pred_prob, y_pred = charnock_model.predict(data_loader.test_set, data_loader.test_set.batch_size)

    # Definimos los labels a utilizar
    dict_labels = {0: 'SNIa', 1: 'nonSNIa'}

    # Imprimimos las metricas obtenidas con los datos de test
    print("___________________________________________________________________")
    print("Resultados del modelo con el conjunto de datos de test")
    print("___________________________________________________________________")
    um.print_metrics_full_lc(y_test, y_pred, y_pred_prob, loss, dict_labels , False)
    print("___________________________________________________________________")

    # Generamos una matriz de confusion
    label_order = ['SNIa', 'nonSNIa']
    figsize = (13, 7)
    up.plot_cm(y_test, y_pred, label_order, dict_labels, figsize, path_fig)

################################################################################################################