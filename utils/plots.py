"""
______________________________________________________________________________________________________________
Functions and utilities to plot the data:
______________________________________________________________________________________________________________
- plot_learning_curves:        
    -> Objective:
    -> Used:     
    -> Input:    
    -> Return:   

- plot_learning_splits:  
    -> Objective:
    -> Used:     
    -> Input:    
    -> Return:   

- plot_cm:
    -> Objective:              
    -> Used:     
    -> Input:    
    -> Return:   

- plot_confusino_matrix:
    -> Objective:              
    -> Used:     
    -> Input:    
    -> Return:

- plot_confusion_matrix_statics :
    -> Objective:              
    -> Used:     
    -> Input:    
    -> Return:   

- plot_lc_from_X:
    -> Objective:  Plot preprocessed data.            
    -> Used:       RAPID model.
    -> Input:      X, X_time,oid, y, n=None, path = ""
    -> Return:     Example lightcurves from data.
    ______________________________________________________________________________________________________________
"""

# Modulos ####################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Functions ##################################################################################################
from sklearn.metrics import confusion_matrix

##############################################################################################################
def plot_learning_curves(train, val, metric, labels_curves, color_curves, loc='upper right', path_save=''):
    
    curves = [train, val]
    epochs = range(1, len(train) + 1)

    plt.figure(figsize=(11,8))

    for i in range(len(curves)):
        plt.plot(epochs, curves[i], color=color_curves[i], label=f'{labels_curves[i]} {metric.lower()}') #, linewidth=2.5, markersize=7)

    plt.xticks(fontsize = 15)
    plt.yticks(size = 15) # Arreglar este tamaño

    plt.title(f'Training and validation {metric.lower()}', size='20', pad=15)
    plt.xlabel('Epochs', fontsize='20', labelpad=15)
    plt.ylabel(f'{metric}', fontsize='20', labelpad=15)

    legend = plt.legend(loc=loc, shadow=True, fontsize='15')
    legend.get_frame().set_facecolor('white')
    
    plt.savefig(path_save + f"learning_curve_{metric}.png", bbox_inches="tight")

    plt.show()

##############################################################################################################
def plot_learning_splits(lists, stage ,metric, labels_curves, color_curves, style_curves,loc='', path_save=''):
    
    curves = lists
    
    plt.figure(figsize=(11,8))

    for i in range(len(curves)):
        
        epochs = range(1, len(curves[i]) + 1)
        plt.plot(epochs, curves[i], color=color_curves[i], ls = style_curves[i], label=f'{labels_curves[i]}') 

    plt.xticks(fontsize = 15)
    plt.yticks(size = 15) # Arreglar este tamaño
    plt.title(f'{stage} {metric}', size='20', pad=15, loc="left")
    plt.xlabel('Epochs', fontsize='20', labelpad=15)
    plt.ylabel(f'{metric}', fontsize='20', labelpad=15)
    plt.grid(alpha= 0.7)

    legend = plt.legend(loc=loc, shadow=True, fontsize='15')
    legend.get_frame().set_facecolor('white')
    
    plt.savefig(path_save + f"learning_curve_{stage}_{metric}.png", bbox_inches="tight")

    plt.show()

##############################################################################################################
def plot_cm(y_true, y_pred, label_order, dict_labels, figsize=(12, 10), save_path=""):
    
    y_true, y_pred = replace_for_visualization(y_true, y_pred, dict_labels)
    
    cm_matrix = confusion_matrix(y_true, y_pred, 
                                 labels=label_order)
                 
    plot_confusion_matrix(cm_matrix, label_order, figsize, path = save_path)

##############################################################################################################
def plot_confusion_matrix(cm, classes, figsize,
                          normalize=False,
                          title="",
                          cmap=plt.cm.Blues,
                          path=""):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, loc="left", fontsize=17)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize = 17)
    plt.yticks(tick_marks, classes, fontsize = 17)

    fmt =  'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%d"%  (cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 16)

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 18)
    plt.xlabel('Predicted label', fontsize = 18)
    plt.savefig(path+"cm_matrix.png", bbox_inches="tight")

##############################################################################################################
def plot_confusion_matrix_statics(cm, cms,  classes_true, classes_pred, fgs,
                          cmap=plt.cm.Blues, title = "", normalize=False, path=""):

    plt.figure(figsize = fgs)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, loc="left", fontsize=12)
    plt.colorbar()

    if normalize:
        cm  = cm.astype("float")  / cm.sum(axis=1)[:,np.newaxis]
        cms = cms.astype("float") / cms.sum(axis=-1)[:,np.newaxis]
        cm  = cm.round(decimals=4)
        cms = cms.round(decimals=4)
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix,without normalization")

    thresh = cm.min() + (cm.max()-cm.min()) / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]) + '\n$\pm$' + '{0:.2f}'.format(cms[i, j]),
                     horizontalalignment = "center",
                     verticalalignment   = "center", fontsize=12,
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)

    plt.savefig(path+"cm_matrix.png", bbox_inches="tight")

##############################################################################################################
def replace_for_visualization(y_true, y_pred, dict_labels):

    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    for idx in range(len(y_true)):
        y_true[idx] = dict_labels[y_true[idx]]
        y_pred[idx] = dict_labels[y_pred[idx]]
    return y_true, y_pred

##############################################################################################################
def plot_example_curve(obid, Dataframe, bands, colors, path):
	# Creamos la figura
    fig = plt.figure(figsize=(12,20))
    gs  = fig.add_gridspec(4,1, hspace=0.1, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
	# Filtramos y ploteamos
    axs[0].set_title(f"Example Light Curve - SPCC Data - Raw - oid: {obid}", loc="left", size = 16)
    for a in range(0, len(bands), 1):
	    # Filtro de id
        fil_id = Dataframe["oid"].isin([obid])
	    # Filtro de banda
        fil_bd = Dataframe["fid"].isin([bands[a]])
	    # Datos a plotear
        light_curve = Dataframe[fil_id & fil_bd]
	    # Ploteamos
        axs[a].errorbar(light_curve["mjd"], light_curve["flux_tot_ujy"], light_curve["fluxunc_tot_ujy_resc"], 
	                fmt=".", c = "black", linestyle = "", elinewidth=0.6, capsize=3)
        axs[a].plot(light_curve["mjd"], light_curve["flux_tot_ujy"], c = colors[a], 
	                label = f"{bands[a]}-band")
        axs[a].set_ylabel(r"$F$ [$Jy$]", size = 14)
        axs[a].legend(loc = "best")
        axs[a].grid()
    axs[3].set_xlabel("Time [MJD]", size = 14)
    plt.savefig(path+"Raw_SPCC_Ex_Light_curve.png", bbox_inches="tight")

##############################################################################################################
def plot_example_augm(obid, Dataframe, path):
    fig = plt.figure(figsize=(12,5))
    axs = fig.add_subplot(111)
    # Filtramos
    fil_id = Dataframe["oid"].isin([obid])
    Dataframe = Dataframe[fil_id]
    # Green
    axs.errorbar(Dataframe["mjd"], Dataframe["g_flux"], Dataframe["g_error"], fmt="", c = "black", linestyle = "", elinewidth=0.6)
    axs.scatter(Dataframe["mjd"], Dataframe["g_flux"], marker=".",c = "green", label = r"$g$-band")
    # red
    axs.errorbar(Dataframe["mjd"], Dataframe["r_flux"], Dataframe["r_error"], fmt="", c = "black", linestyle = "", elinewidth=0.6)
    axs.scatter(Dataframe["mjd"], Dataframe["r_flux"], marker=".",c = "red", label = r"$r$-band")
    # i
    axs.errorbar(Dataframe["mjd"], Dataframe["i_flux"], Dataframe["i_error"], fmt="", c = "black", linestyle = "", elinewidth=0.6)
    axs.scatter(Dataframe["mjd"], Dataframe["i_flux"], marker=".", c = "violet", label = r"$i$-band")
    # z
    axs.errorbar(Dataframe["mjd"], Dataframe["z_flux"], Dataframe["z_error"], fmt="", c = "black", linestyle = "", elinewidth=0.6)
    axs.scatter(Dataframe["mjd"], Dataframe["z_flux"], marker=".", c = "black", label = r"$z$-band")
    axs.set_ylabel(r"$F$ [$Jy$]", size = 14)
    axs.set_xlabel(r"Time [Days]", size = 14)
    axs.legend(loc = "best")
    axs.set_title(f"Example Light Curve - SPCC Data - Processed - oid: {obid}", loc="left", size = 16)
    axs.grid()
    plt.savefig(path+"/Raw_SPCC_Ex_augmented.png", bbox_inches="tight")

##############################################################################################################
def plot_lc_from_X(X, X_time, oid, y, path = ""):
    
    print(f"SNs:   {oid}")
    
    X = X.reshape((X.shape[1] , X.shape[0]))
    
    plt.figure(figsize=(12,5))
    plt.scatter(X_time,X[0],marker='.', linestyle='None',color="g", label = r"$g$-band")
    plt.scatter(X_time,X[1],marker='.', linestyle='None',color="r", label = r"$r$-band")
    plt.axvline(x =0, c = "gray", label = r"$t_0$")    
    plt.title(f"Object ID:   {oid}", loc="left")
    plt.xlabel("Days since trigger")
    plt.ylabel("Relative Flux")
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig(path+oid+".png")

##############################################################################################################

