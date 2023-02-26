import argparse

# Configuration RAPID model

def get_args():

    parser = argparse.ArgumentParser()  
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to be used")
    parser.add_argument("--fold", type=int, default=0, 
                        help="Fold to be used")
    parser.add_argument("--normalize_source", type=bool, default=False,
                        help="Normalize the lightcurves of the source data")
    parser.add_argument("--normalize_target", type=bool, default=False,
                        help="Normalize the lightcurves of the target data")

    # Data --> no me esta sirviendo
    parser.add_argument("--num_augments", type=int, default=1,
                        help="Numero de aumentaciones de los datos")

    #••••••••••••••••••••••••••••#
    #•••••••    MODELO    •••••••#
    #••••••••••••••••••••••••••••#

    # Data Loader
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Tamanio de cada mini-batch")

    # Compile
    parser.add_argument("--loss_list", type=list,
                        default=['crossentropy'],
                        choices=[['crossentropy']],
                        help="Losses function a utilizar")
    # Tal vez sea necesario disminuir el lr.                    
    parser.add_argument("--lr", type=float, 
                        default=[0.00005, 1e-4], 
                        choices=[[0.00005], [1e-4]],
                        help="Learning rate")
                        
    parser.add_argument("--optimizer", type=str,
                        default=['adam', 'adam'], 
                        choices=[['adam']],
                        help="Optimizador")

    # Network arquitecture
    parser.add_argument("--rnn_type", type=str, default='GRU',
                        choices=['RNN', 'LSTM', 'GRU'], help="Unidad recurrente")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Probabilidad aplicada en la salida de cada RNN")
    parser.add_argument("--input_size", type=int,
                        default=2, help="Numero de features")
    parser.add_argument("--num_classes", type=int,
                        default=3, help="Numero de clases")
    parser.add_argument("--hidden_size", type=int,
                        default=100, help="Numero de neuronas")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Numero de capas ocultas")
    parser.add_argument("--activation", type=str, default='tanh',
                        help="Función de activación para la capa final")
    parser.add_argument("--batch_norm", type=bool,
                        default=False, help="Batch normalization")

    # Fit
    parser.add_argument("--type_train", type=str,
                        choices=['normal', 'kd', 'ft', 'mme'],
                        default='normal', help="Tipo de entrenamiento a utilizar")    
    parser.add_argument("--num_epochs", type=int,
                        default=10, help="Numero de epocas")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="Imprime la fase de entrenamiento")

    # Early stopping
    parser.add_argument("--patience", type=int, default=5,
                        help="Numero de epocas para activar early stoping")
    parser.add_argument("--metric_eval", type=str, default='accuracy',
                        help="Numero de epocas para activar early stoping")
    parser.add_argument("--name_model", type=str, default='rapid',
                        help="Nombre del modelo a utilizar")

    # Load and save experiment
    parser.add_argument("--experiment", type=str, default='/exp_class_0/', 
                        help="Nombre del experimento a realizar")
    parser.add_argument("--load_file", type=str, default='save_results',
                        help="file to load params from")
    parser.add_argument("--save_file", type=str, default='save_results',
                        help="file to save params to / load from if not loading from checkpoint")

    # Pre-trained model
    parser.add_argument("--load_model", type=bool,
                        default=False, help="Load pre-trained model") 
    parser.add_argument("--model_path", type=str, default='/best_result/rapidpt-385-0.9143.pt', 
                        help="relative path del modelo desde el experimento")

    # Domain Adaptation
    parser.add_argument("--source", type=str, default='PLAsTiCC',
                         choices=['snana', 'spcc', 'PLAsTiCC'],
                         help="Dataset to be used as source for the experiment.")
    parser.add_argument("--target", type=str, default='ZTF',
                         choices=['ZTF'],
                         help="Dataset to be used as target for the experiment.")
    
    
    # For the MME method
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimension of the latent space of the Autoencoder.")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Hyperparameter temperature.")
    parser.add_argument("--lambda_", type=float, default=0.1,
                        help="Hyperparameter lambda.")
    parser.add_argument("--n_shots", type=int, default=10,
                         help="Number of labeled samples to be used for the target.")
    parser.add_argument("--n_val", type=int, default=10,
                         help="Number of labeled samples to be used for validation.")
    parser.add_argument("--num_iterations", type=int, default=20000,
                         help="Number of of iteration.")
    
    args = parser.parse_args(args=[])

    return args
