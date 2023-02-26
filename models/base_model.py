# Modulos ####################################################################################################
import numpy as np
import time
import pickle
import torch

# Functions ##################################################################################################
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Utiliities #################################################################################################

# Checkpoint and early-stopping
import utils.checkpoint

# Clf-models
import models.sne_models

# MME-Charnock
import models.mme.classifier_charnock
import models.mme.fea_ext_charnock

# MME RAPID
import models.mme.classifier_rapid
import models.mme.fea_ext_rapid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

##############################################################################################################
class Model(object):
    
    def __init__(self, CONFIG):
        
        super(Model, self).__init__()
        
        # Load and save model
        self.name_model     = CONFIG.name_model
        self.param_path     = CONFIG.load_file + CONFIG.experiment + CONFIG.name_model
        self.type_train     = CONFIG.type_train

        self.epochs_trained = 1

        if CONFIG.load_model:
            
            model_path = self.param_path + CONFIG.model_path
            self.epochs_trained, CONFIG = self.load_checkpoint(self.param_path, model_path, CONFIG)
            
            print(f"The model was successfully loaded.\n")

        # Network
        self.rnn_type    = CONFIG.rnn_type
        self.input_size  = CONFIG.input_size
        self.hidden_size = CONFIG.hidden_size
        self.num_layers  = CONFIG.num_layers
        self.num_classes = CONFIG.num_classes
        self.dropout     = CONFIG.dropout
        self.activation  = CONFIG.activation
        self.batch_norm  = CONFIG.batch_norm

        # Early stopping
        self.patience    = CONFIG.patience
        self.metric_eval = CONFIG.metric_eval

        # MME
        self.temp           = CONFIG.temperature
        self.lambda_        = CONFIG.lambda_
        self.num_iterations = CONFIG.num_iterations
        
        # Create checpoint if new mode
        if CONFIG.load_model == False: 
            self.__choose_model()
            self.checkpoint = utils.checkpoint.Checkpoint(CONFIG)

    def summary(self):

        for model in self.model:
            print(f'{model}\n')

    def compile(self, loss_list, opt_list, lr_list, metric_eval, scheduler=None, save=True):

        # choose loss function
        self.loss = self.__choose_loss_function(loss_list) 
        
        # choose optimizer
        self.optimizer = self.__choose_optimizer(opt_list, lr_list) 
        
        # Scheduler
        if scheduler is not None:
            self.lr_scheduler = self.__choose_lr_scheduler(scheduler) 
        else:
            self.lr_scheduler = scheduler
        
        # Metrics
        self.metric_eval = metric_eval

        # Save init paremeters
        if save:
            self.checkpoint.save_init(loss_list, opt_list, scheduler, lr_list)

    def fit(self, train_dl=None, source_dl=None, target_lab_dl=None, target_unl_dl=None, val_dl=None, 
            train_type='mme', num_epochs=200, verbose=1):
        
        
        for epoch in range(self.epochs_trained, num_epochs+1):
            # Training epoch start time
            start_epoch = time.time()

            # Train and validation
            train_loss, train_acc   = self.__train_one_epoch(train_dl, source_dl, target_lab_dl,
                                                             target_unl_dl, epoch)
            val_loss  , val_acc     = self.__val_one_epoch(val_dl)
            
            # Training epoch last time
            end_epoch = time.time()

            # Mean loss and accuracy 
            epoch_train_loss, epoch_val_loss = np.mean(train_loss), np.mean(val_loss)
            epoch_train_acc , epoch_val_acc  = np.mean(train_acc),  np.mean(val_acc)

            # Information Summary
            if verbose:
                
                # For the mme method
                if train_type   == 'mme': epoch_name = 'Iteration'
                
                else: epoch_name = 'Epoch'
                
                print(f'\n{epoch_name} {epoch}/{num_epochs} '
                    + f'- time: {(end_epoch-start_epoch):.3f} seg\n'
                    + f'loss = {epoch_train_loss:.5f}, '
                    + f'val_loss = {epoch_val_loss:.5f}, '
                    + f'acc = {epoch_train_acc:.5f}, '
                    + f'val_acc = {epoch_val_acc:.5f}, ') 

            # Save mean loss and accuracy
            self.checkpoint.train_loss.append(epoch_train_loss), self.checkpoint.val_loss.append(epoch_val_loss)
            self.checkpoint.train_acc.append(epoch_train_acc)  , self.checkpoint.val_acc.append(epoch_val_acc)

            # If accuracy improves save the best model
            self.checkpoint.early_stopping(epoch, self.model, self.optimizer, self.lr_scheduler)

            # Checkpoin
            if epoch % 1 == 0:
                self.checkpoint.save_training(epoch, self.model, self.optimizer, self.lr_scheduler)

            # Early Stopping
            if self.checkpoint.early_stop:

                print("Best parameters restored")
                self.__load_weights()
                break
            
            # Scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Load weights
        self.__load_weights()
        print("Best parameters restored.")
    
    def evaluate(self, test_dataloader, batch_size):

        loss_list, acc_list = [], []
        self.model[0].eval() 

        with torch.no_grad():
            
            if (self.name_model == "charnock"):
                    
                for batch_idx, (data, targets, _) in enumerate(test_dataloader):

                    # Get data to cuda if possible
                    X_test_sorted = data.to(device)
                    y_test_sorted = torch.stack(targets).to(device)

                    batch_size = test_dataloader.batch_size

                    scores_sorted = self.model[0](X_test_sorted, batch_size) 
                    loss = self.loss[0](scores_sorted, y_test_sorted) 
                    
                    loss_list.append(loss)

                    # metrics 
                    acc = self.__get_metrics_charnock(y_test_sorted, scores_sorted)
                    acc_list.append(acc)

            elif (self.name_model == "rapid"):
                    
                for batch_idx, (data, targets) in enumerate(test_dataloader):
                    # Get data to cuda if possible
                    X_test_sorted = data.float().to(device)
                    y_test_sorted = targets.to(device)
                    
                    if (self.type_train == "mme"):
                        
                        self.model[0].eval()                                           
                        self.model[1].eval()  

                        x_out         = self.model[0](X_test_sorted)
                        scores_sorted = self.model[1](x_out)

                        loss          = self.loss[0](scores_sorted, y_test_sorted)                      
                    
                    else: 
                        scores_sorted = self.model[0](X_test_sorted)
                        loss          = self.loss[0](scores_sorted, y_test_sorted) 

                    loss_list.append(loss)

                    # metrics 
                    acc = self.__get_metrics_rapid(y_test_sorted, scores_sorted)
                    acc_list.append(acc)


        loss = torch.mean(torch.stack(loss_list)).item()
        acc = np.mean(acc_list)

        return loss, acc

    def predict(self, test_dataloader, batch_size):

        y_pred_prob, y_pred = [], []
        self.model[0].eval() 

        if (self.name_model == "charnock"):

            with torch.no_grad():
                for batch_idx, (data, targets, idx_sorted) in enumerate(test_dataloader):
                    # Get data to cuda if possible
                    X_test_sorted = data.to(device)

                    #batch_size = X_test.batch_sizes[0].item()
                    batch_size = test_dataloader.batch_size

                    # forward/predict
                    scores_sorted = self.model[0](X_test_sorted, batch_size)

                    # Vuelve las predicciones y etiquetas al orden normal
                    # sirve para evaluar las metricas
                    scores = torch.zeros(size=scores_sorted.size())
                    for i in range(len(idx_sorted)):
                        scores[idx_sorted[i]] = scores_sorted[i]

                    y_pred_prob_batch = torch.nn.functional.softmax(scores, dim=1).to(device)
                    y_pred_prob.append(y_pred_prob_batch)

                    y_batch_pred = torch.argmax(y_pred_prob_batch, axis=1)
                    y_pred.append(y_batch_pred)

            y_pred_prob = torch.cat((y_pred_prob),dim=0).cpu().numpy()
            y_pred = torch.cat((y_pred),dim=0).cpu().numpy()

            return y_pred_prob, y_pred
        
        if (self.name_model == "rapid"):

            with torch.no_grad():

                for batch_idx, (data, targets) in enumerate(test_dataloader):
                    # Get data to cuda if possible
                    X_test_sorted = data.float().to(device)

                    # forward/predict
                    if (self.type_train == "mme"):
                        
                        self.model[0].eval()
                        self.model[1].eval()

                        x_out         = self.model[0](X_test_sorted)
                        scores_sorted = self.model[1](x_out)

                    else:    
                        scores_sorted = self.model[0](X_test_sorted)

                    
                    y_pred_prob_batch = torch.nn.functional.softmax(scores_sorted, dim=1).to(device)
                    y_pred_prob.append(y_pred_prob_batch)

                    y_batch_pred = torch.argmax(y_pred_prob_batch, axis=1)
                    y_pred.append(y_batch_pred)

            y_pred_prob  = torch.cat((y_pred_prob),dim=0).cpu().numpy()
            y_pred       = torch.cat((y_pred),dim=0).cpu().numpy()
                    
            return y_pred_prob, y_pred

    def load_checkpoint(self, param_path, model_path, CONFIG):

        with open(param_path + '/parameters.model', 'rb') as f:
            dictionary_model = pickle.load(f)

        CONFIG.type_train              = self.type_train
        dictionary_model['type_train'] = self.type_train
        
        # Network
        CONFIG.rnn_type    = self.rnn_type    = dictionary_model['rnn_type']
        CONFIG.input_size  = self.input_size  = dictionary_model['input_size']
        CONFIG.hidden_size = self.hidden_size = dictionary_model['hidden_size']
        CONFIG.num_layers  = self.num_layers  = dictionary_model['num_layers']
        CONFIG.num_classes = self.num_classes = dictionary_model['num_classes']
        CONFIG.dropout     = self.dropout     = dictionary_model['dropout']
        CONFIG.activation  = self.activation  = dictionary_model['activation']
        CONFIG.batch_norm  = self.batch_norm  = dictionary_model['batch_norm']

        # Compile
        CONFIG.loss_list = dictionary_model['loss_list']
        CONFIG.optimizer = dictionary_model['optimizer']
        CONFIG.lr        = dictionary_model['lr']
        
        # Early stopping
        CONFIG.patience    = dictionary_model['patience']
        CONFIG.metric_eval = dictionary_model['metric_eval']
    
        # Model load
        checkpoint     = torch.load(model_path)
        epochs_trained = checkpoint['epoch']
        
        if (CONFIG.type_train == "ft"):
            epochs_trained = 1
        
        if (CONFIG.type_train == "mme"):
            self.temp = CONFIG.temperature
        
        self.__choose_model() 
        for i in range(len(self.model)):
            self.model[i].load_state_dict(checkpoint[f'model_state_{i}'])
        
        self.compile(CONFIG.loss_list, CONFIG.optimizer, CONFIG.lr, CONFIG.metric_eval, save=False)
        for i in range(len(self.model)):
            self.optimizer[i].load_state_dict(checkpoint[f'optim_state_{i}'])

        self.lr_scheduler = None 

        # To checkpoint
        self.checkpoint            = utils.checkpoint.Checkpoint(CONFIG)
        self.checkpoint.train_loss = checkpoint['train_loss']
        self.checkpoint.val_loss   = checkpoint['val_loss']
        self.checkpoint.train_acc  = checkpoint['train_acc']
        self.checkpoint.val_acc    = checkpoint['val_acc']

        # early stopping values
        self.checkpoint.val_acc_min      = max(self.checkpoint.val_acc)
        self.checkpoint.best_acc         = max(self.checkpoint.val_acc)
        self.checkpoint.val_loss_min     = min(self.checkpoint.val_loss)
        self.checkpoint.min_loss         = min(self.checkpoint.val_loss)
        self.checkpoint.counter          = checkpoint['counter']
        self.checkpoint.checkpoint_early = checkpoint['checkpoint_early']

        if (CONFIG.type_train == "ft"):
            self.checkpoint.counter          = 0
            self.checkpoint.checkpoint_early =  None

        return epochs_trained, CONFIG

    def __load_weights(self):
        """Load the best trained parameters of the model"""
        state_dict = torch.load(self.checkpoint.checkpoint_early)
        for i in range(len(self.model)):
            self.model[i].load_state_dict(state_dict[f'model_state_{i}'])

    def __train_one_epoch(self, train_dl, source_dl, target_lab_dl, target_unl_d, epoch):
        """
        Se escoge el tipo de entrenamiento:
        normal: aprendizaje supervisado
        kd: aprendizaje semi supervisado (knowledge distilation)
        mme: aprendizaje semi supervisado con domain adaptation (minimax entropy)
        """
        if self.type_train == 'normal':
            return self.__train_normal(train_dl)
        elif self.type_train == 'ft':
            return self.__train_normal(train_dl)
        elif self.type_train == 'mme':
            return self.__train_mme(source_dl, target_lab_dl, target_unl_d, epoch)

    def __val_one_epoch(self, val_dataloader):
        """
        Se escoge el tipo de validación:
        normal: aprendizaje supervisado
        kd: aprendizaje semi supervisado (knowledge distilation)
        mme: aprendizaje semi supervisado con domain adaptation (minimax entropy)
        """
        if self.type_train == 'normal':
            return self.__validation_normal(val_dataloader)
        elif self.type_train == 'ft':
            return self.__validation_normal(val_dataloader)
        elif self.type_train == 'mme':
            return self.__validation_mme(val_dataloader)

    def __train_normal(self, train_dataloader):
        
        self.model[0].train()
        train_loss, train_acc = [], []
        y_pred = []

        if (self.name_model == "charnock"):

            for batch_idx, (data, targets, idx_sorted) in enumerate(train_dataloader):
                
                # Get data in cuda if it was possible
                X_train_sorted = data.to(device)
                y_train_sorted = torch.stack(targets).to(device)

                batch_size = train_dataloader.batch_size

                # forward
                scores_sorted  = self.model[0](X_train_sorted, batch_size)
                loss           = self.loss[0](scores_sorted, y_train_sorted)
                
                scores         = torch.zeros(size=scores_sorted.size())
                y_train        = torch.zeros(size=y_train_sorted.size()) 
                
                for i in range(len(idx_sorted)):
                    scores[idx_sorted[i]]  = scores_sorted[i]
                    y_train[idx_sorted[i]] = y_train_sorted[i]           

                # backward
                self.optimizer[0].zero_grad() # Quizas tenga que poner [0]
                loss.backward()

                # gradient descent or adam step
                self.optimizer[0].step()

                # metrics 
                acc = self.__get_metrics_charnock(y_train, scores)

                # save metrics for step
                train_loss.append(loss.item()) # standard Python number
                train_acc.append(acc)
            
            return train_loss, train_acc

        elif (self.name_model == "rapid"):

            for batch_idx, (data, targets) in enumerate(train_dataloader):
                
                # Get data in cuda if it was possible
                X_train_sorted = data.float().to(device)

                y_train_sorted = targets.to(device)

                # forward
                scores_sorted  = self.model[0](X_train_sorted)

                loss  = self.loss[0](scores_sorted, y_train_sorted)          

                # backward
                self.optimizer[0].zero_grad() # Quizas tenga que poner [0]
                loss.backward()

                # gradient descent or adam step
                self.optimizer[0].step()

                # metrics 
                acc = self.__get_metrics_rapid(y_train_sorted, scores_sorted)

                # save metrics for step
                train_loss.append(loss.item()) # standard Python number
                train_acc.append(acc)
            
            return train_loss, train_acc


    def __validation_normal(self, val_dataloader):
        """Solo para aprendizaje supervisado con UN MODELO"""
        val_loss, val_acc = [], []
        self.model[0].eval() # No activate dropout and batchNorm

        with torch.no_grad():

            if (self.name_model == "charnock"):

                for batch_idx, (data, targets, idx_sorted) in enumerate(val_dataloader):
                    # Get data to cuda if possible
                    X_val_sorted = data.to(device)
                    y_val_sorted = torch.stack(targets).to(device)

                    batch_size = val_dataloader.batch_size

                    # forward
                    scores_sorted = self.model[0](X_val_sorted, batch_size)
                    loss = self.loss[0](scores_sorted, y_val_sorted)
                    
                    # Vuelve las predicciones y etiquetas al orden normal
                    # sirve para evaluar las metricas
                    scores = torch.zeros(size=scores_sorted.size())
                    y_val = torch.zeros(size=y_val_sorted.size()) 
                    for i in range(len(idx_sorted)):
                        scores[idx_sorted[i]] = scores_sorted[i]
                        y_val[idx_sorted[i]]  = y_val_sorted[i]
                
                    # metrics 
                    acc = self.__get_metrics_charnock(y_val, scores)

                    # save metrics for step
                    val_loss.append(loss.item()) # standard Python number
                    val_acc.append(acc)

                return val_loss, val_acc

            if (self.name_model == "rapid"):

                for batch_idx, (data, targets) in enumerate(val_dataloader):
                    # Get data to cuda if possible
                    X_val_sorted = data.float().to(device)
                    y_val_sorted = targets.to(device)

                    # forward
                    scores_sorted = self.model[0](X_val_sorted)
                    loss          = self.loss[0](scores_sorted, y_val_sorted)
                
                    # metrics 
                    acc = self.__get_metrics_rapid(y_val_sorted, scores_sorted)

                    # save metrics for step
                    val_loss.append(loss.item()) # standard Python number
                    val_acc.append(acc)

                return val_loss, val_acc


    def __train_mme(self, source_dl, target_lab_dl, target_unl_dl, iteration):

        # Creamos listas para guardar los resultados
        train_loss, train_acc = [], []
        
        # Separamos las etapas del modelo
        self.model[0].train()                                           # feature extractor
        self.model[1].train()                                           # clasificador

        # Recorrido por batch en una cierta cantidad de iteraciones
        iteration = iteration - 1                                       # porque la epoch empieza en 1

        # Iteramos dentro de cada batch
        for iteration in range(self.num_iterations):

            if iteration % len(source_dl) == 0:
                self.iter_source = iter(source_dl)
            
            if iteration % len(target_lab_dl) == 0:
                self.iter_lab_target = iter(target_lab_dl)
            
            if iteration % len(target_unl_dl) == 0:
                self.iter_unl_target = iter(target_unl_dl) 

            
            x_s    , y_s      = next(self.iter_source)
            x_lab_t, y_lab_t, = next(self.iter_lab_target)
            x_unl_t, _        = next(self.iter_unl_target)

            x_s     = x_s.float().to(device)
            y_s     = y_s.to(device)
            
            x_lab_t = x_lab_t.float().to(device)
            y_lab_t = y_lab_t.to(device)
            
            x_unl_t = x_unl_t.float().to(device)

            # CrossEntropy Minimization (orden check)
            self.__zero_grad_mme() 

            x_out_s = self.model[0](x_s)
            x_out_t = self.model[0](x_lab_t)

            scores_s = self.model[1](x_out_s, grad_rev = False)
            scores_t = self.model[1](x_out_t, grad_rev = False)

            l_xent_1 = self.loss[0](scores_s, y_s)     # cross entropy
            l_xent_2 = self.loss[0](scores_t, y_lab_t) # cross entropy
            
            l_xent = l_xent_1 + l_xent_2


            l_xent.backward(retain_graph=True)

            self.__step_mme()
             
            
            ## For the metrics
            scores = torch.cat((scores_s, scores_t))
            y      = torch.cat((y_s, y_lab_t))

            # Minimax Loss
            x_out_u  = self.model[0](x_unl_t)
            scores_u = self.model[1](x_out_u, grad_rev=True)
            
            l_t = self.__adentropy(scores_u)

            l_t.backward()
            self.__step_mme()

            self.__zero_grad_mme()
            
            # metrics
            acc = self.__get_metrics_rapid(y, scores) 

            # save metrics for step
            train_loss.append(l_xent.item()) # standard Python number
            train_acc.append(acc)

        return train_loss, train_acc


    def __validation_mme(self, val_dataloader):
        """Solo para aprendizaje supervisado con UN MODELO, esto se cambio para aceptar SOLO A RAPID"""
        
        val_loss, val_acc = [], []
        
        self.model[0].eval() 
        self.model[1].eval()

        with torch.no_grad():
            
            for batch_idx, (data, targets) in enumerate(val_dataloader):
                # Get data to cuda if possible
                X_val = data.float().to(device)
                y_val = targets.to(device)

                batch_size = val_dataloader.batch_size

                # forward
                x_out  = self.model[0](X_val)
                
                scores = self.model[1](x_out)

                l_xent = self.loss[0](scores, y_val) # cross entropy
                
                # metrics 
                acc = self.__get_metrics_rapid(y_val, scores)

                # save metrics for step
                val_loss.append(l_xent.item()) # standard Python number
                val_acc.append(acc)

        return val_loss, val_acc


    def __choose_model(self):
        """Selecciona el modelo de clasificación de supernovas"""
        if self.type_train == 'normal':
            if self.name_model.lower() == 'charnock':
                model = models.sne_models.CharnockModel(self.rnn_type, self.input_size, 
                                                             self.hidden_size, self.num_layers,
                                                             self.num_classes, self.dropout, 
                                                             self.activation, self.batch_norm).to(device)
            elif self.name_model.lower() == 'rapid':
                model = models.sne_models.RAPIDModel(self.rnn_type, self.input_size, 
                                                    self.hidden_size, self.num_layers, 
                                                    self.num_classes, self.dropout, 
                                                    self.activation, self.batch_norm).to(device)
            self.model = [model]
        
        elif self.type_train == "ft":
            if self.name_model.lower() == 'charnock':
                model = models.sne_models.CharnockModel(self.rnn_type, self.input_size, 
                                                             self.hidden_size, self.num_layers,
                                                             self.num_classes, self.dropout, 
                                                             self.activation, self.batch_norm).to(device)
            elif self.name_model.lower() == 'rapid':
                model = models.sne_models.RAPIDModel(self.rnn_type, self.input_size, 
                                                    self.hidden_size, self.num_layers, 
                                                    self.num_classes, self.dropout, 
                                                    self.activation, self.batch_norm).to(device)
            self.model = [model]

        elif self.type_train == 'mme':
            if self.name_model.lower() == 'charnock':
                self.ftr_ext        = models.mme.fea_ext_charnock.FeatureExtractor(self.rnn_type, self.input_size, 
                                                                                self.hidden_size, self.num_layers,
                                                                                self.dropout).to(device)

                self.clf            = models.mme.classifier_charnock.Classifier(self.hidden_size, 
                                                                                self.num_classes, 
                                                                                self.temp).to(device)
            elif self.name_model.lower() == 'rapid':

                self.ftr_ext        = models.mme.fea_ext_rapid.FeatureExtractor(self.rnn_type, self.input_size, 
                                                                                self.hidden_size, self.num_layers,
                                                                                self.dropout).to(device)

                self.clf            = models.mme.classifier_rapid.Classifier(self.hidden_size, 
                                                                                self.num_classes, 
                                                                                self.temp).to(device)

            self.model = [self.ftr_ext, self.clf]

 
    def __choose_loss_function(self, loss_list):
        criterion = []
        for loss in loss_list:
            if loss.lower() == 'crossentropy':
                criterion.append(torch.nn.CrossEntropyLoss().to(device))
            elif loss.lower() == 'kldiv':
                criterion.append(torch.nn.KLDivLoss().to(device))
            elif loss.lower() == 'cce':
                # it need (torch.nn.functional.log_softmax(y_pred), y_true)
                criterion.append(torch.nn.NLLLoss().to(device))

        return criterion


    def __choose_optimizer(self, optim_list, lr_list):
        optimizer = []
        for model, optim, lr in zip(self.model, optim_list, lr_list):
            if optim.lower() == 'adam':
                optimizer.append(torch.optim.Adam(model.parameters(), lr=lr))
            elif optim.lower() == 'rms':
                optimizer.append(torch.optim.RMSprop(model.parameters(), lr=lr))

        return optimizer


    def __choose_lr_scheduler(self, scheduler):
        if scheduler.lower() == 'cosineannealing': # ARREGLAR
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 50, 1e-6)

        return lr_scheduler


    def __adentropy(self, out):
        """Adentropy loss for the MME minimax optimization."""
        out = torch.nn.functional.softmax(out, dim=1)
        adentropy = self.lambda_ * torch.mean(torch.sum(out * torch.log(out + 1e-10), dim=1))
        return adentropy


    def __zero_grad_mme(self):
        self.optimizer[0].zero_grad() # feature optimizer
        self.optimizer[1].zero_grad() # classifier optimizer


    def __step_mme(self):
        self.optimizer[0].step() # feature optimizer
        self.optimizer[1].step() # classifier optimizer


    def __get_metrics_charnock(self, y_true, scores):
        y_pred = torch.argmax(scores, axis=1)

        # Try if y_true is list or one-hot 
        try:
            acc = (y_true == y_pred).sum().item() / len(y_true)
        except RuntimeError:
            y_true = torch.argmax(y_true, dim=1)
            acc = (y_true == y_pred).sum().item() / len(y_true)       

        if self.metric_eval.lower() == 'f1_score':
            #f1_score_custom = utils.F1Score('macro')(y_val_pred, y_val)
            f1_score = f1_score(y_true=y_true.tolist(), y_pred=y_pred.tolist(), 
                                average='macro', pos_label=None)   

            return acc, f1_score     

        return acc
    
    def __get_metrics_rapid(self, y_true, scores):
        y_pred = torch.argmax(scores, axis=1)

        # Try if y_true is list or one-hot 
        try:
            acc = (y_true == y_pred).sum().item() /  np.prod(y_true.shape) 
        except RuntimeError:
            y_true = torch.argmax(y_true, dim=1)
            acc = (y_true == y_pred).sum().item() / np.prod(y_true.shape)        

        if self.metric_eval.lower() == 'f1_score':
            #f1_score_custom = utils.F1Score('macro')(y_val_pred, y_val)
            f1_score = f1_score(y_true=y_true.tolist(), y_pred=y_pred.tolist(), 
                                average='macro', pos_label=None)   

            return acc, f1_score     

        return acc
        

def model_type(CONFIG):
    
    if CONFIG.load_model:
        model = Model(CONFIG)
        model.summary()

    else:
        model = Model(CONFIG)
        model.summary()
        model.compile(CONFIG.loss_list, CONFIG.optimizer, CONFIG.lr, CONFIG.metric_eval)

    return model
