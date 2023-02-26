"""
______________________________________________________________________________________________________________
Object Classes to generate dataloaders:
______________________________________________________________________________________________________________
- PhotometryDataset:        
    -> Objective:   Helps with the separation of the classes to generate a dataloader.
    -> Used:        RAPID model, ZTFDataloader, MME, PhotomeryDataLoader.
    -> Input:       self, data, labels, one_hot, nb_classes
    -> Return:      Class object

- PhotometryDataLoaders:  
    -> Objective:   
    -> Used:        
    -> Input:       
    -> Return:      

- ZTFDataLoaders:  
    -> Objective:    Create a Dataloader using ZTF type data.
    -> Used:         RAPID model.
    -> Input (init): X_train, y_train, X_val, y_val, X_test, y_test, one_hot, model ,batch_size=256,
                     num_workers=0, shuffle=False, collate_fn=None, normalize=False, n_quantiles=1000,
                     weight_norm=False   
    -> Return:       Class object.
    
- CharnockDataLoaders:
    -> Objective:            
    -> Used:     
    -> Input:    
    -> Return:
______________________________________________________________________________________________________________
"""

# Modulos ####################################################################################################
import torch

# Functions ##################################################################################################
import utils.data_handling

# Type of device #############################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Using {} device'.format(device))

##############################################################################################################
class PhotometryDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, one_hot, nb_classes):
        self.data = data

        if one_hot:
            self.labels = torch.nn.functional.one_hot(labels, nb_classes).float()  
        else:
            self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

##############################################################################################################
class PhotometryDataLoaders(torch.utils.data.DataLoader):
    """
    Dataloader que permite reproducir el metodo MME.
    """
    def __init__(self, source_data, target_data_lab, target_data_unl, target_data_val, test_data,
                 one_hot=False, batch_size=256, num_workers=0, shuffle=False, collate_fn=None, 
                 normalize=False, n_quantiles=1000, weight_norm=False):

        self.source_data      = utils.data_handling.input_model_rapid(source_data)
        self.target_data_lab  = utils.data_handling.input_model_rapid(target_data_lab)
        self.target_data_unl  = utils.data_handling.input_model_rapid(target_data_unl)
        self.target_data_val  = utils.data_handling.input_model_rapid(target_data_val)
        self.target_data_test = utils.data_handling.input_model_rapid(test_data)

        self.one_hot = one_hot

        self.__loader(batch_size, num_workers, shuffle, collate_fn, weight_norm)

    def __loader(self, batch_size, num_workers, shuffle, collate_fn, weight_norm):
        # Datasets
        self.nb_classes = self.source_data[1].unique().size(0)

        self.source_set       = PhotometryDataset(self.source_data[0], 
                                                self.source_data[1], 
                                                self.one_hot, 
                                                self.nb_classes)

        self.target_set_lab   = PhotometryDataset(self.target_data_lab[0], 
                                                self.target_data_lab[1], 
                                                self.one_hot, 
                                                self.nb_classes)
                                                
        self.target_set_unl   = PhotometryDataset(self.target_data_unl[0], 
                                                self.target_data_unl[1], 
                                                self.one_hot, 
                                                self.nb_classes)

        self.target_set_val   = PhotometryDataset(self.target_data_val[0], 
                                                self.target_data_val[1], 
                                                self.one_hot, 
                                                self.nb_classes)

        self.target_data_test = PhotometryDataset(self.target_data_test[0], 
                                                self.target_data_test[1], 
                                                self.one_hot, 
                                                self.nb_classes)

        # If is necesary to balance the weights of the clases
        sampler=None

        if weight_norm:
        
            weight_labels = self.__weights_for_balanced_classes(self.source_set)
            sampler       = torch.utils.data.sampler.WeightedRandomSampler(weights=weight_labels,
                                                                            num_samples=len(self.source_set),
                                                                            replacement=True)

        # Dataloaders (sin collate)
        self.source_set       = torch.utils.data.DataLoader(dataset=self.source_set, 
                                                        batch_size=batch_size, 
                                                        shuffle=shuffle, 
                                                        num_workers=num_workers, 
                                                        sampler=sampler)

        self.target_set_lab   = torch.utils.data.DataLoader(dataset=self.target_set_lab, 
                                                        batch_size=batch_size, 
                                                        shuffle=shuffle, 
                                                        num_workers=num_workers)

        self.target_set_unl   = torch.utils.data.DataLoader(dataset=self.target_set_unl, 
                                                        batch_size=batch_size, 
                                                        shuffle=shuffle, 
                                                        num_workers=num_workers)

        self.target_set_val   = torch.utils.data.DataLoader(dataset=self.target_set_val, 
                                                        batch_size=batch_size, 
                                                        shuffle=shuffle, 
                                                        num_workers=num_workers)

        self.target_data_test = torch.utils.data.DataLoader(dataset=self.target_data_test, 
                                                        batch_size=batch_size, 
                                                        shuffle=shuffle, 
                                                        num_workers=num_workers)


    def __weights_for_balanced_classes(self, train_set):
        weights = train_set.labels.bincount()
        weights = [1/i for i in weights]

        weight_labels = []
        for label in train_set.labels:
            for category in range(len(weights)):
                if label == category:
                    weight_labels.append(weights[category])
        
        return torch.Tensor(weight_labels)

##############################################################################################################
class ZTFDataLoaders(torch.utils.data.DataLoader):
    
    def __init__(self, X_train, y_train, 
                 X_val, y_val, 
                 X_test, y_test,
                 one_hot,
                 batch_size=256, 
                 num_workers=0, 
                 shuffle=False, 
                 collate_fn=False, 
                 normalize=False, 
                 n_quantiles=1000, 
                 weight_norm=False):

        # Atributes
        self.X_train = X_train
        self.X_val   = X_val
        self.X_test  = X_test
        
        # Labels
        self.y_train = y_train
        self.y_val   = y_val
        self.y_test  = y_test

        # One-hot
        self.one_hot = one_hot
        
        # Loader
        self.__loader(batch_size, num_workers, shuffle, collate_fn, weight_norm)

    def __loader(self, batch_size, num_workers, shuffle, collate_fn, weight_norm):
        
        # Datasets
        self.nb_classes = self.y_train.unique().size(0)

        train_set = PhotometryDataset(self.X_train, 
                                      self.y_train, 
                                      self.one_hot, 
                                      self.nb_classes)
        
        val_set   = PhotometryDataset(self.X_val, 
                                      self.y_val, 
                                      self.one_hot, 
                                      self.nb_classes)
        
        test_set  = PhotometryDataset(self.X_test, 
                                      self.y_test, 
                                      self.one_hot, 
                                      self.nb_classes)

        # If is necesary to balance the weights of the clases
        sampler = None

        if weight_norm:
        
            weight_labels = self.__weights_for_balanced_classes(train_set)
        
            sampler       = torch.utils.data.sampler.WeightedRandomSampler(weights     = weight_labels,
                                                                           num_samples = len(train_set),
        
                                                                           replacement = True)
        if collate_fn:
        
            # Dataloaders with collate
            self.train_set = torch.utils.data.DataLoader(dataset     = train_set, 
                                                         batch_size  = batch_size, 
                                                         shuffle     = shuffle, 
                                                         num_workers = num_workers, 
                                                         collate_fn  = collate_fn, 
                                                         sampler     = sampler)

            self.val_set   = torch.utils.data.DataLoader(dataset=val_set, 
                                                         batch_size=batch_size, 
                                                         shuffle=shuffle, 
                                                         num_workers=num_workers, 
                                                         collate_fn=collate_fn)

            self.test_set  = torch.utils.data.DataLoader(dataset     = test_set, 
                                                         batch_size  = batch_size, 
                                                         shuffle     = shuffle, 
                                                         num_workers = num_workers, 
                                                         collate_fn  = collate_fn)
        else:

            # Dataloaders without collate
            self.train_set = torch.utils.data.DataLoader(dataset     = train_set, 
                                                         batch_size  = batch_size, 
                                                         shuffle     = shuffle, 
                                                         num_workers = num_workers, 
                                                         sampler     = sampler)

            self.val_set   = torch.utils.data.DataLoader(dataset      = val_set, 
                                                         batch_size   = batch_size, 
                                                         shuffle      = shuffle, 
                                                         num_workers  = num_workers)

            self.test_set  = torch.utils.data.DataLoader(dataset     = test_set,
                                                         batch_size  = batch_size, 
                                                         shuffle     = shuffle, 
                                                         num_workers = num_workers)

    def __weights_for_balanced_classes(self, train_set):
        
        weights = train_set.labels.bincount()
        weights = [1/i for i in weights]

        weight_labels = []
        
        for label in train_set.labels:
            for category in range(len(weights)):
                if label == category:
                    weight_labels.append(weights[category])
        
        return torch.Tensor(weight_labels)

##############################################################################################################
class CharnockDataLoaders(torch.utils.data.DataLoader):
    """
    Permite reproducir los resultados del paper de Charnock utilizando solo los datos de 
    entrenamiento y testeo (ultimos como validación y test).
    """
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, one_hot, batch_size=256,
                num_workers=1, shuffle=False, collate_fn=None, normalize=False, n_quantiles=1000,
                weight_norm=False):

        # Atributos
        self.X_train = X_train
        self.X_val   = X_val
        self.X_test  = X_test
        
        # Etiquetas
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        # One-hot
        self.one_hot = one_hot

        # Loader
        self.__loader(batch_size, num_workers, shuffle, collate_fn, weight_norm)

    def __loader(self, batch_size, num_workers, shuffle, collate_fn, weight_norm):
        # Datasets
        self.nb_classes = self.y_train.unique().size(0)

        train_set = PhotometryDataset(self.X_train, 
                                    self.y_train, 
                                    self.one_hot, 
                                    self.nb_classes)
        
        val_set   = PhotometryDataset(self.X_val, 
                                    self.y_val, 
                                    self.one_hot, 
                                    self.nb_classes)        
        
        test_set  = PhotometryDataset(self.X_test, 
                                    self.y_test, 
                                    self.one_hot, 
                                    self.nb_classes)

        # If is necesary to balance the weights of the clases
        sampler=None
        if weight_norm:
            weight_labels = self.__weights_for_balanced_classes(train_set)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weight_labels,
                                                                    num_samples=len(train_set),
                                                                    replacement=True)

        # Dataloaders con collate
        self.train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=num_workers, collate_fn=collate_fn, sampler=sampler)
        
        self.val_set = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=num_workers, collate_fn=collate_fn)

        self.test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=num_workers, collate_fn=collate_fn)

    def __weights_for_balanced_classes(self, train_set):
        weights = train_set.labels.bincount()
        weights = [1/i for i in weights]

        weight_labels = []
        for label in train_set.labels:
            for category in range(len(weights)):
                if label == category:
                    weight_labels.append(weights[category])

        return torch.Tensor(weight_labels)
