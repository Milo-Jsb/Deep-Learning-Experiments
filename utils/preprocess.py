""" 
Script que genera el formato de los datos de entrada al modelo.
Se considera el preprocesamiento que se realizo en el respectivo paper de clasificación de supernovas.
Todos estan diseñados para preprocesar los datos provenientes del ZTF.
"""

import numpy as np
import pandas as pd

from itertools import groupby
import csv
import math

class CharnockPreprocess(object):
    def __init__(self, key_types, path_data, path_save, bands_name, normalize=False, 
                reproduce_charnock=False, online=False):
        self.flux_norm = 1.0
        self.time_norm = 1.0
        self.position_norm = 1.0
        self.grouping = 1

        self.bands_name = bands_name
        self.path_save = path_save

        self.key_types = key_types #{'SNIa': 0, 'nonSNIa': 1}
        self.lightcurves = pd.read_pickle(path_data)
        self.reproduce_charnock = reproduce_charnock # Utilizar dps

        # Preprocesamiento de la data
        if online:
            self.preprocess_online()
        else:
            self.preprocess_offline()

    def preprocess_online(self, sn_id, bands_name):
        pass

    def preprocess_offline(self):

        self.lightcurves = self.lightcurves[self.lightcurves.fid.isin(self.bands_name)]
        ids_sn_g_r = set(self.lightcurves.oid)

        print('Processing data...')

        fnohost = open(self.path_save + '/data_preprocessed.csv', 'w')
        wnohost = csv.writer(fnohost)

        fmask = open(self.path_save + '/mask_data.csv', 'w')
        wmask = csv.writer(fmask)

        sn_types = {}
        nb_sn = 0

        for id_sn in ids_sn_g_r:
            #print(f"Supernova: {id_sn} - it {nb_sn}")
            try:
                if nb_sn % int(len(ids_sn_g_r)/10) == 0:
                    print(f"{nb_sn} supernovae have been preprocessed")
            except ZeroDivisionError:
                print(f"{nb_sn} supernovae have been preprocessed")

            observaciones = self.lightcurves[self.lightcurves['oid'] == id_sn].sort_values(by=['mjd'])
                            
            first_obs = float(observaciones['mjd'].iloc[0])
            
            snid = observaciones['oid'].iloc[0]
            ra = observaciones['ra'].iloc[0]
            dec = observaciones['dec'].iloc[0]
            try:
                sn_type = observaciones['class'].iloc[0]
            except KeyError:
                sn_type = observaciones['alerceclass'].iloc[0]

            if self.reproduce_charnock:
                mwebv = observaciones['mwebv'].iloc[0]
                hostz = observaciones['photo_z'].iloc[0]
                sim_z = observaciones['sim_redshift'].iloc[0]
                            
            obs = []
            
            for idx_obs in range(len(observaciones)):
                # Mantienen el mismo largo de los flujos y errores
                bands = np.full(len(self.bands_name), -999, dtype=float)
                bands_error = np.full(len(self.bands_name), -999, dtype=float)
                
                # MJD
                mjd = float(observaciones['mjd'].iloc[idx_obs])
                
                # FLT
                # FLUXCAL AND FLUXCALERR
                for idx_band in range(len(bands)):
                    if observaciones['fid'].iloc[idx_obs] == self.bands_name[idx_band] and not np.isnan(observaciones['flux_tot_ujy'].iloc[idx_obs]):
                        bands[idx_band] = float(observaciones['flux_tot_ujy'].iloc[idx_obs])/self.flux_norm # No entiendo porque lo parte por un flux_norm = 1.0 ?
                        bands_error[idx_band] = float(observaciones['fluxunc_tot_ujy_resc'].iloc[idx_obs])/self.flux_norm              

                obs.append([(mjd - first_obs)/self.time_norm] + bands.tolist() + bands_error.tolist()) # Porque lo divide por un time_norm = 1.0 ?
            
            # Listas que contienen el tiempo, flujos y errores (mismo largo)
            t_arr = [obs[i][0] for i in range(len(obs))]
            bands_arr = []
            bands_err_arr = []

            for idx_band in range(len(self.bands_name)):
                bands_arr.append([obs[i][idx_band+1] for i in range(len(obs))])
                bands_err_arr.append([obs[i][idx_band+1+len(self.bands_name)] for i in range(len(obs))])

            # Preprocesamiento para una banda
            if len(self.bands_name) == 1:
                # Rellena todos los valores -999
                t = t_arr.copy()
                self.mask = np.ones(shape=(len(self.bands_name), len(t), 2))

                for idx_band in range(len(self.bands_name)):
                    bands_arr[idx_band], bands_err_arr[idx_band] = self.__fill_in_points(bands_arr[idx_band], 
                                                                                         bands_err_arr[idx_band])
            
                band_temp_arr = bands_arr.copy()
                band_err_temp_arr = bands_err_arr.copy()
                

            # Preprocesamiento para dos o mas bandas
            else:
                correctplacement = True
                frac = self.grouping # Porque grouping = 1 ? --> En el orden de un dia ?
                j = 0
                # Hasta que no haya SOLO UNA observación en un grupo de tiempos el frac sigue aumentando
                # ¿Por que quiere solo una?
                while correctplacement:
                    # t es tiempo promedio
                    # index son los indices de los tiempos que formaron el tiempo promedio
                    t, index, frac = self.__time_collector(t_arr, frac) 
                    self.mask = np.ones(shape=(len(self.bands_name), len(t), 2)) # enmascarara los flujos imputados
                    
                    band_aux_arr = []
                    band_err_aux_arr = []
                    tot = []

                    for idx_band in range(len(self.bands_name)):
                        band_aux_arr.append([])
                        band_err_aux_arr.append([])

                    for i in range(len(index)):
                        band_temp_arr = []
                        band_err_temp_arr = []
                        bandfail = []

                        for idx_band in range(len(self.bands_name)):
                            band_aux_arr[idx_band], band_err_aux_arr[idx_band], \
                                bandfail_aux = self.__create_colourband_array(index[i], 
                                                                              bands_arr[idx_band], 
                                                                              bands_err_arr[idx_band], 
                                                                              band_aux_arr[idx_band], 
                                                                              band_err_aux_arr[idx_band])

                            band_temp_arr.append(band_aux_arr[idx_band])
                            band_err_temp_arr.append(band_err_aux_arr[idx_band])
                            bandfail.append(bandfail_aux)
                                                           
                        tot.append(math.prod(bandfail))
            
                    # Hasta que todos tengo 1 observación puede ser -999 o una observación real
                    if all(tot):
                        correctplacement = False
                    else:
                        frac += 0.2
                
                # Rellena todos los valores -999
                for idx_band in range(len(self.bands_name)):
                    band_temp_arr[idx_band], band_err_temp_arr[idx_band] = self.__fill_in_points(band_temp_arr[idx_band], 
                                                                                                 band_err_temp_arr[idx_band], idx_band)

            obs = []
            for i in range(len(t)):
                aux = []
                aux.append(t[i])
                for idx_band in range(len(self.bands_name)):
                    aux.append(band_temp_arr[idx_band][i])
                    aux.append(band_err_temp_arr[idx_band][i])
                obs.append(aux)

            #if id_sn == 148636:
                #print(f"sn_type: {sn_type}")
                #print(obs)

            # Concatenación de datos
            try:
                if self.reproduce_charnock:
                    unblind = [sim_z, self.key_types[sn_type]]
                else:
                    unblind = [self.key_types[sn_type]]
            except:
                print('No information for', snid)

            for o in obs:
                if self.reproduce_charnock:
                    wnohost.writerow([snid, o[0], ra, dec, mwebv, hostz] + o[1:] + unblind) 
                else:
                    wnohost.writerow([snid, o[0], ra, dec] + o[1:] + unblind)
                
            self.mask = np.concatenate(self.mask, axis=1)
            for m in self.mask:
                wmask.writerow([snid] + m[0:].tolist())

            try:
                if self.reproduce_charnock:
                    sn_types[unblind[1]] += 1
                else:
                    sn_types[unblind[0]] += 1

            except:
                if self.reproduce_charnock:
                    sn_types[unblind[1]] = 1
                else:
                    sn_types[unblind[0]] = 1

            nb_sn += 1
            
        fnohost.close()
        fmask.close()
        
        print(f'Num train: {nb_sn}')
        print(f'SN types: {sn_types}\n')

    def __time_collector(self, arr, frac=1):
        '''
        Returns the an array of average times about clustered observation times. Default grouping is
        for times on the order of 1 day, although this is reduced if there are too many observations
        in that time. Also returns the index of the indices of the closest times in each flux band
        and the grouping fraction.
        * arr is an array containing all of the observation times
        * frac is the clustering scale where frac=1 is group times within a day
        * a is the array of grouped times
        - Used in parser_spline() for grouping flux errors to the nearest grouped time
        - Used in parser_augment() for grouping times from all observations
        '''
        ## Funciona como una tabla hash con un largo de 4 elementos por celdas, sino se va "agrandando la tabla"
        bestclustering = True
        while bestclustering:
            a = []
            for key, group in groupby(arr, key=lambda n: n//(1./frac)): # Extrae la parte entera
                s = sorted(group)
                a.append(np.sum(s)/len(s)) # Obtiene el tiempo promedio por dia, es decir, todos los 0,... , 1,... , 2,...
                                        # Aunque va variando un poco mientras aumenta el frac

            ind = []
            i = 0
            for key, group in groupby(arr, key=lambda n: n//(1./frac)):
                ind.append([])
                for j in group: # Indices de los tiempos que formaron el valor promedio guardado en a
                    ind[i].append(self.__index_min(abs(j-np.array(arr)))) # Obtiene los indices de "a" agrupados en listas de listas
                i += 1

            # Ajusta el tamaño de los grupos por dia, hasta que queden menor o igual que 4 [[0,1], [2,3,4], ..., [10]]
            if len([len(i) for i in ind if len(i)>4])!=0:
                frac += 0.1
            else:
                bestclustering = False
                
        return a, ind, frac

    def __index_min(self, values):
        '''
        Return the index of an array.
        * values is an array (intended to be times)
        - Used in time_collector() for grouping times
        - Used in parser_spline() for placing flux errors at the correct time in the time sequence
        '''
        return min(range(len(values)), key=values.__getitem__)
    
    def __create_colourband_array(self, ind, arr, err_arr, temp_arr, err_temp_arr): # Los ultimos dos parametros son listas vacias
        # Ingresa por grupos de tiempos formados en time collector
        '''
        Returns arrays containing the all of the flux observations, all of the flux error observations
        and an option to check that times are grouped such that there is only one observation in a
        cluster of times.
        * ind is the list of indices containing the nearest grouped time for each observation
        * arr is array of all of the flux observations at all observation times
        * err_arr is the array of all of the flux error observations at all observation times
        * temp_arr is the array containing the fluxes at grouped times
        * temp_err_arr is the array containing the flux errors at grouped times
        * out is a boolean which is True if there is only one observation per grouped time and False
        if there is more than one grouped time - the grouping factor is then reduced.
        - Used in parser_augment() to create the flux and flux error arrays at grouped times
        '''
        # Guarda el flujo y error correspondiente a ese grupo de tiempo
        temp = [arr[ind[i]] for i in range(len(ind)) if arr[ind[i]]!=-999]
        err_temp = [err_arr[ind[i]] for i in range(len(ind)) if err_arr[ind[i]]!=-999]
        
        # No tiene observaciones en ese grupo de tiempo.
        if len(temp)==0:
            temp_arr.append(-999)
            err_temp_arr.append(-999)
            out = True
            
        # Verifica que los tiempos estén agrupados de manera que solo haya una observación en un grupo de tiempos.
        elif len(temp)>1:
            out = False
            
        # Se agrega la observación de ese grupo de tiempo
        else:
            temp_arr.append(temp[0])
            err_temp_arr.append(err_temp[0])
            out = True
            
        return temp_arr, err_temp_arr, out

    def __fill_in_points(self, arr, err_arr, idx_band):
        '''
        Returns flux and flux error arrays where missing data is filled in with a random value between
        the previous and the next filled array elements. Missing intial or final data is filled in with
        the first or last non-missing data value respectively.
        * arr is the array of fluxes
        * err_arr is the array of flux errors
        - Used in parser_augment() to fill in missing data in flux and flux error arrays.
        '''

        # Devuelve los indices de valores distintos a -999
        ind = np.where(np.array(arr)!=-999)[0]
        length = len(arr)
        
        # Si todos los elementos son -999, es decir, no hay observaciones
        if len(ind)==0:
            arr = [-1 for i in range(length)]
            err_arr = [-1 for i in range(length)]
            self.mask[idx_band] = [[-1,-1] for i in range(length)]
            
        else:
            # Se rellenan los valores -999 con valores aleatorios entre la observación actual vs la siguiente
            for i in range(len(ind)-1):
                diff = ind[i+1]-ind[i]
                arr[ind[i]+1:ind[i+1]] = np.random.uniform(arr[ind[i]], arr[ind[i+1]], diff-1)
                err_arr[ind[i]+1:ind[i+1]] = np.random.uniform(err_arr[ind[i]], err_arr[ind[i+1]], diff-1)
                self.mask[idx_band][ind[i]+1:ind[i+1]] = [0,0] # Por el flujo y el error

            # Todos los valores anteriores al primer indice se rellenan con el primer valor (flujo) de esa observación
            for i in range(len(arr[:ind[0]])):
                arr[i] = arr[ind[0]]
                err_arr[i] = err_arr[ind[0]]
                self.mask[idx_band][i] = [0,0]
                
            # Todos los valores despues del utlimo indice se rellenan con el ultimo valor (flujo) de esa observación
            for i in range(len(arr[ind[-1]+1:])):
                arr[ind[-1]+1+i] = arr[ind[-1]]
                err_arr[ind[-1]+1+i] = err_arr[ind[-1]]
                self.mask[idx_band][ind[-1]+1+i] = [0,0]
                
        return arr, err_arr

def charnock_augmentation(num_augments, mask, data, col_data, bands_name, path_save='.'):      
    """
    Caso 1: el valor es la primera observación de esa banda
    Caso 2: el valor es la ultima observación de esa banda
    Caso 3: el valor se encuentra entre dos tiempos
    * Solo se consideran las observaciones reales para formar el intervalo de la distribución
    """
    # Obtiene el nombre de la columnas en donde hay bandas
    col_bands = get_bands_columns(bands_name, 'oid')

    # Read files
    ## if the data was preprocessed 
    if not isinstance(mask, pd.DataFrame):   
        data = pd.read_csv(data, header=None)
        mask = pd.read_csv(mask, header=None)

    data.columns = col_data
    mask.columns = col_bands

    # Starting aumentación
    print('Aumentation data...')

    id_sn = mask.oid.unique()
    filter_data = data[col_bands]

    it = 0
    new_data_aug = []
    for id in id_sn:

        if it % int(len(id_sn)/10) == 0:
            print(f"{it} supernovae have been preprocessed")

        np_mask = mask[mask.oid == id].values
        np_data = filter_data[filter_data.oid == id].values

        np_data_aug = []
        for augment in range(num_augments):
            np_data_aug.append(np_data.copy())
            if it == 0:
                new_data_aug.append([])

        idx_tuple = np.where(np_mask == 0)

        for row, col in zip(idx_tuple[0], idx_tuple[1]):
            pre_obs = pos_obs = row

            # encuentra la anterior observación real o bien llega al caso 1
            while np_mask[pre_obs, col] != 1 and pre_obs != 0:
                pre_obs -= 1
            
            # encuentra la posterior observación real o bien llega al caso 2          
            while np_mask[pos_obs, col] != 1 and pos_obs != len(np_data)-1:
                pos_obs += 1
            
            samples = np.random.uniform(np_data[pre_obs, col], 
                                        np_data[pos_obs, col], 
                                        num_augments)

            for augment in range(num_augments):
                np_data_aug[augment][row, col] = samples[augment]
                
        for augment in range(num_augments):
            new_data_aug[augment].append(np_data_aug[augment])
        it += 1

    i = 1 
    for new_data in new_data_aug:
        new_unblind = np.concatenate(new_data)
        data[col_bands[1:]] = new_unblind[0:,1:]
        data.to_csv(f'{path_save}/data_augmented_{i}.csv', index=False, header=False)
        i += 1

def drop_null_bands(path, value=-1):
    """"Elimina las sn que no tenian información en una de sus bandas"""
    file = pd.read_csv(path, header=None)

    file = file.replace(value, np.nan)
    file = file.dropna()

    return file

def get_bands_columns(bands_name, extra_features, separate=False):
    """
    separate = True, significa entregar todas las features separadas
    separate = False, todas juntas
    """
    col_bands = [extra_features]
    col_flux = []
    col_flux_err = []
    
    for band in bands_name:
        str_flux = band + '_flux'
        str_flux_err = band + '_error'
        
        # Separate
        col_flux.append(str_flux)
        col_flux_err.append(str_flux_err)

        # Junto
        col_bands.append(str_flux)
        col_bands.append(str_flux_err)
    
    if separate:
        return extra_features, col_flux, col_flux_err
    else:
        return col_bands

def change_flux_columns():
    
    pass
        
        
class RAPIDPreprocess():
    def __init__():
        pass


class DonosoPreprocess():
    def __init__():
        pass

class SuperNNovaPreprocess():
    def __init__():
        pass