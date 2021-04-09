####################################################################################################

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import unicodedata
import numpy as np
from matplotlib import pylab, rc 
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.utils import plot_model, to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Input, Concatenate, Dropout, Flatten, Embedding, BatchNormalization, RepeatVector, Reshape
from keras.models import Model
from keras import backend as K
K.clear_session()
import random
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import HTML, display
import time
import re
from itertools import combinations
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
pylab.style.use('ggplot')
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import matplotlib.patches as mpatches
from collections import defaultdict

####################################################################################################

#reproducible results
def reproducible_results(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  tf.random.set_seed(seed)
  sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
  tf.compat.v1.keras.backend.set_session(sess)

####################################################################################################

#normalize coomune names
def nornalize(text):
  text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
  text = text.upper()
  text = str(text, 'utf-8')
  return text
#read data and fix commune names
def read_data(path, pos, return_communes = True):
  data = pd.read_csv(path)
  if 'Region' in data.columns:
    data['Region'] = data['Region'].apply(lambda x: nornalize(x))
    data['Region'] = data['Region'].replace('LA ARAUCANIA', 'ARAUCANIA')
    data['Region'] = data['Region'].replace('MAGALLANES Y LA ANTARTICA', 'MAGALLANES')
    data['Region'] = data['Region'].replace('DEL LIBERTADOR GENERAL BERNARDO OHIGGINS', 'OHIGGINS')
  if return_communes:
    data['Comuna'] = data['Comuna'].apply(lambda x: nornalize(x))
    data['Comuna'] = data['Comuna'].replace('COIHAIQUE', 'COYHAIQUE') 
  headers = data.columns
  date = headers[pos:]
  if return_communes:
    communes = data['Comuna'].to_numpy()
    return data, date.to_numpy(), communes
  else:
    return data, date.to_numpy()
#fix current date
def adjust_date(date, dates):
  date_aux = date
  minus = 1
  while date_aux not in dates:
    date_aux = date
    date_aux = str( ( datetime.strptime(date_aux, "%Y-%m-%d")-timedelta(minus) ).date() )
    minus+=1
  return date_aux
#get previous dates
def get_dates(previous, date, dates):
  dates_aux = [date]
  minus = 1
  while len(dates_aux)<previous+1:
    actual = datetime.strptime(date, "%Y-%m-%d")
    before = datetime.strftime(actual-timedelta(minus), '%Y-%m-%d')
    if before in dates:
      dates_aux.append(before)
    minus+=1
  dates_aux.reverse()
  return np.array(dates_aux)
#get uci-bed data
def get_uci_bed_covid_region_data(data, regiones_covid):
    uci_bed_covid_region, dates_uci_bed_covid_region = data
    uci_bed_covid_region_aux = pd.DataFrame( columns=['Region']+list(dates_uci_bed_covid_region) )
    uci_bed_covid_region_aux['Region'] = regiones_covid
    for idx in range(len(regiones_covid)):
      region_aux = regiones_covid[idx]
      df_aux = uci_bed_covid_region[uci_bed_covid_region['Region'] == region_aux]
      total_uci_covid = df_aux[df_aux['Serie']=='Camas UCI ocupadas COVID-19'][dates_uci_bed_covid_region].to_numpy().flatten()
      total_uci_hab = df_aux[df_aux['Serie']=='Camas UCI habilitadas'][dates_uci_bed_covid_region].to_numpy().flatten()
      usage = total_uci_covid/total_uci_hab
      uci_bed_covid_region_aux.loc[idx, dates_uci_bed_covid_region] = usage
    uci_bed_covid_region = uci_bed_covid_region_aux.copy()
    del uci_bed_covid_region_aux
    return uci_bed_covid_region, dates_uci_bed_covid_region
#get ventilator data
def get_vent_covid_region_data(data):
    vent_covid_chile, dates_vent_covid_chile = data
    df_aux = pd.DataFrame(columns=['Valor']+list(dates_vent_covid_chile))
    df_aux.loc[0, 'Valor'] = 'Ventiladores'
    df_aux.loc[0, dates_vent_covid_chile] = vent_covid_chile[vent_covid_chile['Ventiladores']=='ocupados'][dates_vent_covid_chile].to_numpy()[0]/vent_covid_chile[vent_covid_chile['Ventiladores']=='total'][dates_vent_covid_chile].to_numpy()[0]
    vent_covid_chile = df_aux.copy()
    del df_aux
    return vent_covid_chile, dates_vent_covid_chile

####################################################################################################

#get data from dataframes
def get_data(search, field, key, data, dates_group):
  dates_key = list(dates_group[key].keys())
  data_aux = data[data[field]==search]
  new_data = []
  for key_date in dates_key:
    new_data.append( data_aux[dates_group[key][key_date]].to_numpy()[0] )
  new_data = list(new_data)
  return new_data
#get national data (multiple columns available for a same kind of feature)
def get_data_chile(data_chile, key_group, dates_group, group):
  key_data = data_chile.columns[0]
  keys_data_chile = data_chile[key_data].to_numpy()
  data_aux_chile = {}
  for aux_key in keys_data_chile:
     data_aux_chile[group+'_'+aux_key+'_chile'] = get_data(aux_key, key_data, key_group, data_chile, dates_group)
  return data_aux_chile
#encode incidence values
def incidence2code(incidence_values):
  labels = []
  for val in incidence_values:
    if val<25: label = 0
    else: label = 1
    labels.append(label)
  return np.array(labels)
#fix incidence missing values
def fix_incidence(incidence):
  for index in range( 1, len(incidence) ):
    if incidence[index]<incidence[index-1]:
      #keep value
      incidence[index] = incidence[index-1]
  return incidence
#plot communal features 
def plot_dynamic_data(dataframes, commune, idx, dates):
  fontsize = 20
  rc('xtick', labelsize=fontsize) 
  rc('ytick', labelsize=fontsize)
  incidence = dataframes['dynamic_commune_'+commune]['incidence'].to_numpy()
  incidence_cum = dataframes['dynamic_commune_'+commune]['cummulative incidence'].to_numpy()
  mov_ext = np.array( dataframes['dynamic_commune_'+commune]['A_mov_ext_commune'].tolist() )[:,-1]
  mov_int = np.array(dataframes['dynamic_commune_'+commune]['A_mov_int_commune'].tolist() )[:,-1]
  deaths = np.array( dataframes['dynamic_commune_'+commune]['B_deaths_covid_commune'].tolist() )[:,-1]
  fig = pylab.figure(idx, figsize=(30,15))
  x = list(range(len(incidence)))
  selected_dates = ['']*len(x)
  selected_dates[0] = dates[0]
  selected_dates[-1] = dates[-1]
  pylab.suptitle(commune, fontsize = fontsize)
  pylab.subplot(3,1,1)  
  incidence_aux = [val if val<25 else 0 for val in incidence]
  pylab.bar(x, incidence_aux, color = 'royalblue', label = 'Negative class \n IN$<25$')
  incidence_aux = [val if val>=25 else 0 for val in incidence]
  pylab.bar(x, incidence_aux, color = 'indianred', label = 'Positive class \n IN$\geq 25$')
  pylab.ylabel('Incidence (IN)', fontsize = fontsize)
  pylab.legend(loc = 'upper left', ncol = 1, fontsize = fontsize)
  pylab.grid(True)
  pylab.xticks(x, selected_dates, rotation =  'horizontal')
  pylab.subplot(3,1,2)
  pylab.plot(x, mov_ext, 'royalblue', label = 'External')
  pylab.plot(x, mov_int, 'indianred', label = 'Internal')
  pylab.ylabel('Mobility index', fontsize = fontsize)
  pylab.legend(loc = 'upper left', ncol = 3, fontsize = fontsize)
  pylab.grid(True)
  pylab.xticks(x, selected_dates, rotation =  'horizontal')
  pylab.subplot(3,1,3)
  pylab.bar(x, deaths, color = 'royalblue')
  pylab.ylabel('# deaths', fontsize = fontsize)
  pylab.grid(True)
  pylab.xticks(x, selected_dates, rotation =  'horizontal')
  pylab.show()
   ####################################################################################################
    
def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

def LSTM_model(param_grid, n_clases, n_steps, n_features_d_commune, n_features_d_region, n_features_d_chile, region_opt, chile_opt, biLSTM, stacked):
  initializer = tf.keras.initializers.GlorotUniform()
  #dynamic features commune
  input_d_commune = Input( shape=(n_steps, n_features_d_commune) )
  if not biLSTM:
    if stacked:
      x = LSTM(param_grid['units_lstm'], return_sequences=True, activation=param_grid['activation'])(input_d_commune)   
      x = LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation'])(x)
    else:
      x = LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation'])(input_d_commune)
  else:
    x = Bidirectional( LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation']) )( input_d_commune ) 
  out_d_commune = Dropout(param_grid['dropout_rate'])(x)  
  #dynamic features region
  if region_opt:
    input_d_region = Input( shape=(n_steps, n_features_d_region) )
    if not biLSTM:
      if stacked:
        x = LSTM(param_grid['units_lstm'], return_sequences=True, activation=param_grid['activation'])(input_d_region)
        x = LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation'])(x)
      else:
        x = LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation'])(input_d_region)
    else:
      x = Bidirectional( LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation']) )( input_d_region ) 
    out_d_region = Dropout(param_grid['dropout_rate'])(x)
  #dynamic features chile
  if chile_opt:
    input_d_chile = Input( shape=(n_steps, n_features_d_chile) )
    if not biLSTM:
      if stacked:
        x = LSTM(param_grid['units_lstm'], return_sequences=True, activation=param_grid['activation'])(input_d_chile)
        x = LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation'])(x)
      else:
        x = LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation'])(input_d_chile)
    else:
      x = Bidirectional( LSTM(param_grid['units_lstm'], return_sequences=False, activation=param_grid['activation']) )( input_d_chile ) 
    out_d_chile = Dropout(param_grid['dropout_rate'])(x)  
  #output types
  if not chile_opt and not region_opt:
    z = out_d_commune
  elif not chile_opt and region_opt:
    z = Concatenate(axis=1)([out_d_commune, out_d_region])
  elif chile_opt and not region_opt:
    z = Concatenate(axis=1)([out_d_commune, out_d_chile])
  elif chile_opt and region_opt:
    z = Concatenate(axis=1)([out_d_commune, out_d_region, out_d_chile])
  z = Dense(param_grid['units_hidden'], activation=param_grid['activation'], kernel_initializer=initializer)(z)
  z = Dropout(param_grid['dropout_rate'])(z)
  out = Dense(n_clases, activation='sigmoid', kernel_initializer=initializer)(z)
  #model
  if not chile_opt and not region_opt:
    model = Model(inputs = [input_d_commune], outputs = [out]) 
  elif not chile_opt and region_opt:
    model = Model(inputs = [input_d_commune, input_d_region], outputs = [out]) 
  elif chile_opt and not region_opt:
    model = Model(inputs = [input_d_commune, input_d_chile], outputs = [out]) 
  elif chile_opt and region_opt:
    model = Model(inputs = [input_d_commune, input_d_region, input_d_chile], outputs = [out]) 
  opt = param_grid['optimizer']
  opt.lr = param_grid['lr']
  model.compile(optimizer = opt, loss = 'binary_crossentropy')
  return model

def transform_dynamic_data(df, n_steps, n_features, scale=False, minmax=False):
  scaler = StandardScaler()
  minmax = MinMaxScaler()
  #make copy
  df = df.copy(deep=True)
  #filter features
  exclude = ['_chile', '_region', '_commune']
  columns = []
  for col in df.columns:
    flag = False
    for exc in exclude:
      if exc in col:
        flag = True
        break
    if flag:
      columns.append(col)
  #flat arrays and pre-processing
  columns = np.array(columns)
  df = df[columns]
  data = []
  for col in columns:
    aux = []
    for val in df[col]:
      aux.extend(val)
    aux = np.array(aux, dtype = np.float32)
    if scale:
      aux = scaler.fit_transform(aux.reshape(-1,1)).flatten()
    if minmax:
      aux = minmax.fit_transform(aux.reshape(-1,1)).flatten()
    data.append( aux )
  #transform features
  n_samples = len(data[0])
  aux_data = []  
  for n_ in range(0, n_samples, n_steps):
    aux_n = []
    for s_ in range(n_, n_+n_steps):
      aux_s = []
      for f_ in range(n_features):
        aux_s.append(data[f_][s_])
      aux_n.append(aux_s)
    aux_data.append(aux_n)
  data = np.array(aux_data) 
  del aux_data
  return data

def features_combination(df):
  #make copy
  df = df.copy(deep=True)
  #filter features
  exclude = ['_chile', '_region', '_commune']
  columns = []
  for col in df.columns:
    flag = False
    for exc in exclude:
      if exc in col:
        flag = True
        break
    if flag:
      columns.append(col)
  #group features
  group = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
  groups = []
  for g_ in group:
    aux_g = []
    flag = False
    for c_ in columns:
      if re.findall(r'^%s\_'% g_, c_):
        flag = True
        aux_g.append(c_)
    if flag:
      groups.append( aux_g )
  #combine features
  combs = []
  for n in range(1, len(groups)+1):
    for cmb in combinations(groups, n):
      combs.append( [f_ for l_ in cmb for f_ in l_] )
  return combs

def analysis_communes_features(df, y, cmbs, clfs, n_steps, param_grid, n_clases):
  results = defaultdict(list)
  #min training examples
  aux_count = [ str(y[0]) ]
  min_training = 1
  while len(set(aux_count))!=2:
    aux_count.append( str(y[min_training]) )
    min_training+=1
  #combination
  for cmb in cmbs:
    df_aux = df[cmb]
    n_features_d_commune = df_aux.shape[1]
    X_d_commune_cmb = transform_dynamic_data(df_aux, n_steps, n_features_d_commune, scale = True)
    aux_cmb = [key[2:].replace('_commune', '') for key in cmb]
    key_cmb = '+'.join(aux_cmb)
    #walk forward validation
    for i in range(min_training, df_aux.shape[0]-1):
      train_index = np.arange(0,i+1)
      test_index = np.arange(i+1, i+2)
      trainX_d_commune = X_d_commune_cmb[train_index, :, :]    
      trainX_d_step_commune = np.mean(trainX_d_commune, axis = 1)
      trainY = y[train_index]
      testX_d_commune = X_d_commune_cmb[test_index, :, :]
      testX_d_step_commune = np.mean(testX_d_commune, axis = 1)
      testY = y[test_index]
      for key_clf in clfs:
        clf = clfs[key_clf]
        if 'SVC' in clf.__class__.__name__:
          clf.fit(trainX_d_step_commune, trainY)
          prediction = clf.predict(testX_d_step_commune)
          results[key_cmb+'_'+key_clf].append( [testY, prediction] )
        else:
          clf = LSTM_model(param_grid, n_clases, n_steps, n_features_d_commune, -1, -1, False, False, False, False)
          clf.fit(x = trainX_d_commune, y = [trainY], epochs = param_grid['epochs'], batch_size=param_grid['batch_size'], verbose=0, validation_split = 0)
          prob = clf.predict( testX_d_commune )
          prediction = int(prob>=0.5)
          results[key_cmb+'_'+'lstm'].append( [testY, prediction] )          
          clf = LSTM_model(param_grid, n_clases, n_steps, n_features_d_commune, -1, -1, False, False, False, True)
          clf.fit(x = trainX_d_commune, y = [trainY], epochs = param_grid['epochs'], batch_size=param_grid['batch_size'], verbose=0, validation_split = 0)
          prob = clf.predict( testX_d_commune )
          prediction = int(prob>=0.5)
          results[key_cmb+'_'+'bi-lstm'].append( [testY, prediction] )   
          clf = LSTM_model(param_grid, n_clases, n_steps, n_features_d_commune, -1, -1, False, False, True, False)
          clf.fit(x = trainX_d_commune, y = [trainY], epochs = param_grid['epochs'], batch_size=param_grid['batch_size'], verbose=0, validation_split = 0)
          prob = clf.predict( testX_d_commune )
          prediction = int(prob>=0.5)
          results[key_cmb+'_'+'stacked-lstm'].append( [testY, prediction] )   
  return results

def analysys_comb_features(df_commune, df_feature, y, clfs, n_steps, param_grid, n_clases, region_opt, chile_opt):
  results = defaultdict(list)
  commune_features = [column for column in df_commune.columns if re.findall(r'^[A-Z]\_', column)]
  columns = df_feature.columns
  group = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
  other_features = []
  for g_ in group:
    aux_g = []
    flag = False
    for c_ in columns:
      if re.findall(r'^%s\_'% g_, c_):
        flag = True
        aux_g.append(c_)
    if flag:
      other_features.append( aux_g )  
  #min training examples
  aux_count = [ str(y[0]) ]
  min_training = 1
  while len(set(aux_count))!=2:
    aux_count.append( str(y[min_training]) )
    min_training+=1
  #communes
  df_aux = df_commune[commune_features]
  n_features_d_commune = df_aux.shape[1]
  X_d_commune = transform_dynamic_data(df_aux, n_steps, n_features_d_commune, scale = True)
  n_features_d_region = 0
  n_features_d_chile = 0
  for feature in other_features:
    #other feature (regional or national)
    df_aux = pd.DataFrame(columns=feature) 
    df_aux[feature] = df_feature[feature]
    X_d_feature = transform_dynamic_data(df_aux, n_steps, 1, scale = True)
    key_cmb = 'commune+'+'+'.join( feature )
    if region_opt:
      n_features_d_region = 1
    elif chile_opt:
      n_features_d_chile = 1
    #DL models
    single_lstm = LSTM_model(param_grid, n_clases, n_steps, n_features_d_commune, n_features_d_region, n_features_d_chile, region_opt, chile_opt, False, False)
    stack_lstm = LSTM_model(param_grid, n_clases, n_steps, n_features_d_commune, n_features_d_region, n_features_d_chile, region_opt, chile_opt, False, True)
    bi_lstm = LSTM_model(param_grid, n_clases, n_steps, n_features_d_commune, n_features_d_region, n_features_d_chile, region_opt, chile_opt, True, False)
    #walk forward validation
    for i in range(min_training, df_aux.shape[0]-1):
      train_index = np.arange(0,i+1)
      test_index = np.arange(i+1, i+2)
      trainX_d_commune = X_d_commune[train_index, :, :]   
      trainX_d_feature = X_d_feature[train_index, :, :]
      trainX_d_step_commune = np.mean(trainX_d_commune, axis = 1)
      trainX_d_step_feature = np.mean(trainX_d_feature, axis = 1)
      trainY = y[train_index]
      testX_d_commune = X_d_commune[test_index, :, :]
      testX_d_feature = X_d_feature[test_index, :, :]
      testX_d_step_commune = np.mean(testX_d_commune, axis = 1)
      testX_d_step_feature = np.mean(testX_d_feature, axis = 1)
      testY = y[test_index]
      X_training = [trainX_d_commune, trainX_d_feature]
      X_test = [testX_d_commune, testX_d_feature]
      X_training_step = np.hstack( (trainX_d_step_commune, trainX_d_step_feature) )
      X_test_step = np.hstack( (testX_d_step_commune, testX_d_step_feature) )
      for key_clf in clfs:
        clf = clfs[key_clf]
        if 'SVC' in clf.__class__.__name__:
          clf.fit(X_training_step, trainY)
          prediction = clf.predict(X_test_step)
          results[key_cmb+'_'+key_clf].append( [testY, prediction] )
        else:
          single_lstm.fit(x = X_training, y = [trainY], epochs = param_grid['epochs'], batch_size=param_grid['batch_size'], verbose=0, validation_split = 0)
          prob = single_lstm.predict( X_test )
          prediction = int(prob>=0.5)
          results[key_cmb+'_'+'lstm'].append( [testY, prediction] )          
          bi_lstm.fit(x = X_training, y = [trainY], epochs = param_grid['epochs'], batch_size=param_grid['batch_size'], verbose=0, validation_split = 0)
          prob = bi_lstm.predict( X_test )
          prediction = int(prob>=0.5)
          results[key_cmb+'_'+'bi-lstm'].append( [testY, prediction] )   
          stack_lstm.fit(x = X_training, y = [trainY], epochs = param_grid['epochs'], batch_size=param_grid['batch_size'], verbose=0, validation_split = 0)
          prob = stack_lstm.predict( X_test )
          prediction = int(prob>=0.5)
          results[key_cmb+'_'+'stacked-lstm'].append( [testY, prediction] )   
  return results

####################################################################################################

def latex_results(communes, predictions_clf):
  metrics = ['ACC(\%)', 'P(\%)', 'R(\%)']
  for commune in communes:
    code = 'Classifier & ACC (\%) & P (\%) & R (\%) & & F1 (\%) \\\ \\hline \\hline \n'
    for key in predictions_clf:
      if re.findall(r'^%s' %commune, key):
        clf = key.replace(commune+'_', '')
        true = [t for t,p in predictions_clf[key] ]
        predictions = [p for t,p in predictions_clf[key] ]
        for metric in metrics:
          if metric == 'ACC(\%)':
            acc = '{:.2f}'.format( np.round(100*accuracy_score(true, predictions), 2) )
          elif metric == 'P(\%)':
            p_val = np.round(100*precision_score(true, predictions, pos_label = 1, average = 'weighted'), 2)
            p = '{:.2f}'.format( np.round(100*precision_score(true, predictions, pos_label = 1, average = 'weighted'), 2) )
          elif metric == 'R(\%)':
            r_val = np.round(100*recall_score(true, predictions, pos_label = 1, average = 'weighted'), 2)
            r = '{:.2f}'.format( np.round(100*recall_score(true, predictions, pos_label = 1, average = 'weighted'), 2) )
            f1 = '{:.2f}'.format( np.round((2*p_val*r_val)/(p_val+r_val), 2) )
        code += clf.upper().replace('_', '-')+' & '+acc+' & '+p+' & '+r+' & '+f1+'\\\ \n'
    print(commune)
    print( code )
    
def plot_dynamic_features_commune(results, commune, metric, idx):
  colors_clf = {'_SVM-RBF': 'royalblue', '_SVM-LINEAR': 'lightsteelblue',
            '_LSTM': 'darkred', '_BI-LSTM': 'indianred', '_STACKED-LSTM': 'rosybrown'}
  handles = []
  for key_clf in colors_clf:
    handles.append( mpatches.Patch(color=colors_clf[key_clf], label=key_clf[1:]) )
  fontsize = 20
  rc('xtick', labelsize=fontsize) 
  rc('ytick', labelsize=fontsize) 
  keys = results.keys()
  values = []
  colors = []
  for key in keys:
    true = [t for t,p in results[key] ]
    predictions = [p for t,p in results[key] ]
    if metric=='acc':
      values.append( ( key.replace('+', '\n').upper(), np.round(100*accuracy_score(true, predictions),2) ) ) 
    elif metric == 'f1':
      values.append( ( key.replace('+', '\n').upper(), np.round(100*f1_score(true, predictions),2) ) ) 
  values = sorted(values, key = lambda x:x[1], reverse=False)
  fig = pylab.figure(idx, figsize=(30,15))
  labels = [k for k,v in values]
  for index_l in range(len(labels)):
    labels[index_l] = labels[index_l].replace('MOV_EXT\nMOV_INT', 'MOBILITY')
    labels[index_l] = labels[index_l].replace('_COVID_', '_')
    labels[index_l] = re.sub( r'\n[A-Z]\_', '\n', labels[index_l] )
    labels[index_l] = re.sub( r'<=39_CHILE\n40-49_CHILE\n50-59_CHILE\n60-69_CHILE\n70-79_CHILE\n80-89_CHILE\n>=90_CHILE', 'AGE_GROUP_1_CHILE', labels[index_l] )
    labels[index_l] = re.sub( r'<=39_CHILE\n40-49_CHILE\n50-59_CHILE\n60-69_CHILE\n>=70_CHILE', 'AGE_GROUP_2_CHILE', labels[index_l] )    
    labels[index_l] = re.sub( r'BASICA_CHILE\nMEDIA_CHILE\nUTI_CHILE\nUCI_CHILE', 'BED_TYPE_CHILE', labels[index_l] )    
    labels[index_l] = re.sub( r'VENTILADORES_CHILE', 'MV_CHILE', labels[index_l] )
    labels[index_l] = re.sub( r'PACIENTES VMI_CHILE', 'IMV_CHILE', labels[index_l] )
    labels[index_l] = re.sub( r'PACIENTES CRITICOS_CHILE', 'CRT_PAT_CHILE', labels[index_l] )
    labels[index_l] = re.sub( r'COMMUNE\n', '', labels[index_l] )    
    for key_c in colors_clf:
      if re.findall(r'%s$' %key_c, labels[index_l]):
        labels[index_l] = re.sub(r'%s$' %key_c, '', labels[index_l])
        colors.append( colors_clf[key_c] )
  performance = [v for k,v in values]
  y_pos = np.arange(len(labels))
  pylab.barh(y_pos, performance,  align='center', color = colors)
  pylab.xlim([min(performance)-5, max(performance)+5])
  pylab.yticks(y_pos, labels)
  pylab.xlabel(metric.upper()+' (%)', fontsize=fontsize)
  pylab.title(commune, fontsize=fontsize)
  pylab.grid(True)
  pylab.legend(handles = handles, loc = 'lower right', fontsize=fontsize )
  for i, v in enumerate(performance):
    pylab.text(v + 0.5, i + 0, str(v)+' (%)', color='black', fontsize=fontsize)
  pylab.show()
  