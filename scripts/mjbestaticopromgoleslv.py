#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:27:00 2019

@author: cristian
"""

import pystan
import pandas as pd
import os
import numpy as np
import sys
import pickle
from time import time
import datetime

"""
FUNCIONES BASE
"""

def ReindexDataFrames(df_cal, df_fix):
    eq_cal = df_cal[['Torneo','Local']].drop_duplicates()['Local'].value_counts().index.tolist()
    eq_val = eq_cal + [e for e in df_fix['Local'].drop_duplicates().tolist() if e not in eq_cal]
    dicteq = dict(zip(eq_val,[idx + 1 for idx in range(len(eq_val))]))
    dictindex = {v: k for k, v in dicteq.items()}
    df_cal['i_local'] = [dicteq[e] for e in df_cal['Local'].tolist()]
    df_cal['i_visita'] = [dicteq[e] for e in df_cal['Visita'].tolist()]
    df_fix['i_local'] = [dicteq[e] for e in df_fix['Local'].tolist()]
    df_fix['i_visita'] = [dicteq[e] for e in df_fix['Visita'].tolist()]
    return df_cal, df_fix, dictindex

def DataFrameToStanData(df_cal):
    n_equipos = max(df_cal['i_local'].max(), df_cal['i_visita'].max())
    n_partidos = df_cal.shape[0]
    # dictpriors = {'equipos': n_equipos,
    #               'partidos': n_partidos,
    #               'mu_pi': 0,
    #               'sd_pi': 0,
    #               'mu_psm': 0,
    #               'sd_psm': 0,
    #               'mu_ph' : [0 for i in range(n_equipos)],
    #               'sd_ph' : [1 for i in range(n_equipos)],
    #               'mu_pa' : [0 for i in range(n_equipos)],
    #               'sd_pa': [0 for i in range(n_equipos)],
    #               'mu_pd' : [0 for i in range(n_equipos)],
    #               'sd_pd' : [1 for i in range(n_equipos)]}
    data = {}
    for col in df_cal.rename(columns = {'i_local': 'local',
                                        'i_visita': 'visita',
                                        'goles L': 'glocal',
                                        'goles V': 'gvisita'}).columns:
        if col != 'Round':
            data[col] = df_cal.rename(columns = {'i_local': 'local',
                                                 'i_visita': 'visita',
                                                 'goles L': 'glocal',
                                                 'goles V': 'gvisita'})[col].tolist()
    data['equipos'] = n_equipos
    data['partidos'] = n_partidos
    return data



"""
FUNCIÓN ÁNIMO SIMPLE
"""
def AgregaParametrosInicialesSimple(df_fix, fit, draw):
    datadict = {}
    for key in ['intercept','homes','atts','defs']:
        datadict[key] = fit.extract(permuted = True)[key][draw]
    df_params = pd.DataFrame(datadict)
    df_params.index = df_params.index + 1
    df_params_l = df_params[['intercept','homes','atts','defs']]
    df_params_l.columns = ['intercept','home','att_local','def_local']
    df_params_v = df_params[['atts','defs']]
    df_params_v.columns = ['att_visita','def_visita']
    df_fix = df_fix.merge(df_params_l, how = 'left', left_on = 'i_local', right_index = True)
    df_fix = df_fix.merge(df_params_v, how = 'left', left_on = 'i_visita', right_index = True)
    df_fix = df_fix.fillna(df_fix.mean())
    df_fix['tasa_local_base'] = np.exp(df_fix['intercept'] + 
               df_fix['home'] + 
               df_fix['att_local'] - 
               df_fix['def_visita'])
    df_fix['tasa_visita_base'] = np.exp(df_fix['intercept'] + 
               df_fix['att_visita'] - 
               df_fix['def_local'])
    df_fix['sm_local'] = 0
    df_fix['sm_visita'] = 0
    df_fix['goles L'] = 0
    df_fix['goles V'] = 0
    df_fix = df_fix[['Round','i_local','i_visita','tasa_local_base','tasa_visita_base', 'sm_local', 'sm_visita','goles L','goles V']]    
    return df_fix

def SimularRondaSimple(arr_ronda, sm):
    # 0 a 4: columnas base
    # 5, 6: stamind_local y stamind_visita
    n_matches = arr_ronda.shape[0]
    tasa_local = arr_ronda[:,3] * np.exp(sm*arr_ronda[:,5])
    tasa_visita = arr_ronda[:,4] * np.exp(sm*arr_ronda[:,6])
    arr_ronda[:,7] = np.minimum(np.random.poisson(lam = tasa_local, size = (1,n_matches)),8)
    arr_ronda[:,8] = np.minimum(np.random.poisson(lam = tasa_visita, size = (1,n_matches)),8)
    # goles están en las columnas 7 y 8
    return arr_ronda

def ActualizarAnimoSimple(arr_ronda_a, dict_sm_or):
    dict_sm = dict_sm_or
    teams = [item for sublist in arr_ronda_a[:,[1, 2]].tolist() for item in sublist]
    matches_outcomes = arr_ronda_a[:,[7,8]].tolist()
    resultados = []
    for j in range(len(matches_outcomes)):
        if matches_outcomes[j][0] > matches_outcomes[j][1]:
            r1 = 1
            r2 = -1
        elif matches_outcomes[j][0] < matches_outcomes[j][1]:
            r1 = -1
            r2 = 1
        else:
            r1 = 0
            r2 = 0
        resultados.append(r1)
        resultados.append(r2)
    dict_replaces = dict(zip(teams,resultados))
    for key, value in dict_replaces.items():
        dict_sm[key] += value
    return [dict_sm]

def SimularTorneoSimple(df_fix_or, rondas, ids_torneo, trace):
    df_fix = df_fix_or.copy()
    draw = np.random.randint(0, 1000)
    df_fix = AgregaParametrosInicialesSimple(df_fix, trace, draw)
    dict_sm = [dict(zip([j for j in ids_torneo],[0 for i in ids_torneo]))]
    # Agrega funciones de estado de animo iniciales y columnas de goles
    fix_arr = df_fix.values.astype(float)
    fix_arr[fix_arr[:,0] == 1] = SimularRondaSimple(fix_arr[fix_arr[:,0] == 1], sm = trace['sm'][draw])
    # Actualizar funcion de ánimo
    dict_sm = ActualizarAnimoSimple(fix_arr[fix_arr[:,0] == 1], *dict_sm)        
    for r in rondas[1:]:
        fix_arr[fix_arr[:,0] == r, 5] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 1]])
        fix_arr[fix_arr[:,0] == r, 6] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 2]])
        fix_arr[fix_arr[:,0] == r] = SimularRondaSimple(fix_arr[fix_arr[:,0] == r], sm = trace['sm'][draw])
        # Actualizar funcion de ánimo
        dict_sm = ActualizarAnimoSimple(fix_arr[fix_arr[:,0] == r], *dict_sm)
    return pd.DataFrame(fix_arr[:, [0,1,2,7,8]], columns = ['Round','Local','Visita','goles L', 'goles V'])

def SimularMCMCSimple(df_cal, df_fix, sm, outputdir, n_sim = 'test'):
#     ti = time()
    # Filtrar torneo calibracion y fixture a simular
    rondas = df_fix['Round'].drop_duplicates().tolist()
    # Reindexar según los equipos participantes
    df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
    df_cal = df_cal[['Round','i_local','i_visita','goles L','goles V', 'sm_local', 'sm_visita']]
    ids_torneo = df_fix['i_local'].drop_duplicates().tolist()
    # Calibración con datos históricos
    trace = sm.sampling(data=DataFrameToStanData(df_cal), iter=1000, chains=2, n_jobs = 1)
    df_sim = [SimularTorneoSimple(df_fix, rondas, ids_torneo, trace) for n in range(1000)]
    df_sim = pd.concat(df_sim, ignore_index = True)
    df_sim['Local'] = [dictindex[int(i)] for i in df_sim['Local'].tolist()]
    df_sim['Visita'] = [dictindex[int(i)] for i in df_sim['Visita'].tolist()]
    return df_sim

"""
FUNCIÓN ÁNIMO LV
"""

def AgregaParametrosInicialesLV(df_fix, fit, draw):
    datadict = {}
    for key in ['intercept','homes','atts','defs']:
        datadict[key] = fit.extract(permuted = True)[key][draw]
    df_params = pd.DataFrame(datadict)
    df_params.index = df_params.index + 1
    df_params_l = df_params[['intercept','homes','atts','defs']]
    df_params_l.columns = ['intercept','home','att_local','def_local']
    df_params_v = df_params[['atts','defs']]
    df_params_v.columns = ['att_visita','def_visita']
    df_fix = df_fix.merge(df_params_l, how = 'left', left_on = 'i_local', right_index = True)
    df_fix = df_fix.merge(df_params_v, how = 'left', left_on = 'i_visita', right_index = True)
    df_fix = df_fix.fillna(df_fix.mean())
    df_fix['tasa_local_base'] = np.exp(df_fix['intercept'] + 
               df_fix['home'] + 
               df_fix['att_local'] - 
               df_fix['def_visita'])
    df_fix['tasa_visita_base'] = np.exp(df_fix['intercept'] + 
               df_fix['att_visita'] - 
               df_fix['def_local'])
    df_fix['sml_local'] = 0
    df_fix['sml_visita'] = 0
    df_fix['goles L'] = 0
    df_fix['goles V'] = 0
    df_fix = df_fix[['Round','i_local','i_visita','tasa_local_base','tasa_visita_base', 'sml_local', 'sml_visita','goles L','goles V']]    
    return df_fix

def SimularRondaLV(arr_ronda, sml, smv):
    # 0 a 4: columnas base
    # 5, 6: stamind_local y stamind_visita
    n_matches = arr_ronda.shape[0]
    tasa_local = arr_ronda[:,3] * np.exp(sml*arr_ronda[:,5])
    tasa_visita = arr_ronda[:,4] * np.exp(smv*arr_ronda[:,6])
    arr_ronda[:,7]= np.minimum(np.random.poisson(lam = tasa_local, size = (1,n_matches)),8)
    arr_ronda[:,8] = np.minimum(np.random.poisson(lam = tasa_visita, size = (1,n_matches)),8)
    # goles están en las columnas 7 y 8
    return arr_ronda

def ActualizarAnimoLV(arr_ronda_a, dict_sml_or, dict_smv_or):
    dict_sml = dict_sml_or
    dict_smv = dict_smv_or
    teams_l = arr_ronda_a[:,1].tolist()
    teams_v = arr_ronda_a[:,2].tolist()
    conds = [arr_ronda_a[:,7] > arr_ronda_a[:,8], arr_ronda_a[:,7] < arr_ronda_a[:,8]]
    choices = [1,-1]
    res = np.select(conds,choices, default = 0)
    resultados_l = res.tolist()
    resultados_v = (-res).tolist()

    dict_replaces_l = dict(zip(teams_l,resultados_l))
    dict_replaces_v = dict(zip(teams_v,resultados_v))  
    for key, value in dict_replaces_l.items():
        dict_sml[key] += value

    for key, value in dict_replaces_v.items():
        dict_smv[key] += value

    return [dict_sml, dict_smv]

def SimularTorneoLV(df_fix_or, rondas, ids_torneo, trace):
    df_fix = df_fix_or.copy()
    draw = np.random.randint(0, 1000)
    df_fix = AgregaParametrosInicialesLV(df_fix, trace, draw)
    dict_sm = [dict(zip([j for j in ids_torneo],[0 for i in ids_torneo]))]*2
    # Agrega funciones de estado de animo iniciales y columnas de goles
    fix_arr = df_fix.values.astype(float)
    fix_arr[fix_arr[:,0] == 1] = SimularRondaLV(fix_arr[fix_arr[:,0] == 1], sml = trace['sml'][draw], smv = trace['smv'][draw])
    # Actualizar funcion de ánimo
    dict_sm = ActualizarAnimoLV(fix_arr[fix_arr[:,0] == 1], *dict_sm)        
    for r in rondas[1:]:
        fix_arr[fix_arr[:,0] == r, 5] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 1]])
        fix_arr[fix_arr[:,0] == r, 6] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 2]])
        fix_arr[fix_arr[:,0] == r] = SimularRondaLV(fix_arr[fix_arr[:,0] == r],sml = trace['sml'][draw], smv = trace['smv'][draw])
        # Actualizar funcion de ánimo
        dict_sm = ActualizarAnimoLV(fix_arr[fix_arr[:,0] == r], *dict_sm)
    return pd.DataFrame(fix_arr[:, [0,1,2,7,8]], columns = ['Round','Local','Visita','goles L', 'goles V'])

def SimularMCMCLV(df_cal, df_fix, sm, outputdir, n_sim = 'test'):
#     ti = time()
    # Filtrar torneo calibracion y fixture a simular
    rondas = df_fix['Round'].drop_duplicates().tolist()
    # Reindexar según los equipos participantes
    df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
    df_cal = df_cal[['Round','i_local','i_visita','goles L','goles V', 'sml_local', 'smv_visita']]
    ids_torneo = df_fix['i_local'].drop_duplicates().tolist()
    # Calibración con datos históricos
    trace = sm.sampling(data=DataFrameToStanData(df_cal), iter=1000, chains=2, n_jobs = 1)
    df_sim = [SimularTorneoLV(df_fix, rondas, ids_torneo, trace) for n in range(1000)]
    df_sim = pd.concat(df_sim, ignore_index = True)
    df_sim['Local'] = [dictindex[int(i)] for i in df_sim['Local'].tolist()]
    df_sim['Visita'] = [dictindex[int(i)] for i in df_sim['Visita'].tolist()]
    return df_sim

"""
FUNCIÓN ÁNIMO PROM GOLES
"""

def AgregaParametrosInicialesPromGoles(df_fix, fit, draw):
    datadict = {}
    for key in ['intercept','homes','atts','defs']:
        datadict[key] = fit.extract(permuted = True)[key][draw]
    df_params = pd.DataFrame(datadict)
    df_params.index = df_params.index + 1
    df_params_l = df_params[['intercept','homes','atts','defs']]
    df_params_l.columns = ['intercept','home','att_local','def_local']
    df_params_v = df_params[['atts','defs']]
    df_params_v.columns = ['att_visita','def_visita']
    df_fix = df_fix.merge(df_params_l, how = 'left', left_on = 'i_local', right_index = True)
    df_fix = df_fix.merge(df_params_v, how = 'left', left_on = 'i_visita', right_index = True)
    df_fix = df_fix.fillna(df_fix.mean())
    df_fix['tasa_local_base'] = np.exp(df_fix['intercept'] + 
               df_fix['home'] + 
               df_fix['att_local'] - 
               df_fix['def_visita'])
    df_fix['tasa_visita_base'] = np.exp(df_fix['intercept'] + 
               df_fix['att_visita'] - 
               df_fix['def_local'])
    df_fix['pgf_local'] = 0
    df_fix['pgf_visita'] = 0
    df_fix['pga_local'] = 0
    df_fix['pga_visita'] = 0    
    df_fix['goles L'] = 0
    df_fix['goles V'] = 0
    df_fix = df_fix[['Round','i_local','i_visita','tasa_local_base','tasa_visita_base', 'pgf_local', 'pgf_visita','pga_local','pga_visita','goles L','goles V']]    
    return df_fix

def SimularRondaPromGoles(arr_ronda, pgf, pga):
    # 0 a 4: columnas base
    # 5, 6: pgf local, pgf visita
    # 7, 8: pga local, pga visita
    n_matches = arr_ronda.shape[0]
    tasa_local = arr_ronda[:,3] * np.exp(pgf*arr_ronda[:,5] + pga*arr_ronda[:,8])
    tasa_visita = arr_ronda[:,4] * np.exp(pgf*arr_ronda[:,6] + pga*arr_ronda[:,7])
    arr_ronda[:,9] = np.minimum(np.random.poisson(lam = tasa_local, size = (1,n_matches)),8)
    arr_ronda[:,10] = np.minimum(np.random.poisson(lam = tasa_visita, size = (1,n_matches)),8)
    # goles están en las columnas 9 y 10
    return arr_ronda

def ActualizarAnimoPromGoles(arr_ronda_a, dict_pgf_or, dict_pga_or):
    ronda = arr_ronda_a[:,0].tolist()[0] # Saca el numero de la ronda anterior
    dict_pgf = dict_pgf_or
    dict_pga = dict_pga_or
    resultados_gf = []
    resultados_ga = []
    teams = [item for sublist in arr_ronda_a[:,[1, 2]].tolist() for item in sublist]
    matches_outcomes = arr_ronda_a[:,[9,10]].tolist()
    for j in range(len(matches_outcomes)):
        resultados_gf.append(matches_outcomes[j][0])
        resultados_ga.append(matches_outcomes[j][1])
        resultados_gf.append(matches_outcomes[j][1])
        resultados_ga.append(matches_outcomes[j][0])
    dict_replaces_gf = dict(zip(teams, resultados_gf))
    dict_replaces_ga = dict(zip(teams, resultados_ga))
    for key, value in dict_replaces_gf.items():
        dict_pgf[key] = (dict_pgf[key]*(ronda-1) + value)/ronda
    
    dict_replaces_ga = dict(zip(teams, resultados_ga))
    for key, value in dict_replaces_ga.items():
        dict_pga[key] = (dict_pga[key]*(ronda-1) + value)/ronda
    return [dict_pgf, dict_pga]

def SimularTorneoPromGoles(df_fix_or, rondas, ids_torneo, trace):
    df_fix = df_fix_or.copy()
    draw = np.random.randint(0, 1000)
    df_fix = AgregaParametrosInicialesPromGoles(df_fix, trace, draw)
    dict_sm = [dict(zip([j for j in ids_torneo],[0 for i in ids_torneo]))]*2
    # Agrega funciones de estado de animo iniciales y columnas de goles
    fix_arr = df_fix.values.astype(float)
    fix_arr[fix_arr[:,0] == 1] = SimularRondaPromGoles(fix_arr[fix_arr[:,0] == 1], pgf = trace['pgf'][draw], pga = trace['pga'][draw])
    # Actualizar funcion de ánimo
    dict_sm = ActualizarAnimoPromGoles(fix_arr[fix_arr[:,0] == 1], *dict_sm)        
    for r in rondas[1:]:
        fix_arr[fix_arr[:,0] == r, 5] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 1]])
        fix_arr[fix_arr[:,0] == r, 6] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 2]])
        fix_arr[fix_arr[:,0] == r] = SimularRondaPromGoles(fix_arr[fix_arr[:,0] == r], pgf = trace['pgf'][draw], pga = trace['pga'][draw])
        # Actualizar funcion de ánimo
        dict_sm = ActualizarAnimoPromGoles(fix_arr[fix_arr[:,0] == r], *dict_sm)
    return pd.DataFrame(fix_arr[:, [0,1,2,9,10]], columns = ['Round','Local','Visita','goles L', 'goles V'])

def SimularMCMCPromGoles(df_cal, df_fix, sm, outputdir, n_sim = 'test'):
#     ti = time()
    # Filtrar torneo calibracion y fixture a simular
    rondas = df_fix['Round'].drop_duplicates().tolist()
    # Reindexar según los equipos participantes
    df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
    df_cal = df_cal[['Round','i_local','i_visita','goles L','goles V', 'pgf_local', 'pgf_visita', 'pga_local','pga_visita']]
    ids_torneo = df_fix['i_local'].drop_duplicates().tolist()
    # Calibración con datos históricos
    trace = sm.sampling(data=DataFrameToStanData(df_cal), iter=1000, chains=2, n_jobs = 1)
    df_sim = [SimularTorneoPromGoles(df_fix, rondas, ids_torneo, trace) for n in range(1000)]
    df_sim = pd.concat(df_sim, ignore_index = True)
    df_sim['Local'] = [dictindex[int(i)] for i in df_sim['Local'].tolist()]
    df_sim['Visita'] = [dictindex[int(i)] for i in df_sim['Visita'].tolist()]
    return df_sim

"""
FUNCIÓN ÁNIMO PROM GOLES LV

"""

def AgregaParametrosInicialesPromGolesLV(df_fix, fit, draw):
    datadict = {}
    for key in ['intercept','homes','atts','defs']:
        datadict[key] = fit.extract(permuted = True)[key][draw]
    df_params = pd.DataFrame(datadict)
    df_params.index = df_params.index + 1
    df_params_l = df_params[['intercept','homes','atts','defs']]
    df_params_l.columns = ['intercept','home','att_local','def_local']
    df_params_v = df_params[['atts','defs']]
    df_params_v.columns = ['att_visita','def_visita']
    df_fix = df_fix.merge(df_params_l, how = 'left', left_on = 'i_local', right_index = True)
    df_fix = df_fix.merge(df_params_v, how = 'left', left_on = 'i_visita', right_index = True)
    df_fix = df_fix.fillna(df_fix.mean())
    df_fix['tasa_local_base'] = np.exp(df_fix['intercept'] + 
               df_fix['home'] + 
               df_fix['att_local'] - 
               df_fix['def_visita'])
    df_fix['tasa_visita_base'] = np.exp(df_fix['intercept'] + 
               df_fix['att_visita'] - 
               df_fix['def_local'])
    df_fix['pgfl_local'] = 0
    df_fix['pgfv_visita'] = 0
    df_fix['pgal_local'] = 0
    df_fix['pgav_visita'] = 0    
    df_fix['goles L'] = 0
    df_fix['goles V'] = 0
    df_fix = df_fix[['Round','i_local','i_visita','tasa_local_base','tasa_visita_base', 'pgfl_local', 'pgfv_visita','pgal_local','pgav_visita','goles L','goles V']]    
    return df_fix

def SimularRondaPromGolesLV(arr_ronda, pgfl, pgfv, pgal, pgav):
    # 0 a 4: columnas base
    # 5, 6: pgfl local, pgfv visita
    # 7, 8: pgal local, pgav visita
    n_matches = arr_ronda.shape[0]
    tasa_local = arr_ronda[:,3] * np.exp(pgfl*arr_ronda[:,5] + pgav*arr_ronda[:,8])
    tasa_visita = arr_ronda[:,4] * np.exp(pgfv*arr_ronda[:,6] + pgal*arr_ronda[:,7])
    arr_ronda[:,9] = np.minimum(np.random.poisson(lam = tasa_local, size = (1,n_matches)),8)
    arr_ronda[:,10] = np.minimum(np.random.poisson(lam = tasa_visita, size = (1,n_matches)),8)
    # goles están en las columnas 9 y 10
    return arr_ronda

def ActualizarAnimoPromGolesLV(arr_ronda_a, dict_pgfl_or, dict_pgfv_or, dict_pgal_or, dict_pgav_or):
    ronda = arr_ronda_a[:,0].tolist()[0] # Saca el numero de la ronda anterior
    dict_pgfl = dict_pgfl_or
    dict_pgal = dict_pgal_or
    dict_pgfv = dict_pgfv_or
    dict_pgav = dict_pgav_or
    resultados_gfl = []
    resultados_gal = []
    resultados_gfv = []
    resultados_gav = []
    teams_l = [i for i in arr_ronda_a[:,1].tolist()]
    teams_v = [i for i in arr_ronda_a[:,2].tolist()]
    matches_outcomes = arr_ronda_a[:,[9,10]].tolist()
    for j in range(len(matches_outcomes)):
        resultados_gfl.append(matches_outcomes[j][0])
        resultados_gal.append(matches_outcomes[j][1])
        resultados_gfv.append(matches_outcomes[j][1])
        resultados_gav.append(matches_outcomes[j][0])

    # Reemplazos local
    dict_replaces_gfl = dict(zip(teams_l, resultados_gfl))
    dict_replaces_gal = dict(zip(teams_l, resultados_gal))
    dict_replaces_gfv = dict(zip(teams_v, resultados_gfv))
    dict_replaces_gav = dict(zip(teams_v, resultados_gav))
    for key, value in dict_replaces_gfl.items():
        dict_pgfl[key] = (dict_pgfl[key]*(ronda-1) + value)/ronda

    dict_replaces_gal = dict(zip(teams_l, resultados_gal))
    for key, value in dict_replaces_gal.items():
        dict_pgal[key] = (dict_pgal[key]*(ronda-1) + value)/ronda
        
    # Reemplazos visita
    dict_replaces_gfv = dict(zip(teams_v, resultados_gfv))
    for key, value in dict_replaces_gfv.items():
        dict_pgfv[key] = (dict_pgfv[key]*(ronda-1) + value)/ronda
    dict_replaces_gav = dict(zip(teams_v, resultados_gav))
    for key, value in dict_replaces_gav.items():
        dict_pgav[key] = (dict_pgav[key]*(ronda-1) + value)/ronda
        
    return [dict_pgfl, dict_pgfv, dict_pgal, dict_pgav]

def SimularTorneoPromGolesLV(df_fix_or, rondas, ids_torneo, trace):
    df_fix = df_fix_or.copy()
    draw = np.random.randint(0, 1000)
    df_fix = AgregaParametrosInicialesPromGolesLV(df_fix, trace, draw)
    dict_sm = [dict(zip([j for j in ids_torneo],[0 for i in ids_torneo]))]*4
    # Agrega funciones de estado de animo iniciales y columnas de goles
    fix_arr = df_fix.values.astype(float)
    fix_arr[fix_arr[:,0] == 1] = SimularRondaPromGolesLV(fix_arr[fix_arr[:,0] == 1], pgfl = trace['pgfl'][draw], pgfv = trace['pgfv'][draw], pgal = trace['pgal'][draw], pgav = trace['pgav'][draw])
    # Actualizar funcion de ánimo
    dict_sm = ActualizarAnimoPromGolesLV(fix_arr[fix_arr[:,0] == 1], *dict_sm)        
    for r in rondas[1:]:
        fix_arr[fix_arr[:,0] == r, 5] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 1]])
        fix_arr[fix_arr[:,0] == r, 6] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 2]])
        fix_arr[fix_arr[:,0] == r] = SimularRondaPromGolesLV(fix_arr[fix_arr[:,0] == r], pgfl = trace['pgfl'][draw], pgfv = trace['pgfv'][draw], pgal = trace['pgal'][draw], pgav = trace['pgav'][draw])
        # Actualizar funcion de ánimo
        dict_sm = ActualizarAnimoPromGolesLV(fix_arr[fix_arr[:,0] == r], *dict_sm)
    return pd.DataFrame(fix_arr[:, [0,1,2,9,10]], columns = ['Round','Local','Visita','goles L', 'goles V'])

def SimularMCMCPromGolesLV(df_cal, df_fix, sm, outputdir, n_sim = 'test'):
#     ti = time()
    # Filtrar torneo calibracion y fixture a simular
    rondas = df_fix['Round'].drop_duplicates().tolist()
    # Reindexar según los equipos participantes
    df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
    df_cal = df_cal[['Round','i_local','i_visita','goles L','goles V', 'pgfl_local', 'pgfv_visita', 'pgal_local','pgav_visita']]
    ids_torneo = df_fix['i_local'].drop_duplicates().tolist()
    # Calibración con datos históricos
    trace = sm.sampling(data=DataFrameToStanData(df_cal), iter=1000, chains=2, n_jobs = 1)
    df_sim = [SimularTorneoPromGolesLV(df_fix, rondas, ids_torneo, trace) for n in range(1000)]
    df_sim = pd.concat(df_sim, ignore_index = True)
    df_sim['Local'] = [dictindex[int(i)] for i in df_sim['Local'].tolist()]
    df_sim['Visita'] = [dictindex[int(i)] for i in df_sim['Visita'].tolist()]
    return df_sim

"""
FUNCIÓN ÁNIMO RATIO ELO
"""
def AgregaParametrosInicialesRatioELO(df_fix, fit, draw):
    datadict = {}
    for key in ['intercept','homes','atts','defs']:
        datadict[key] = fit.extract(permuted = True)[key][draw]
    df_params = pd.DataFrame(datadict)
    df_params.index = df_params.index + 1
    df_params_l = df_params[['intercept','homes','atts','defs']]
    df_params_l.columns = ['intercept','home','att_local','def_local']
    df_params_v = df_params[['atts','defs']]
    df_params_v.columns = ['att_visita','def_visita']
    df_fix = df_fix.merge(df_params_l, how = 'left', left_on = 'i_local', right_index = True)
    df_fix = df_fix.merge(df_params_v, how = 'left', left_on = 'i_visita', right_index = True)
    df_fix = df_fix.fillna(df_fix.mean())
    df_fix['tasa_local_base'] = np.exp(df_fix['intercept'] + 
               df_fix['home'] + 
               df_fix['att_local'] - 
               df_fix['def_visita'])
    df_fix['tasa_visita_base'] = np.exp(df_fix['intercept'] + 
               df_fix['att_visita'] - 
               df_fix['def_local'])
    df_fix['goles L'] = 0
    df_fix['goles V'] = 0
    df_fix = df_fix[['Round','i_local','i_visita','tasa_local_base','tasa_visita_base', 'elo_local', 'elo_visita', 'goles L','goles V']]    
    return df_fix

def SimularRondaRatioELO(arr_ronda, sm):
    # 0 a 4: columnas base
    # 5, 6: elo_local y elo_visita
    lre = np.log(arr_ronda[:,5]/arr_ronda[:,6])
    n_matches = arr_ronda.shape[0]
    tasa_local = arr_ronda[:,3] * np.exp(sm*lre)
    tasa_visita = arr_ronda[:,4] * np.exp(-sm*lre)
    arr_ronda[:,7] = np.random.poisson(lam = tasa_local, size = (1,n_matches))
    arr_ronda[:,8] = np.random.poisson(lam = tasa_visita, size = (1,n_matches))
    # goles están en las columnas 7 y 8
    return arr_ronda

def UpdateELO(l,l_rival, alfa, c = 10, d = 400, k = 20):
    gamma = 1/(1 + c**((l_rival - l)/d))
    return l + k*(alfa-gamma)

def ActualizarAnimoRatioELO(arr_ronda_a, dict_elo_or):
    dict_elo = dict_elo_or
    teams_l = arr_ronda_a[:,1].tolist()
    teams_v = arr_ronda_a[:,2].tolist()
    conds = [arr_ronda_a[:,7] > arr_ronda_a[:,8], arr_ronda_a[:,7] < arr_ronda_a[:,8]]
    choices = [1,0]
    res = np.select(conds,choices, default = 0.5)
    for m in range(len(teams_l)):
        l, v = teams_l[m], teams_v[m]
        elo_l = dict_elo[l]
        elo_v = dict_elo[v]
        dict_elo[l] = UpdateELO(elo_l,elo_v, res[m])
        dict_elo[v] = UpdateELO(elo_v,elo_l, 1 - res[m])
    return [dict_elo]

def SimularTorneoRatioELO(df_fix_or, rondas, ids_torneo, trace):
    df_fix = df_fix_or.copy()
    draw = np.random.randint(0, 1000)
    df_fix = AgregaParametrosInicialesRatioELO(df_fix, trace, draw)
    # Agrega funciones de estado de animo iniciales y columnas de goles
    fix_arr = df_fix.values.astype(float)
    locales = fix_arr[fix_arr[:,0] == 1, 1].tolist()
    visitas = fix_arr[fix_arr[:,0] == 1, 2].tolist()
    e_locales = fix_arr[fix_arr[:,0] == 1, 5].tolist()
    e_visitas = fix_arr[fix_arr[:,0] == 1, 6].tolist()
    dict_sm = [dict(zip(locales + visitas, e_locales + e_visitas))]
    fix_arr[fix_arr[:,0] == 1] = SimularRondaRatioELO(fix_arr[fix_arr[:,0] == 1],  sm = trace['lre'][draw])
    # Actualizar funcion de ánimo
    dict_sm = ActualizarAnimoRatioELO(fix_arr[fix_arr[:,0] == 1], *dict_sm)
    fix_arr[fix_arr[:,0] == 1] = SimularRondaRatioELO(fix_arr[fix_arr[:,0] == 1], sm = trace['lre'][draw])
    # Actualizar funcion de ánimo
    dict_sm = ActualizarAnimoSimple(fix_arr[fix_arr[:,0] == 1], *dict_sm)        
    for r in rondas[1:]:
        fix_arr[fix_arr[:,0] == r, 5] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 1]])
        fix_arr[fix_arr[:,0] == r, 6] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 2]])
        fix_arr[fix_arr[:,0] == r] = SimularRondaRatioELO(fix_arr[fix_arr[:,0] == r], sm = trace['lre'][draw])
        # Actualizar funcion de ánimo
        dict_sm = ActualizarAnimoRatioELO(fix_arr[fix_arr[:,0] == r], *dict_sm)
    return pd.DataFrame(fix_arr[:, [0,1,2,7,8]], columns = ['Round','Local','Visita','goles L', 'goles V'])

def SimularMCMCRatioELO(df_cal, df_fix, sm, outputdir, n_sim = 'test'):
#     ti = time()
    # Filtrar torneo calibracion y fixture a simular
    rondas = df_fix['Round'].drop_duplicates().tolist()
    # Reindexar según los equipos participantes
    df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
    df_cal = df_cal[['Round','i_local','i_visita','goles L','goles V', 'elo_local', 'elo_visita']]
    ids_torneo = df_fix['i_local'].drop_duplicates().tolist()
    # Calibración con datos históricos
    trace = sm.sampling(data=DataFrameToStanData(df_cal), iter=1000, chains=2, n_jobs = 1)
    df_sim = [SimularTorneoRatioELO(df_fix, rondas, ids_torneo, trace) for n in range(1000)]
    df_sim = pd.concat(df_sim, ignore_index = True)
    df_sim['Local'] = [dictindex[int(i)] for i in df_sim['Local'].tolist()]
    df_sim['Visita'] = [dictindex[int(i)] for i in df_sim['Visita'].tolist()]
    return df_sim

def EjecutarValidacionMCMC(datadir, outputdir, archivo, nombres, modelos, funciones, t_vals):
    df_data = pd.read_excel(os.path.join(datadir, archivo))
    torneos = df_data['Torneo'].drop_duplicates().tolist()
    archivo_partidos = "Partidos Simulacion mcmc Funcion %s %s torneos Validacion %s.csv"
    for t_val in t_vals:
        torneos_cal = torneos[:torneos.index(t_val)]
        df_fix = df_data[df_data['Torneo'] == t_val].reset_index(drop=True)
        for j in range(len(torneos_cal)):
            df_cal = df_data[df_data['Torneo'].isin(torneos_cal[j:])]
            df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
            n_cal = len(torneos_cal) - j
            for i in range(len(nombres)):
                try:
                    nombre = nombres[i]
                    modelo = modelos[i]
                    rep = (nombre, n_cal, t_val.replace('/','-'))
                    funcion = funciones[i]
                    sm = pickle.load(open(modelo, 'rb'))
                    print('Simulando %s con %s torneos para validacion %s' % rep)
                    df_sim = funcion(df_cal, df_fix, sm, outputdir)
                    df_sim.to_csv(os.path.join(outputdir, archivo_partidos % rep), index = False)
                except:
                    print('Error en %s con %s torneos para validacion %s' % rep)

#def TestValidacionMCMC(datadir, outputdir, archivo, nombre, modelo, funcion, t_val = 'Premier League 2017/2018', n_cal = 1):
#    df_cal = pd.read_excel(os.path.join(datadir,archivo))
#    torneos = df_cal['Torneo'].drop_duplicates().tolist()
#    torneos_cal = torneos[:torneos.index(t_val)]
#    df_fix = df_cal[df_cal['Torneo'] == t_val].reset_index(drop=True)
#    df_cal = df_cal[df_cal['Torneo'].isin(torneos_cal)].reset_index(drop=True)    
#    df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
#    archivo_partidos = "Partidos Simulacion mcmc Funcion %s %s torneos Validacion %s.csv"
#    sm = pickle.load(open(modelo, 'rb'))
#    rep = (nombre, n_cal, t_val.replace('/','-'))
#    print('Simulando %s con %s torneos para validacion %s' % rep)
#    df_sim = funcion(df_cal, df_fix, sm, outputdir)
#    df_sim.to_csv(os.path.join(outputdir, archivo_partidos % rep), index = False)  
    
"""
EJECUCIÓN
"""
#datadir = os.path.join(os.path.pardir, 'Datos', 'Simulacion','Calibracion')
#outputdir = os.path.join(os.path.pardir, 'Datos', 'Resultados','Validacion simulaciones')
datadir = os.path.join(os.path.pardir, 'datos') 
outputdir = os.path.join(os.path.pardir,'simulacion')
archivo = 'Inglaterra.xlsx'
#nombres = ['Simple', 'LV', 'PromGoles', 'PromGolesLV', 'RatioELO']
#modelos = ['simple.pkl', 'lv.pkl', 'promgoles.pkl','promgoleslv.pkl','ratioelo.pkl']
#t_vals = ['Premier League 2016/2017', 'Premier League 2017/2018']
#funciones = [SimularMCMCSimple, SimularMCMCLV, SimularMCMCPromGoles, SimularMCMCPromGolesLV, SimularMCMCRatioELO]
t_val = 'Premier League 2017/2018'
try:
    n_cal = int(sys.argv[1])
except:
    n_cal = 1
    
df_cal = pd.read_excel(os.path.join(datadir,archivo))
torneos = df_cal['Torneo'].drop_duplicates().tolist()
torneos_cal = torneos[:torneos.index(t_val)][-n_cal:]
df_fix = df_cal[df_cal['Torneo'] == t_val].reset_index(drop=True)
df_cal = df_cal[df_cal['Torneo'].isin(torneos_cal)].reset_index(drop=True)    
df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
archivo_partidos = "Partidos Simulacion mcmc Funcion %s %s torneos Validacion %s.csv"


sm = pickle.load(open('promgoleslv.pkl', 'rb'))
nombre = 'PromGolesLV'
rep = (nombre, n_cal, t_val.replace('/','-'))
print('Simulando %s con %s torneos para validacion %s' % rep)
df_sim = SimularMCMCPromGolesLV(df_cal, df_fix, sm, outputdir)


df_sim.to_csv(os.path.join(outputdir, archivo_partidos % rep), index = False)  
#archivo_partidos = "Partidos Simulacion mcmc Funcion %s %s torneos Validacion %s.csv"

#TestValidacionMCMC(datadir, outputdir, archivo, nombre, modelo, funcion, t_val, n_cal)
#EjecutarValidacionMCMC(datadir, outputdir, archivo, nombres, modelos, funciones, t_vals)



































#def AgregaParametrosIniciales(df_fix, fit, draw):
#    datadict = {}
#    for key in ['intercept','homes','atts','defs']:
#        datadict[key] = fit.extract(permuted = True)[key][draw]
#    df_params = pd.DataFrame(datadict)
#    df_params.index = df_params.index + 1
#    df_params_l = df_params[['intercept','homes','atts','defs']]
#    df_params_l.columns = ['intercept','home','att_local','def_local']
#    df_params_v = df_params[['atts','defs']]
#    df_params_v.columns = ['att_visita','def_visita']
#    df_fix = df_fix.merge(df_params_l, how = 'left', left_on = 'i_local', right_index = True)
#    df_fix = df_fix.merge(df_params_v, how = 'left', left_on = 'i_visita', right_index = True)
#    df_fix = df_fix.fillna(df_fix.mean())
#    df_fix['tasa_local_base'] = np.exp(df_fix['intercept'] + 
#               df_fix['home'] + 
#               df_fix['att_local'] - 
#               df_fix['def_visita'])
#    df_fix['tasa_visita_base'] = np.exp(df_fix['intercept'] + 
#               df_fix['att_visita'] - 
#               df_fix['def_local'])
#    df_fix = df_fix[['Round','i_local','i_visita','tasa_local_base','tasa_visita_base']]    
#    return df_fix
#
#def ActualizarParametrosBase(trace, draw,arr_ronda):
#    tlb = []
#    tvb = []
#    for m in range(arr_ronda.shape[0]):
#        i = int(arr_ronda[m,1])-1
#        j = int(arr_ronda[m,2])-1
#        tlb.append(np.exp(trace['intercept'][draw] + 
#                          trace['homes'][draw][i] + 
#                          trace['atts'][draw][i] - 
#                          trace['defs'][draw][j]))
#        tvb.append(np.exp(trace['intercept'][draw] + 
#                          trace['atts'][draw][j] - 
#                          trace['defs'][draw][i]))
#    arr_ronda[:,3] = np.asarray(tlb)
#    arr_ronda[:,4] = np.asarray(tvb)
#    return arr_ronda

#def ActualizarAnimo(arr_ronda_a, funcion_actualizar_animo, dicts):
#    return funcion_actualizar_animo(arr_ronda_a, *dicts) 