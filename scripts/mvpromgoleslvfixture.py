#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:30:39 2019

@author: cristian
"""
import pandas as pd
import os
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
import statsmodels.formula.api
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson

"""
FUNCIONES BASE
"""

def ReindexDataFrames(df_cal, df_fix):
    eq_cal = df_cal[['Torneo','Local']].drop_duplicates()['Local'].value_counts().index.tolist()
    eq_val = eq_cal + [e for e in df_fix['Local'].drop_duplicates().tolist() if e not in eq_cal]
    dicteq = dict(zip(eq_val,[idx for idx in range(len(eq_val))]))
    dictindex = {v: k for k, v in dicteq.items()}
    df_cal['i_local'] = [dicteq[e] for e in df_cal['Local'].tolist()]
    df_cal['i_visita'] = [dicteq[e] for e in df_cal['Visita'].tolist()]
    df_fix['i_local'] = [dicteq[e] for e in df_fix['Local'].tolist()]
    df_fix['i_visita'] = [dicteq[e] for e in df_fix['Visita'].tolist()]
    return df_cal, df_fix, dictindex

def AgregaParametrosIniciales(df_fix, datadict):
    df_params = pd.DataFrame(datadict)
    df_params_l = df_params[['intercept','homes','atts','defs']]
    df_params_l.columns = ['intercept','home','att_local','def_local']
    df_params_v = df_params[['atts','defs']]
    df_params_v.columns = ['att_visita','def_visita']
    df_fix = df_fix.merge(df_params_l, how = 'left', left_on = 'i_local', right_index = True)
    df_fix = df_fix.merge(df_params_v, how = 'left', left_on = 'i_visita', right_index = True)
    df_fix = df_fix.fillna(df_fix.mean())
    # Es con '+' porque statsmodels lo estima como regresión
    df_fix['tasa_local_base'] = np.exp(df_fix['intercept'] + 
               df_fix['home'] + 
               df_fix['att_local'] + 
               df_fix['def_visita'])
    df_fix['tasa_visita_base'] = np.exp(df_fix['intercept'] + 
               df_fix['att_visita'] + 
               df_fix['def_local'])
    df_fix = df_fix[['Round','i_local','i_visita','tasa_local_base','tasa_visita_base']]    
    return df_fix


def ActualizarParametrosBase(trace, arr_ronda):
    tlb = []
    tvb = []
    for m in range(arr_ronda.shape[0]):
        i = int(arr_ronda[m,1])
        j = int(arr_ronda[m,2])
        # Es con '+' porque statsmodels lo estima como regresión
        tlb.append(np.exp(trace['intercept'] + 
                          trace['homes'][i] + 
                          trace['atts'][i] + 
                          trace['defs'][j]))
        tvb.append(np.exp(trace['intercept'] + 
                          trace['atts'][j] + 
                          trace['defs'][i]))
    arr_ronda[:,3] = np.asarray(tlb)
    arr_ronda[:,4] = np.asarray(tvb)
    return arr_ronda

def OutputPoissReg(model, variables, ids_torneo):
    params = pd.Series(model.params.values, index = model.params.index.str.replace(':atts','s'))
    dictparams = dict()
    dictparams['intercept'] = params['Intercept']
    var_base = ['atts','defs','homes']
    for var in var_base:
        arr = np.zeros(int(max(ids_torneo)) + 1)
        for i in ids_torneo:
            try:
                arr[i] = params['%s_%s' % (var,int(i))]
            except:
                pass
        dictparams[var] = arr
    for var in variables:
        dictparams[var] = params[var]
    return dictparams

"""
FUNCIONES PARA MODELO PROM GOLES LV
"""


def ReshapeDataFramePromGolesLV(df_cal):
    df_local = df_cal[['goles L', 'i_local', 'i_visita', 'pgfl_local', 'pgfv_visita', 'pgal_local','pgav_visita']]
    df_local['home'] = 1
    df_local['pgfv_visita'] = 0
    df_local['pgal_local'] = 0
    df_local.columns = ['goles', 'atts', 'defs', 'pgfl', 'pgfv', 'pgal', 'pgav', 'home']
    df_visita = df_cal[['goles V', 'i_visita', 'i_local', 'pgfl_local', 'pgfv_visita', 'pgal_local','pgav_visita']]
    df_visita['home'] = 0
    df_visita['pgfl_local'] = 0
    df_visita['pgav_visita'] = 0
    df_visita.columns = ['goles', 'atts', 'defs', 'pgfl', 'pgfv', 'pgal', 'pgav', 'home']
    df_reg = pd.concat([df_local,df_visita], ignore_index = True)
    df_reg = df_reg[['goles', 'pgfl', 'pgfv', 'pgal', 'pgav', 'home']].join(pd.get_dummies(df_reg[['atts','defs']], columns = ['atts','defs']))
    cols_atts = [i for i in df_reg.columns.tolist() if 'atts' in i]
    for col in cols_atts:
        df_reg[col.replace('atts','homes')] = df_reg['home']*df_reg[col]
    df_reg = df_reg.loc[:, (df_reg != 0).any(axis=0)]
    return df_reg

def FormulaPromGolesLV(df_reg_columns):
    formula = "goles ~ %s + %s + %s + pgfl + pgfv + pgal + pgav"
    atts_str = " + ".join([i for i in df_reg_columns if 'atts_' in i])
    defs_str = " + ".join([i for i in df_reg_columns if 'defs_' in i])
    homes_str = " + ".join([i for i in df_reg_columns if 'homes_' in i])
    formula = formula % (atts_str, defs_str, homes_str)
    constraints = [atts_str, defs_str, homes_str]
    return formula, constraints


def EstimacionMVPromGolesLV(df_cal, ids_torneo):
    df_reg = ReshapeDataFramePromGolesLV(df_cal)
    formula, constraints = FormulaPromGolesLV(df_reg.columns.tolist())
    model = glm(formula,
                groups = None,
                data = df_reg,
                family=Poisson()).fit_constrained(constraints)
    dictparams = OutputPoissReg(model, ['pgfl','pgfv','pgal', 'pgav'], ids_torneo)
    return dictparams


def SimularRondaPromGolesLV(arr_ronda, pgfl, pgfv, pgal, pgav):
    # 0 a 4: columnas base
    # 5, 6: pgfl local, pgfv visita
    # 7, 8: pgal local, pgav visita
    n_matches = arr_ronda.shape[0]
    tasa_local = arr_ronda[:,3] * np.exp(pgfl*arr_ronda[:,5] + pgav*arr_ronda[:,8])
    tasa_visita = arr_ronda[:,4] * np.exp(pgfv*arr_ronda[:,6] + pgal*arr_ronda[:,7])
    arr_ronda[:,9] = np.random.poisson(lam = tasa_local, size = (1,n_matches))
    arr_ronda[:,10] = np.random.poisson(lam = tasa_visita, size = (1,n_matches))
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

def ActualizarAnimo(arr_ronda_a, funcion_actualizar_animo, dicts):
    return funcion_actualizar_animo(arr_ronda_a, *dicts) 

def ActualizarDataCalPromGolesLV(df_cal, arr_ronda):
    #Columnas necesarias: i_local, i_visita, sm_local, sm_visita, goles L, goles V
    ronda = np.max(arr_ronda[:,0])
    df_cal = df_cal[~(df_cal['Round'] == ronda)][['Round','i_local','i_visita','pgfl_local','pgfv_visita','pgal_local','pgav_visita' , 'goles L', 'goles V']].reset_index(drop=True)
    df_ronda = pd.DataFrame({'Round': arr_ronda[:,0], 
                             'i_local': arr_ronda[:,1],
                             'i_visita': arr_ronda[:,2],
                             'pgfl_local': arr_ronda[:,5],
                             'pgfv_visita': arr_ronda[:,6],
                             'pgal_local': arr_ronda[:,7],
                             'pgav_visita': arr_ronda[:,8],
                             'goles L': arr_ronda[:,9],
                             'goles V': arr_ronda[:,10]})
    df_cal = pd.concat([df_cal, df_ronda], ignore_index = True).reset_index(drop=True)
    df_cal['i_local'] = df_cal['i_local'].astype(int)
    df_cal['i_visita'] = df_cal['i_visita'].astype(int)
    df_cal['goles L'] = df_cal['goles L'].astype(int)
    df_cal['goles V'] = df_cal['goles V'].astype(int)
    return df_cal

def SimularTorneoPromGolesLV(fix_arr_or, dict_sm, rondas, trace):
    fix_arr = fix_arr_or
    fix_arr[fix_arr[:,0] == 1] = SimularRondaPromGolesLV(fix_arr[fix_arr[:,0] == 1],pgfl = trace['pgfl'], pgfv = trace['pgfv'], pgal = trace['pgal'], pgav = trace['pgav'])
    # Actualizar funcion de ánimo
    dict_sm = ActualizarAnimoPromGolesLV(fix_arr[fix_arr[:,0] == 1], *dict_sm)
    for r in rondas[1:]:
        fix_arr[fix_arr[:,0] == r, 5] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 1]])
        fix_arr[fix_arr[:,0] == r, 6] = np.asarray([dict_sm[0][int(i)] for i in fix_arr[fix_arr[:,0] == r, 2]])
        fix_arr[fix_arr[:,0] == r] = SimularRondaPromGolesLV(ActualizarParametrosBase(trace,fix_arr[fix_arr[:,0] == r]), pgfl = trace['pgfl'], pgfv = trace['pgfv'], pgal = trace['pgal'], pgav = trace['pgav'])
        # Actualizar funcion de ánimo
        dict_sm = ActualizarAnimoPromGolesLV(fix_arr[fix_arr[:,0] == r], *dict_sm)
    return fix_arr[:, [0,1,2,9,10]]

def ObtenerResultadosTorneo(arr_sim):
    # Esencialmente es lo mismo que armar_tabla, solo que recibe arreglos
    # y entrega solo el id del campeón de la simulación
    df = pd.DataFrame(data = arr_sim, columns = ['Round','i_local','i_visita','goles L','goles V'])
    conditions = [
        (df['goles L'] > df['goles V']),
        (df['goles L'] < df['goles V'])]
    choices = ['local', 'visita']
    df = df.join(pd.get_dummies(np.select(conditions, choices, default = 'empate')))
    ghome = df.groupby('i_local')
    gaway = df.groupby('i_visita')
    df_home = ghome.agg({'local': 'sum',
                         'empate':'sum',
                         'visita':'sum',
                         'goles L': 'sum',
                         'goles V': 'sum'})
    df_home.columns = ['wins_h','draws_h','losses_h','gf_h','ga_h']
    df_away = gaway.agg({'visita': 'sum',
                         'empate':'sum',
                         'local':'sum',
                         'goles V': 'sum',
                         'goles L': 'sum'})
    df_away.columns = ['wins_a','draws_a','losses_a','gf_a','ga_a']
    tabla = df_home.join(df_away,how = 'left').fillna(0).reset_index(drop = False)
    tabla.rename(columns = {'i_local':'equipo'}, inplace = True)
    tabla['wins'] = tabla.wins_h + tabla.wins_a
    tabla['draws'] = tabla.draws_h + tabla.draws_a
    tabla['losses'] = tabla.losses_h + tabla.losses_a
    tabla['gf'] = tabla.gf_h +tabla.gf_a
    tabla['ga'] = tabla.ga_h +tabla.ga_a
    tabla['gd'] = tabla['gf']-tabla['ga']
    tabla['points'] = (tabla['wins']*3 + tabla['draws']).astype(int)
    tabla = tabla.sort_values(by=['points','gd'], ascending = False).reset_index(drop=True)
    tabla['position'] = tabla.index + 1
    tabla['ch'] = (tabla.position == 1).astype(int)
    tabla['cla'] = (tabla.position < 5).astype(int)
#    tabla['uel'] = ((tabla.position >=5) & (tabla.position <7)).astype(int)
#    tabla['rel'] = (tabla.position > 17).astype(int)
    return tabla[['equipo','ch','cla']]


def SimularMVPromGolesLV(df_cal, df_fix, outputdir, outname, n_sims = 10000):
#     ti = time()
    # Filtrar torneo calibracion y fixture a simular
    rondas = df_fix['Round'].drop_duplicates().tolist()
    # Reindexar según los equipos participantes
    df_cal, df_fix, dictindex = ReindexDataFrames(df_cal, df_fix)
    df_cal = df_cal[['Round','i_local','i_visita','goles L','goles V', 'pgfl_local', 'pgfv_visita', 'pgal_local','pgav_visita']]
    # Calibración con datos históricos
    ids_torneo = df_fix['i_local'].drop_duplicates().tolist()
    trace = EstimacionMVPromGolesLV(df_cal, ids_torneo)
    df_fix = AgregaParametrosIniciales(df_fix,trace)
    dict_sm = [dict(zip([j for j in ids_torneo],[0 for i in ids_torneo]))]*4
    # Agrega funciones de estado de animo iniciales y columnas de goles
    df_fix['pgf_local'] = 0
    df_fix['pgf_visita'] = 0
    df_fix['pga_local'] = 0
    df_fix['pga_visita'] = 0    
    df_fix['goles L'] = 0
    df_fix['goles V'] = 0
    fix_arr = df_fix.values.astype(float)
    df_sim = pd.concat([ObtenerResultadosTorneo(SimularTorneoPromGolesLV(fix_arr, dict_sm, rondas, trace)) for i in range(n_sims)], ignore_index = True)
#    df_sim = df_sim.groupby('equipo').agg({'ch': 'mean', 'ucl':'mean', 'uel' : 'mean', 'rel': 'mean'}).reset_index()
    df_sim = df_sim.groupby('equipo').agg({'ch': 'mean', 'cla':'mean'}).reset_index()
    df_sim['equipo'] = [dictindex[int(i)] for i in df_sim['equipo'].tolist()]
    df_sim.to_csv(os.path.join(outputdir, outname), index = False)

"""EJECUCIÓN"""
#datadir = os.path.join(os.path.pardir, 'Datos','Simulacion','Calibracion')
#fixdir = os.path.join(os.path.pardir, 'Datos','Simulacion','Fixtures')
#outputdir = os.path.join(os.path.pardir,'Resultados')

datadir = os.path.join(os.path.pardir, 'datos','calibracion')
fixdir = os.path.join(os.path.pardir, 'datos','fixtures')
outputdir = os.path.join(os.path.pardir,'resultados','simulacion')


dataname = 'Argentina.xlsx'
fixname = 'arg-primera-division-2019-2020.xlsx'


try:
    n_fix = int(sys.argv[1])
except:
    n_fix = 0

df_cal = pd.read_excel(os.path.join(datadir, dataname))
torneos = df_cal['Torneo'].drop_duplicates().tolist()
df_cal = df_cal[df_cal['Torneo'].isin(torneos[-5:])].reset_index(drop=True)
df_fix = pd.read_excel(os.path.join(fixdir, fixname), sheet_name = 'Fix %s' % n_fix) 
outname = 'argentina-simulacion-promgoleslv-fix-%s.csv' % n_fix
SimularMVPromGolesLV(df_cal, df_fix, outputdir, outname, n_sims = 1000)


