"""
Ejemplo Practica 3, Calibracion y Pronosticos probabilisticos
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from funciones_practicas import PlotContourf_SA
# ---------------------------------------------------------------------------- #
ruta = '~/PronoClim/obs_seteadas/'
ruta_mod = '~/PronoClim/modelos_seteados/'

mod_gem = xr.open_dataset(ruta_mod + 'prec_CMC-GEM5-NEMO_SON.nc')*(92/3)
prec = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')

################################################################################
# ------------------------------- Calibración -------------------------------- #
################################################################################
from funciones_practicas import CCA_calibracion_training_testing, \
    CCA_calibracion_CV, Calibracion_MediaSD

# Funcion auxiliar solo para mostrar el efecto de la calibracion.
# calcularán esta metricas y otras similares en la practica 4
# Mean Absolute Error
def MAE(data1, data2):

    if 'r' in data1._dims:
        data1 = data1.mean('r')
    if 'r' in data2._dims:
        data2 = data2.mean('r')

    mae = np.abs(data1[list(data1.data_vars)[0]].values -
                 data2[list(data2.data_vars)[0]].values).mean(axis=0)

    return mae
################################################################################
# ---------------------------------------------------------------------------- #
# Calibración con media y desvío
# ---------------------------------------------------------------------------- #
mod_gem_prec_cal, data_to_verif = Calibracion_MediaSD(
    X_modelo=mod_gem, Y_observacion=prec,
    Y_mes=10, Y_trimestral=True,
    X_anios=[1983, 2020],
    Y_anios=[1983, 2020])

mae_sin_c = MAE(mod_gem, data_to_verif)

PlotContourf_SA(mod_gem, mae_sin_c,
                scale=np.arange(0, 180, 20), cmap='YlOrRd',
                title='MAE precipitación - GEM5-NEMO Sin Calibrar')

mae_cal_mean_sd = MAE(mod_gem_prec_cal, data_to_verif)

PlotContourf_SA(mod_gem, mae_cal_mean_sd,
                scale=np.arange(0, 180, 20), cmap='YlOrRd',
                title='MAE precipitación - GEM5-NEMO Calibrado Media-SD')

# ---------------------------------------------------------------------------- #
# Ejemplo de calibracion con CCA usando training y testing
# ---------------------------------------------------------------------------- #
mod_gem_calibrado_cca_tt, data_to_verif_cal_cca_tt = (
    CCA_calibracion_training_testing(
        X_modelo=mod_gem,
        Y_observacion=prec,
        var_exp=0.7,
        Y_mes=10,
        Y_trimestral=True,
        X_anios=[1983, 2020],
        Y_anios=[1983, 2020],
        anios_training=[1983, 2000],  # testing
        anios_testing=[2001, 2020],  # training
        reconstruct_full=True)) # Similar a la practica 2

print(mod_gem_calibrado_cca_tt.dims)
# Obtenemos el modelo calibrado con las mismas dimenciones que el sin calibrar

mae_cal_cca_tt = MAE(mod_gem_calibrado_cca_tt, data_to_verif_cal_cca_tt)

PlotContourf_SA(mod_gem, mae_cal_cca_tt,
                scale=np.arange(0, 180, 20), cmap='YlOrRd',
                title='MAE precipitación - GEM5-NEMO Calibrado CCA-TT')

# ---------------------------------------------------------------------------- #
# Ejemplo de calibracion con CCA usando validacion cruzada
# ---------------------------------------------------------------------------- #
mod_gem_calibrado_cca_cv, data_to_verif_cal_cca_cv = (
    CCA_calibracion_CV(
        X_modelo=mod_gem,
        Y_observacion=prec,
        var_exp=0.7,
        Y_mes=10, Y_trimestral=True,
        X_anios=[1983, 2020],
        Y_anios=[1983, 2020],
        window_years=3))

print(mod_gem_calibrado_cca_cv.dims)
# Igual que el anterior

# Tecnicamente no deberiamos comparar con el primer grafico sin calibrar ya que:
# Al usar CV con ventana de 3 años perdemos los extremos
# (da practicamente igual)
anom_mod_gem = mod_gem.sel(time=slice('1984-08-01', '2019-08-01'))
mae_sin_c2 = MAE(anom_mod_gem, data_to_verif_cal_cca_cv)

PlotContourf_SA(mod_gem, mae_sin_c2,
                scale=np.arange(0, 180, 20), cmap='YlOrRd',
                title='MAE precipitación - GEM5-NEMO Sin Calibrar')

mae_cal_cca_cv = MAE(mod_gem_calibrado_cca_cv, data_to_verif_cal_cca_cv)

PlotContourf_SA(mod_gem, mae_cal_cca_cv,
                scale=np.arange(0, 180, 20), cmap='YlOrRd',
                title='MAE precipitación - GEM5-NEMO Calibrado CCA-CV')

################################################################################
# ----------------------- Pronosticos probabilisticos ------------------------ #
from funciones_practicas import (Prono_Qt, Prono_AjustePDF,
                                 Plot_CategoriaMasProbable)
################################################################################
# ---------------------------------------------------------------------------- #
# 1. Contando la cantidad de miembros de ensamble que caen dentro de las
# categorias definidas por percentiles, en este caso terciles
# ---------------------------------------------------------------------------- #
# PARA EVITAR PROBLEMAS CON LA CODIFICACION:
# para la fecha se recomienda obtenerlas de los valores de "time" de los modelos
print(mod_gem.time.values)

fecha_pronostico = mod_gem.time.values[-6] # 2015-08-01

# Si el modelo ya fue calibrado, debemos dar las observaciones (obs_referencia)
# para que la funcion tome de alli los terciles para comparar
prono_prob_gem = Prono_Qt(modelo=mod_gem_prec_cal,  # calibrado con media y sd. PROBAR CON LOS OTROS
                          fecha_pronostico=fecha_pronostico,
                          obs_referencia=prec) # <-- Si el modelo fue calibrado

print(prono_prob_gem.shape)
print(prono_prob_gem)

# obtenemos un xr.Datarray que para cada punto de reticula tiene un valor de
# probabilidad de 0-1 separado en tres categorias.
# below: por debajo del primer tercil
# normal: entre el 1er y 2do tercil
# above: por encima del 2do tercil

# Podemos graficarlo:
# En cada punto de reticula se grafica la categoria mas probable
Plot_CategoriaMasProbable(data_categorias=prono_prob_gem,
                          variable='prec', # para el color de las categorias
                          titulo=f"Pronostico probabilistico  GEM5-NEMO - SON "
                                 f"{fecha_pronostico.year}")

# Si el modelo no fue calibrado, obs_referencia=None
prono_prob_gem = Prono_Qt(modelo=mod_gem,
                          fecha_pronostico=fecha_pronostico,
                          obs_referencia=None) # <-- Modelo no calibrado

Plot_CategoriaMasProbable(data_categorias=prono_prob_gem,
                          variable='prec',
                          titulo=f"Pronostico probabilistico  GEM5-NEMO - SON "
                                 f"{fecha_pronostico.year}")

# ---------------------------------------------------------------------------- #
# 2. Ajustando cada pronostico a una PDF gaussiana a partir de sus miembros
# de ensamble y comparando con una PDF de referencia del modelo u observada
# Luego se comparan terciles igual que antes.
# ---------------------------------------------------------------------------- #
# Argumentos y salida de la funcion son los mismos que para Prono_Qt()
prono_prob_cm4 = Prono_AjustePDF(modelo=mod_gem_calibrado_cca_tt, # Calibrado con CCA-CV
                                 fecha_pronostico=fecha_pronostico,
                                 obs_referencia=prec)

# En cada punto de reticula se grafica la categoria mas probable
Plot_CategoriaMasProbable(data_categorias=prono_prob_cm4,
                          variable='prec',
                          titulo=f"Pronostico probabilistico CanCM4i-IC3 - SON "
                                 f"{fecha_pronostico.year} \n Ajuste Gaussiano",
                          mask_ocean=True, mask_andes=True)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #