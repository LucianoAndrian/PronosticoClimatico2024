"""
Ejemplo de uso de funciones de calibracion y pronosticos probabilisticos.

"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from funciones_practicas import (Calibracion_MediaSD, Prono_Qt, Prono_AjustePDF,
                                 Plot_CategoriaMasProbable, PlotContourf_SA)
# ---------------------------------------------------------------------------- #
ruta = '~/PronoClim/'

# Pronosticos para SON
mod_cm4_prec =  xr.open_dataset(
    f'{ruta}modelos_seteados/prec_CMC-CanCM4i-IC3_SON.nc')*91/3 # mm/mes)

mod_gem_prec =  xr.open_dataset(
    f'{ruta}modelos_seteados/prec_CMC-GEM5-NEMO_SON.nc')*91/3 # mm/mes


mod_cm4_tref =  xr.open_dataset(
    f'{ruta}modelos_seteados/tref_CMC-CanCM4i-IC3_SON.nc')

mod_gem_tref =  xr.open_dataset(
    f'{ruta}modelos_seteados/tref_CMC-GEM5-NEMO_SON.nc')


# precipitacion observada
prec = xr.open_dataset(f'{ruta}obs_seteadas/prec_monthly_nmme_cpc_sa.nc')
prec = prec.rolling(time=3, center=True).mean()
prec = prec.sel(time=prec.time.dt.month.isin(10)) # SON

# precipitacion observada
tref = xr.open_dataset(f'{ruta}obs_seteadas/tref_monthly_nmme_ghcn_cams_sa.nc')
tref = tref.rolling(time=3, center=True).mean()
tref = tref.sel(time=tref.time.dt.month.isin(10)) # SON

# Calibración ---------------------------------------------------------------- #
mod_gem_prec_cal = Calibracion_MediaSD(mod=mod_gem_prec, obs=prec)
mod_gem_tref_cal = Calibracion_MediaSD(mod=mod_gem_tref, obs=tref)
mod_cm4_prec_cal = Calibracion_MediaSD(mod=mod_cm4_prec, obs=prec)
mod_cm4_tref_cal = Calibracion_MediaSD(mod=mod_cm4_tref, obs=tref)
# Listo.

# Esta funcion es axuliar solo para mostrar el efecto de la calibracion.
# calcularán esta metricas y otras similares en la practica 4
def MAE(data1, data2):

    if 'r' in data1._dims:
        data1 = data1.mean('r')
    if 'r' in data2._dims:
        data2 = data2.mean('r')

    mae = np.abs(data1[list(data1.data_vars)[0]].values -
                 data2[list(data2.data_vars)[0]].values).mean(axis=0)

    return mae

mae_sin_c = MAE(mod_gem_prec, prec)
PlotContourf_SA(mod_gem_prec, mae_sin_c,
                scale=np.arange(0,150,20), cmap='YlOrRd',
                title='MAE precipitación - GEM5-NEMO Sin Calibrar')

mae_c = MAE(mod_gem_prec_cal, prec)
PlotContourf_SA(mod_gem_prec_cal, mae_c,
                scale=np.arange(0,150,20), cmap='YlOrRd',
                title='MAE precipitación - GEM5-NEMO Calibrado')

mae_sin_c = MAE(mod_cm4_prec, prec)
PlotContourf_SA(mod_cm4_prec, mae_sin_c,
                scale=np.arange(0,150,20), cmap='YlOrRd',
                title='MAE precipitación - CanCM4-IC3 Sin Calibrar')

mae_c = MAE(mod_cm4_prec_cal, prec)
PlotContourf_SA(mod_cm4_prec_cal, mae_c,
                scale=np.arange(0,150,20), cmap='YlOrRd',
                title='MAE precipitación - CanCM4-IC3 Calibrado')


# Pronosticos probabilisticos ------------------------------------------------ #
# ---------------------------------------------------------------------------- #
# 1. Contando la cantidad de miembros de ensamble que caen dentro de las
# categorias definidas por percentiles, en este caso terciles
# ---------------------------------------------------------------------------- #
# PARA EVITAR PROBLEMAS CON LA CODIFICACION:
# para la fecha se recomienda obtenerlas de los valores de "time" de los modelos
print(mod_gem_prec.time.values)

fecha_pronostico = mod_gem_prec.time.values[-6] # 2015-08-01

# Si el modelo ya fue calibrado, debemos dar las observaciones para que la
# funcion tome de alli los terciles para comparar
prono_prob_gem = Prono_Qt(modelo=mod_gem_prec_cal,
                          fecha_pronostico=fecha_pronostico,
                          obs_referencia=prec)

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
prono_prob_gem = Prono_Qt(modelo=mod_gem_prec_cal,
                          fecha_pronostico=fecha_pronostico,
                          obs_referencia=None)

Plot_CategoriaMasProbable(data_categorias=prono_prob_gem,
                          variable='prec', # para el color de las categorias
                          titulo=f"Pronostico probabilistico  GEM5-NEMO - SON "
                                 f"{fecha_pronostico.year}")


# Temperatura:
prono_prob_temp_cm4 = Prono_Qt(modelo=mod_cm4_tref_cal,
                               fecha_pronostico=fecha_pronostico,
                               obs_referencia=tref)

Plot_CategoriaMasProbable(data_categorias=prono_prob_temp_cm4,
                          variable='tref', # para el color de las categorias
                          titulo=f"Pronostico probabilistico CanCM4i-IC3 - SON "
                                 f"{fecha_pronostico.year}", mask_land=True)


# ---------------------------------------------------------------------------- #
# 2. Ajustando cada pronostico a una PDF gaussiana a partir de sus miembros
# de ensamble y comparando con una PDF de referencia del modelo u observada
# Luego se comparan terciles igual que antes.
# ---------------------------------------------------------------------------- #

# Argumentos y salida de la funcion son los mismos que para Prono_Qt()

prono_prob_cm4 = Prono_AjustePDF(modelo=mod_cm4_prec_cal,
                                 fecha_pronostico=fecha_pronostico,
                                 obs_referencia=prec)

# En cada punto de reticula se grafica la categoria mas probable
Plot_CategoriaMasProbable(data_categorias=prono_prob_cm4,
                          variable='prec', # para el color de las categorias
                          titulo=f"Pronostico probabilistico CanCM4i-IC3 - SON "
                                 f"{fecha_pronostico.year} \n Ajuste Gaussiano")
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #