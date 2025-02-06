"""
Ejemplos Practica 1.
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
# ---------------------------------------------------------------------------- #
"""
Se pueden importar las funciones desde el script siempre y cuando 
funciones_practicas.py este en el directorio de trabajo
"""
from funciones_practicas import Signal, Noise, CrossAnomaly_1y, ACC, \
    ACC_Teorico, PlotContourf_SA

# Todas las funciones tienen su docstring. Se recomienda mirarlo
help(Signal)
help(ACC)
"""
Tambien se pueden copiar y pegar en el script que se quiera trabajar. Tener en 
cuenta que algunas funciones como ACC_Teorico requiere definir ACC antes.

Por lo que en las practicas siguientes puede ser necesario copiar muchas 
funciones para correr una sola
"""

# ---------------------------------------------------------------------------- #
# cambiar rutas segun corresponda
prec_mod_gem = xr.open_dataset(
    '~/PronoClim/modelos_seteados/prec_CMC-GEM5-NEMO_SON.nc')
prec_obs = xr.open_dataset(
    '~/PronoClim/obs_seteadas/prec_monthly_nmme_cpc_sa.nc')

prec_mod_gem = prec_mod_gem*(91/3) # mm/dia --> mm/mes (91 dias el trimestre)

# Señal ---------------------------------------------------------------------- #
pp_model_signal = Signal(prec_mod_gem)

PlotContourf_SA(data=pp_model_signal, # dataset de donde se toman lon, lat
                data_var=pp_model_signal.prec,  # valores a plotear
                scale=np.arange(0, 500), # escala
                cmap='YlOrRd', # paleta de colores
                title='Señal') # titulo

# Enmascarando el oceano y la cordillera
PlotContourf_SA(data=pp_model_signal,
                data_var=pp_model_signal.prec,
                scale=np.arange(0, 500),
                cmap='YlOrRd',
                title='Señal',
                mask_ocean=True,  # mascara del oceano. PROBAR False
                mask_andes=True) # mascara de la cordillera. PROBAR False

# TODAS las funciones de ploteo permiten guardar la figura generada:
PlotContourf_SA(data=pp_model_signal,
                data_var=pp_model_signal.prec,
                scale=np.arange(0, 500),
                cmap='YlOrRd',
                title='Señal',
                mask_ocean=True,
                mask_andes=True,
                save=True, # <-- guardar
                out_dir = '/home/luciano.andrian/PronoClim/salidas/', # <-- directorio de salida
                name_fig='signal') # <-- nombre del archivo

# Ruido ---------------------------------------------------------------------- #
pp_model_noise = Noise(prec_mod_gem)

PlotContourf_SA(data=pp_model_noise,
                data_var=pp_model_noise.prec,
                scale=np.arange(0, 500),
                cmap='YlGnBu', title='Ruido',
                mask_ocean=True, mask_andes=True)

# signal-to-noise ratio ------------------------------------------------------ #
pp_stn_ratio = pp_model_signal/pp_model_noise

PlotContourf_SA(data=pp_stn_ratio, data_var=pp_stn_ratio.prec,
                scale=np.arange(0, 1.5, 0.1),
                cmap='Spectral_r',
                title='Cociente Señal vs. Ruido',
                mask_ocean=True, mask_andes=True)


# Anomalia cruzada con ventana de un año ------------------------------------- #
pp_obs_anom = CrossAnomaly_1y(prec_obs)

# Anomaly Correlation Coefficient -------------------------------------------- #
# VER docstring, tiene varias formas de uso
help(ACC)

# El modelo tiene una estación por año
# Calculo Sep-Oct-Nov
aux = prec_obs.rolling(time=3, center=True).mean() # promedio movil 3 meses
pp_obs_son = aux.sel(time=aux.time.dt.month.isin(10)) # Octubre ---> s-O-n
del aux # borramos la variable auxiliar

# Las anomalías se toman cruzadas con ventana de un año
acc = ACC(data1=prec_mod_gem.mean('r'), # media del ensamble
          data2=pp_obs_son,
          cvanomaly=True) # anomalia cruzada

PlotContourf_SA(data=acc.prec, data_var=acc.prec,
                scale=np.arange(0, 1, 0.1), cmap='Reds',
                title='ACC con anomalía cruzada',
                mask_ocean=True, mask_andes=True)

# Las anomalías se toman en base a un periodo de referencia
acc_ref_time_period = ACC(data1=prec_mod_gem.mean('r'),
                          data2=pp_obs_son,
                          reference_time_period=[2015, 2020])

PlotContourf_SA(data=acc_ref_time_period, data_var=acc_ref_time_period.prec,
                scale=np.arange(0, 1, 0.1), cmap='Reds',
                title='ACC - clim. 2015-2020',
                mask_ocean=True, mask_andes=True)

# Si no se especifica cvanomaly o reference_time_period,
# se asume que data1 y data2 son anomalías
pp_mod_anom = prec_mod_gem - prec_mod_gem.mean(['r','time'])
pp_obs_son_anom = pp_obs_son - pp_obs_son.mean('time')
# Esto va dar practicamente igual a la anomalia cruzada de un año

acc2 = ACC(data1=pp_mod_anom.mean('r'), # anomalia de la media del ensamble
           data2=pp_obs_son_anom)

PlotContourf_SA(data=acc2,
                data_var=acc2.prec,
                scale=np.arange(0, 1, 0.1),
                cmap='Reds', title='ACC',
                mask_ocean=True, mask_andes=True)


# ACC teorico ---------------------------------------------------------------- #
acc_t = ACC_Teorico(prec_mod_gem)

PlotContourf_SA(data=acc_t, data_var=acc_t.prec,
                scale=np.arange(0, 1, 0.1), cmap='Reds',
                title='ACC Teorico - GEM5-NEMO - SON',
                mask_ocean=True, mask_andes=True)
# ---------------------------------------------------------------------------- #
################################################################################