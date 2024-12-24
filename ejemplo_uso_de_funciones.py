"""
Ejemplo de como usar las funciones del script funciones_practicas.py
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
cuenta que algunas funciones como ACC_Teorico requiere definir ACC antes
"""
# ---------------------------------------------------------------------------- #
# cambiar rutas segun corresponda
pp_modelo = xr.open_dataset('~/PronoClim/modelos_seteados/prec_CMC-GEM5-NEMO_SON.nc')
pp_obs = xr.open_dataset('~/PronoClim/obs_seteadas/prec_monthly_nmme_cpc_sa.nc')

pp_modelo = pp_modelo*(91/3) # mm/dia --> mm/mes (91 dias el trimestre)

# Señal ---------------------------------------------------------------------- #
pp_model_signal = Signal(pp_modelo)

PlotContourf_SA(data=pp_model_signal, data_var=pp_model_signal.prec,
                scale=np.arange(0, 500), cmap='YlOrRd', title='Señal')

# Enmascarando el oceano y la cordillera
PlotContourf_SA(data=pp_model_signal, data_var=pp_model_signal.prec,
                scale=np.arange(0, 500), cmap='YlOrRd', title='Señal',
                mask_land=True, mask_andes=True)

# Ruido ---------------------------------------------------------------------- #
pp_model_noise = Noise(pp_modelo)

PlotContourf_SA(data=pp_model_noise, data_var=pp_model_noise.prec,
                scale=np.arange(0, 500), cmap='YlGnBu', title='Ruido',
                mask_land=True, mask_andes=True)

# signal-to-noise ratio ------------------------------------------------------ #
stn_ratio = pp_model_signal/pp_model_noise

PlotContourf_SA(data=stn_ratio, data_var=stn_ratio.prec,
                scale=np.arange(0, 1.5, 0.1), cmap='Spectral_r',
                title='Cociente Señal vs. Ruido',
                mask_land=True, mask_andes=True)

# Anomalia cruzada con ventana de un año ------------------------------------- #
pp_obs_anom = CrossAnomaly_1y(pp_obs)

# Anomaly Correlation Coefficient -------------------------------------------- #
# VER docstring, tiene varias formas de uso
help(ACC)

# El modelo tiene una estación por año
# Calculo SON para cada año de los datos observados
aux = pp_obs.rolling(time=3, center=True).mean()
pp_obs_son = aux.sel(time=aux.time.dt.month.isin(10))
del aux # borramos la variable auxiliar

# Las anomalías se toman cruzadas con ventana de un año
acc = ACC(data1=pp_modelo.mean('r'), # media del ensamble
          data2=pp_obs_son,
          cvanomaly=True)

PlotContourf_SA(data=acc.prec, data_var=acc.prec,
                scale=np.arange(0, 1, 0.1), cmap='Reds',
                title='ACC con anomalía cruzada',
                mask_land=True, mask_andes=True)

# Las anomalías se toman en base a un periodo de referencia
acc_rtp = ACC(data1=pp_modelo.mean('r'), # media del ensamble
              data2=pp_obs_son,
              reference_time_period=[2015, 2020])

PlotContourf_SA(data=acc_rtp.prec, data_var=acc_rtp.prec,
                scale=np.arange(0, 1, 0.1), cmap='Reds',
                title='ACC - clim. 2015-2020',
                mask_land=True, mask_andes=True)

# No se toman anomalías, se asume que data1 y data2 son anomalías
pp_mod_anom = pp_modelo - pp_modelo.mean(['r','time'])
pp_obs_son_anom = pp_obs_son - pp_obs_son.mean('time')
# Esto va dar practicamente igual a la anomalia cruzada de un año

acc_ensemble_mean = ACC(data1=pp_mod_anom.mean('r'), # anomalia de la media del ensamble
                        data2=pp_obs_son_anom)

PlotContourf_SA(data=acc_ensemble_mean.prec,
                data_var=acc_ensemble_mean.prec,
                scale=np.arange(0, 1, 0.1),
                cmap='Reds', title='ACC',
                mask_land=True, mask_andes=True)

# ACC teorico ---------------------------------------------------------------- #
acc_t = ACC_Teorico(pp_modelo)

PlotContourf_SA(data=acc_t.prec, data_var=acc_t.prec,
                scale=np.arange(0, 1, 0.1), cmap='Reds',
                title='ACC Teorico - GEM5-NEMO - SON',
                mask_land=True, mask_andes=True)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #