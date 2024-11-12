"""
Ejemplos de calculos de metricas - practica 4
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from climpred import HindcastEnsemble

from funciones_practicas import (PlotContourf_SA, CrossAnomaly_1y,
                                 Calibracion_MediaSD, MLR, Compute_MLR_CV,
                                 MLR_pronostico, CCA_mod_CV,
                                 CCA_training_testing,
                                 CCA_calibracion_CV,
                                 CCA_calibracion_training_testing,
                                 SetPronos_Climpred, BSS, RPSS, ROC, REL,
                                 PlotROC, PlotRelDiag, PlotPcolormesh_SA)
# ---------------------------------------------------------------------------- #
# Apertura y seteo de variables
# ---------------------------------------------------------------------------- #
ruta = '~/PronoClim/obs_seteadas/'

# PP - SON
ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')
ds2 = ds2.rolling(time=3, center=True).mean('time')
ds2 = ds2.sel(time=ds2.time.dt.year.isin(np.arange(1983,2021)))
pp_son = ds2.sel(time=ds2.time.dt.month.isin(10))

# SLP - ASO
ds3 = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')
ds3 = ds3.rolling(time=3, center=True).mean('time')
ds3 = ds3.sel(time=ds3.time.dt.year.isin(np.arange(1983,2021)))
ds3 = ds3.sel(lon=slice(210,320), lat=slice(-20,-75))
slp_aso = ds3.sel(time=ds3.time.dt.month.isin(9))

mod_gem =  (xr.open_dataset(f'~/PronoClim/modelos_seteados/'
                            f'prec_CMC-GEM5-NEMO_SON.nc')*91/3) # mm/mes

# indices
n34 = xr.open_dataset('~/PronoClim/indices_nc/nino34_mmean.nc')
n34 = n34 - n34.mean('time') # este no está sin anomalias
dmi = xr.open_dataset('~/PronoClim/indices_nc/iod_mmean.nc')
sam = xr.open_dataset('~/PronoClim/indices_nc/sam_mmean.nc')

# Selección de fechas que me importan.
# En este caso (arbitrario) mismo mes que los pronosticos.
# Agosto --> SON.
dmi = dmi.sel(time=dmi.time.dt.year.isin(mod_gem.time.dt.year))
dmi = dmi.sel(time=dmi.time.dt.month.isin(9)) # septiembre
n34 = n34.sel(time=n34.time.dt.year.isin(mod_gem.time.dt.year))
n34 = n34.sel(time=n34.time.dt.month.isin(8)) # agosto
sam = sam.sel(time=sam.time.dt.year.isin(mod_gem.time.dt.year))
sam = sam.sel(time=sam.time.dt.month.isin(7)) # julio

prec_cv_anom = CrossAnomaly_1y(pp_son, norm=True)

################################################################################
# ---------------------------------------------------------------------------- #
# Los siguientes son EJEMPLOS para mostrar como se usan las funciones y el
# calculo de las metricas.
################################################################################
# Ejemplo 1: ----------------------------------------------------------------- #
# Pronostico Calibrado
mod_gem_calibrado = Calibracion_MediaSD(mod=mod_gem, obs=pp_son)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Ejemplo 2a: ---------------------------------------------------------------- #
# Pronostico MLR con validacion cruzada
predictores = [n34/n34.std(), dmi/dmi.std(), sam/sam.std()]
mlr = MLR(predictores)

# Tarda aprox 5min
regre_result_prec_cv, mod_prec_years_out, predictores_years_out = (
    Compute_MLR_CV(prec_cv_anom, predictores, window_years=3, intercept=True))

# reconstruimos el pronostico y recortamos el periodo de prec_cv_anom
# ya que al hacer CV perdemos algunos años de los extremos
prec_cv_anom_to_verif_cv, mlr_cv_forecast = (
    MLR_pronostico(data=prec_cv_anom, regre_result=regre_result_prec_cv,
                   predictores=predictores_years_out))

# Ejemplo2b: ----------------------------------------------------------------- #
# Pronostico MLR con training y testing
testing_times = 10 # ultimos tiempos
training = prec_cv_anom.sel(time=prec_cv_anom.time.values[:-testing_times])
testing = prec_cv_anom.sel(time=prec_cv_anom.time.values[-testing_times:])
testing_times_values = testing.time.values

# predictores
predictores_training = []
predictores_testing = []
for p in predictores:
    # para evitar problemas con distintos meses...
    aux_p = p.sel(time=p.time.values[:-testing_times])
    predictores_training.append(aux_p)
    aux_p_testing = p.sel(time=p.time.values[-testing_times:])
    aux_p_testing['time'] = testing_times_values
    predictores_testing.append(aux_p_testing)

# MLR en trainging solamente
mlr = MLR(predictores_training)
regremodel = mlr.compute_MLR(training.prec)

# idem que antes
prec_cv_anom_to_verif_tt, mlr_tt_forecast = (
    MLR_pronostico(data=prec_cv_anom, regre_result=regremodel,
                   predictores=predictores_testing))

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Ejemplo 3a: ---------------------------------------------------------------- #
# Pronostico CCA con validacion cruzada
pp_cca_forecast_cv, pp_cca_to_verif_cv  = CCA_mod_CV(X=slp_aso, Y=pp_son,
                                                     var_exp=0.7,
                                                     window_years=3)

# Ejemplo 3b: ---------------------------------------------------------------- #
# Pronostico CCA con training y testing
pp_cca_forecast_tt, pp_cca_to_verif_tt = (
    CCA_training_testing(X=slp_aso, Y=pp_son, var_exp=0.7,
                         anios_training=[1983, 2011],
                         anios_testing=[2011, 2020],
                         plus_anio=0))

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Ejemplo 4a: ---------------------------------------------------------------- #
# Calibracion CCA con validacion cruzada
mod_gem_calibrado_cca_cv, pp_cca_verif_cal_cv = (
    CCA_calibracion_CV(X_modelo_full=mod_gem,
                       Y=pp_son,
                       var_exp=0.7,
                       window_years=3))

# Ejemplo 4b: ---------------------------------------------------------------- #
# Calibracion CCA con training y testing
mod_gem_calibrado_cca_tt, pp_cca_verif_cal_tt = (
    CCA_calibracion_training_testing(mod_gem, pp_son, var_exp=0.7,
                                     anios_training=[1984, 2011],
                                     anios_testing=[2011, 2020]))

# ---------------------------------------------------------------------------- #
################################################################################
# Metricas de verificación
################################################################################
# Las metricas para pronosticos deterministicos siguen todas el mismo formato
# con climpred

# PROBAR reemplazando "pronostico" y "verificacion" con los ejemplos de arriba.
pronostico = mod_gem_calibrado
verificacion = pp_son

# Ordenamos los datos para que climpred entienda...
pronostico_set = SetPronos_Climpred(pronostico, verificacion)

# incializamos HindcastEnsemble de con la variable del PRONOSTICO seteada
hindcast = HindcastEnsemble(pronostico_set)
hindcast = hindcast.add_observations(verificacion) # agregamos observaciones

# A modo de EJEMPLO para no llenar el codigo de lineas casi identicas:
for metrica in ['mae', 'rmse', 'bias', 'spearman_r']:
    result = hindcast.verify(metric=metrica,
                             comparison='e2o', dim='init',
                             alignment='maximize')

    PlotContourf_SA(data=result, data_var=result.prec,
                    scale=np.linspace(result.prec.min(), result.prec.max(), 13),
                    cmap='Spectral_r', title=metrica)

# ---------------------------------------------------------------------------- #
# Metricas Probabilisticos
# ---------------------------------------------------------------------------- #
# PROBAR reemplazando "pronostico" y "verificacion" modelo con y sin calibrar
# tambien ejemplo 4a y 4b.

pronostico = mod_gem_calibrado
verificacion = pp_son
calibrado = True # IMPORTANTE! Determina como se calculan los quantiles!

# Se pueden seleccionar los tiempos que se quieran evaluar
# Recomendación: usar fechas de la variable del pronostico para evitar
# problemas de formato
fechas_pronostico = pronostico.time.values

# BSS ------------------------------------------------------------------------ #
bss_forecast = BSS(pronostico, verificacion, fechas_pronostico, calibrado)

PlotContourf_SA(data=bss_forecast, data_var=bss_forecast.BSS,
                scale=np.arange(-0.4, 1.2, 0.2),
                cmap='Spectral_r', title='BSS')

# puede ser mas util graficar con esta funcion.
# es menos estetica pero no interpola los valores entre puntos de grilla
# en esta funcion scale solo va usar los extremos max y min
PlotPcolormesh_SA(data=bss_forecast, data_var=bss_forecast.BSS,
                  scale=np.arange(-0.4, 1.2, 0.2),
                  cmap='Spectral_r', title='BSS')

# RPSS ----------------------------------------------------------------------- #
rpss_forecast = RPSS(pronostico, verificacion, fechas_pronostico, calibrado)

PlotContourf_SA(data=rpss_forecast, data_var=rpss_forecast.RPSS,
                scale=np.arange(-0.4, 1.2, 0.2),
                cmap='Spectral_r', title='RPSS')

# ROC ------------------------------------------------------------------------ #
c_roc = ROC(pronostico, verificacion, fechas_pronostico, calibrado)

PlotROC(c_roc)

# reliability diagram -------------------------------------------------------- #
c_rel, hist_above, hist_below = REL(pronostico, verificacion, fechas_pronostico,
                                    calibrado)

PlotRelDiag(c_rel, hist_above, hist_below)
# ---------------------------------------------------------------------------- #
################################################################################