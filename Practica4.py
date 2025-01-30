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
                                 PlotROC, PlotRelDiag, PlotPcolormesh_SA,
                                 Compute_MLR_training_testing)
# ---------------------------------------------------------------------------- #
# Apertura y seteo de variables
# ---------------------------------------------------------------------------- #
ruta = '~/PronoClim/obs_seteadas/'

# Prec
prec = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')

# SLP - ASO
slp = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')
slp = slp.sel(lon=slice(210,320), lat=slice(-20,-75))

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
#dmi = dmi.sel(time=dmi.time.dt.month.isin(9)) # septiembre
n34 = n34.sel(time=n34.time.dt.year.isin(mod_gem.time.dt.year))
#n34 = n34.sel(time=n34.time.dt.month.isin(8)) # agosto
sam = sam.sel(time=sam.time.dt.year.isin(mod_gem.time.dt.year))
#sam = sam.sel(time=sam.time.dt.month.isin(7)) # julio

#prec_cv_anom = CrossAnomaly_1y(pp_son, norm=True)

################################################################################
# ---------------------------------------------------------------------------- #
# Los siguientes son EJEMPLOS para mostrar como se usan las funciones y el
# calculo de las metricas.
################################################################################
# Ejemplo 1: ----------------------------------------------------------------- #
# Pronostico Calibrado
mod_gem_calibrado_MediaSD, data_to_verif_MediaSD = Calibracion_MediaSD(
    X_modelo=mod_gem, Y_observacion=prec,
    Y_mes=10, Y_trimestral=True,
    X_anios=[1983, 2020],
    Y_anios=[1983, 2020])

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Ejemplo 2a: ---------------------------------------------------------------- #
# Pronostico MLR con validacion cruzada
_, _, prec_mlr_forecast_cv = Compute_MLR_CV(predictando=prec,
                                            mes_predictando=10,
                                            predictores=[n34, dmi, sam],
                                            meses_predictores=[8, 9, 7],
                                            predictando_trimestral=True,
                                            predictores_trimestral=False,
                                            anios_predictores=None,
                                            window_years=3)

# Ejemplo2b: ----------------------------------------------------------------- #
# Pronostico MLR con training y testing
_, _, prec_mlr_forecast_tt = Compute_MLR_training_testing(
    predictando=prec, mes_predictando=10,
    predictores=[n34, dmi, sam],
    meses_predictores=[8, 9, 7],
    anios_training=[1983, 2000],
    anios_testing=[2001, 2020],
    predictando_trimestral=True,
    predictores_trimestral=False,
    anios_predictores_testing = None,
    anios_predictores_training = None,
    reconstruct_full = True)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Ejemplo 3a: ---------------------------------------------------------------- #
# Pronostico CCA con validacion cruzada
pp_cca_forecast_cv, pp_cca_to_verif_cv  = CCA_mod_CV(
    X=slp, Y=prec, var_exp=0.7, X_mes=7, Y_mes=10,
    X_trimestral=True, Y_trimestral=True,
    X_anios=[1983, 2020], Y_anios=[1983, 2020],
    window_years=3, X_test=None)

# Ejemplo 3b: ---------------------------------------------------------------- #
# Pronostico CCA con training y testing
pp_cca_forecast_tt, pp_cca_to_verif_tt  = CCA_training_testing(
    X=slp, Y=prec, var_exp=0.7,
    X_mes=7, Y_mes=10, X_trimestral=True, Y_trimestral=True,
    X_anios=[1983, 2020], Y_anios=[1983, 2020],
    anios_training=[1983, 2000],  anios_testing=[2001, 2020],
    reconstruct_full=True)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Ejemplo 4a: ---------------------------------------------------------------- #
# Calibracion CCA con validacion cruzada
mod_gem_calibrado_cca_cv, data_to_verif_cal_cca_cv = CCA_calibracion_CV(
    X_modelo=mod_gem, Y_observacion=prec, var_exp=0.7,
    Y_mes=10, Y_trimestral=True, X_anios=[1983, 2020], Y_anios=[1983, 2020],
    window_years=3)

# Ejemplo 4b: ---------------------------------------------------------------- #
# Calibracion CCA con training y testing
mod_gem_calibrado_cca_tt, data_to_verif_cal_cca_tt = (
    CCA_calibracion_training_testing(
        X_modelo=mod_gem, Y_observacion=prec,
        var_exp=0.7, Y_mes=10, Y_trimestral=True,
        X_anios=[1983, 2020], Y_anios=[1983, 2020],
        anios_training=[1983, 2000],  anios_testing=[2001, 2020],
        reconstruct_full=True))

# ---------------------------------------------------------------------------- #
################################################################################
# Metricas de verificación
################################################################################
# Las metricas para pronosticos deterministicos siguen todas el mismo formato
# con climpred

# PROBAR reemplazando "pronostico" y "verificacion" con los ejemplos de arriba.
pronostico = mod_gem_calibrado_cca_cv
verificacion = data_to_verif_cal_cca_cv

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


# RMSE ---> NMRSE


# ---------------------------------------------------------------------------- #
# Metricas Probabilisticos
# ---------------------------------------------------------------------------- #
# PROBAR reemplazando "pronostico" y "verificacion" modelo con y sin calibrar
# tambien ejemplo 4a y 4b.

pronostico = mod_gem_calibrado_MediaSD
verificacion = data_to_verif_MediaSD
calibrado = True # IMPORTANTE! Determina como se calculan los quantiles!

# Se pueden seleccionar los tiempos que se quieran evaluar
# Recomendación: usar fechas de la variable del pronostico para evitar
# problemas de formato
fechas_pronostico = pronostico.time.values

# BSS ------------------------------------------------------------------------ #
bss_forecast = BSS(pronostico, verificacion, fechas_pronostico, calibrado,
                   funcion_prono='Prono_Qt')

PlotContourf_SA(data=bss_forecast, data_var=bss_forecast.BSS,
                scale=np.arange(-0.4, 1.2, 0.2),
                cmap='Spectral_r', title='BSS')

bss_forecast = BSS(pronostico, verificacion, fechas_pronostico, calibrado,
                   funcion_prono='Prono_AjustePDF')

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
rpss_forecast = RPSS(pronostico, verificacion, fechas_pronostico, calibrado,
                   funcion_prono='Prono_Qt')

PlotContourf_SA(data=rpss_forecast, data_var=rpss_forecast.RPSS,
                scale=np.arange(-0.4, 1.2, 0.2),
                cmap='Spectral_r', title='RPSS')


rpss_forecast = RPSS(pronostico, verificacion, fechas_pronostico, calibrado,
                   funcion_prono='Prono_AjustePDF')

PlotContourf_SA(data=rpss_forecast, data_var=rpss_forecast.RPSS,
                scale=np.arange(-0.4, 1.2, 0.2),
                cmap='Spectral_r', title='RPSS')

# ROC ------------------------------------------------------------------------ #
c_roc = ROC(pronostico, verificacion, fechas_pronostico, calibrado,
                   funcion_prono='Prono_Qt')
PlotROC(c_roc)

c_roc = ROC(pronostico, verificacion, fechas_pronostico, calibrado,
                   funcion_prono='Prono_AjustePDF')
PlotROC(c_roc)

# reliability diagram -------------------------------------------------------- #
c_rel, hist_above, hist_below = REL(pronostico, verificacion, fechas_pronostico,
                                    calibrado, funcion_prono='Prono_Qt')
PlotRelDiag(c_rel, hist_above, hist_below)


c_rel, hist_above, hist_below = REL(pronostico, verificacion, fechas_pronostico,
                                    calibrado, funcion_prono='Prono_AjustePDF')
PlotRelDiag(c_rel, hist_above, hist_below)
# ---------------------------------------------------------------------------- #
################################################################################