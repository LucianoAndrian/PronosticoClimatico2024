"""
Ejemplos de MLR.
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np

from funciones_practicas import (PlotContourf_SA, MLR, Compute_MLR_CV, ACC,
                                 MLR_pronostico, Compute_MLR,
                                 Compute_MLR_training_testing, CrossAnomaly_1y)
# ---------------------------------------------------------------------------- #
# precipitacion observada
prec = xr.open_dataset('~/PronoClim/obs_seteadas/prec_monthly_nmme_cpc_sa.nc')

# indices
n34 = xr.open_dataset('~/PronoClim/indices_nc/nino34_mmean.nc')
n34 = n34 - n34.mean('time') # este no está sin anomalias
dmi = xr.open_dataset('~/PronoClim/indices_nc/iod_mmean.nc')
sam = xr.open_dataset('~/PronoClim/indices_nc/sam_mmean.nc')

################################################################################
# Ejemplos:
# Suponiendo que queremos hacer una regresión lineal multiple usando los
# indices nino34, dmi y sam: pp = b1*n34 + b2*dmi + b3*sam + c
prec_regre, prec_regre_anoms = Compute_MLR(predictando=prec, mes_predictando=10,
                                           predictores = [n34, dmi, sam],
                                           meses_predictores = [8, 8, 8],
                                           predictando_trimestral = True,
                                           predictores_trimestral=False)
print(prec_regre.dims)

# La dimensión "coef" contiene los coeficientes del modelo de regresión para
# CADA PUNTO DE RETICULA
print(prec_regre.coef)
# En este caso:
# -prec_1: ordenada al origen
# -nino34_1: el coeficiente de regresión asociadao al indice dmi. (b1*n34)
# -dmi_2: el coeficiente de regresión asociadao al indice dmi. (b2*dmi)
# -sam_3: etc...
# Si los predictores son solo listas los coeficientes seran "coef_1", "coef_2"

# Ploteamos los valores del coeficiente asociado al n34. (probar los otros)
PlotContourf_SA(prec_regre,
                prec_regre.sel(coef='nino34_1').prec,
                scale=np.arange(0,180, 20),
                cmap='Blues', title='Coef. Niño3.4 - PP SON',
                mask_andes=True, mask_land=False)

PlotContourf_SA(prec_regre_anoms,
                prec_regre_anoms.sel(coef='nino34_1'),
                scale=np.arange(-1,1.2,0.2),
                cmap='BrBG', title='Coef. Niño3.4 - PP anomalía SON',
                mask_andes=True, mask_land=False)

# ---------------------------------------------------------------------------- #
# Ejemplo con pronostico en un año diferente
# DMI en octubre, N34 en noviembre, SAM en enero y pp en FMA

anios_predictores = [np.arange(1982,2020), # un año antes que prec
                     np.arange(1982,2020), # un año antes que prec
                     prec.time.dt.year]

prec_regre, prec_regre_anoms = (
    Compute_MLR(predictando=prec, mes_predictando=3, # Marzo
                predictores = [n34, dmi, sam],
                meses_predictores = [10, 11, 1], # meses de cada indice
                predictando_trimestral = True, # fMa
                predictores_trimestral=False,
                anios_predictores=anios_predictores)) # <====
################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo usando los coeficientes en
# un ejemplo con periodos de training y testing
# ---------------------------------------------------------------------------- #
prec_regre, prec_regre_anoms, reconstruc  = (
    Compute_MLR_training_testing(predictando=prec, mes_predictando=10,
                                 predictores=[n34, dmi, sam],
                                 meses_predictores=[8,8,8],
                                 anios_training=[1983, 2000], # Training
                                 anios_testing=[2001, 2020], # Testing
                                 predictando_trimestral = True,
                                 predictores_trimestral=False,
                                 anios_predictores_testing = None,
                                 anios_predictores_training = None,
                                 reconstruct_full = False))
# comentar que hace

# Ejemplo graficar un año de los reconstruidos
PlotContourf_SA(data=reconstruc,
                data_var=reconstruc.sel(time='2015-10-01').squeeze().prec,
                scale=np.arange(0, 180, 20),
                cmap='Blues',
                title='MLR Forecast',
                mask_andes=True, mask_land=False)

from funciones_practicas import SingleAnomalia_CV
pp_mlr_2015 = SingleAnomalia_CV(reconstruc, 2015)
PlotContourf_SA(data=pp_mlr_2015,
                data_var=pp_mlr_2015,
                scale=np.arange(-50, 55, 5),
                cmap='BrBG',
                title='MLR Forecast',
                mask_andes=True, mask_land=False)

# Anomalías observadas
aux = prec.rolling(time=3, center=True).mean()
aux = aux.sel(time=aux.time.dt.month.isin(10))
pp_obs_2015 = SingleAnomalia_CV(aux, 2015)
del aux
PlotContourf_SA(data=pp_obs_2015,
                data_var=pp_obs_2015,
                scale=np.arange(-50, 55, 5),
                cmap='BrBG',
                title='observado',
                mask_andes=True, mask_land=False)

# Si usamos reconstruct_full = True
# Va a usar training y reconstruir testing pero ademas va usar testing y
# luego reconstruir training. Asi reconstruye el periodo completo

# ademas, en anios_predictores_testing/training podemos especificar diferentes
# años para cada predictor, al igual que con Compute_MLR
# DEBE SER CONSISTENTE ENTRE meses_predictores, anios_training y anios_testing

prec_regre, prec_regre_anoms, reconstruc,  = \
    Compute_MLR_training_testing(predictando=prec, mes_predictando=3,
                                 predictores=[n34, dmi, sam],
                                 meses_predictores=[10,11,1],
                                 anios_training=[1983, 2000], # Training
                                 anios_testing=[2001, 2020], # Testing
                                 predictando_trimestral = True,
                                 predictores_trimestral=False,
                                 anios_predictores_testing = [[2000,2019],
                                                              [2000,2019],
                                                              [2001, 2020]],
                                 anios_predictores_training = [[1982,1999],
                                                              [1982,1999],
                                                              [1983, 2000]],
                                 reconstruct_full = True)


# Podemos operar con ellos también: ------------------------------------------ #
# La función requiere que sean xr.DataSet (puede ser facilmente implementado
# dentro de la función)

prec_fma = prec.sel(time=prec.time.dt.month.isin(3))

acc_result = ACC(prec_fma, reconstruc, cvanomaly=True, reference_time_period=[1983,2020])

PlotContourf_SA(acc_result,
                acc_result.prec,
                scale=np.arange(-1, 1.2, 0.2),
                cmap='BrBG', title='ACC - MLR reconstruct - 2011-2020',
                mask_andes=True, mask_land=False)

################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo de MLR con validación cruzada.
# ---------------------------------------------------------------------------- #
# Podemos reconstruir el periodo de testing directamente con MLR_pronostico()
# Ejemplo usando observaciones:

# MODIFICAR ESTO qUE FUNCIONE IGUAL Q LAS ANTERIoRES
regre_result_prec_cv, mod_prec_years_out, predictores_years_out = (
    Compute_MLR_CV(xrds=prec,
                   predictores=[n34, dmi, sam],
                   window_years=3, intercept=True))

prec_crop, prec_recostruct = MLR_pronostico(data=prec,
                                     regre_result=regre_result_prec_cv,
                                     predictores=predictores_years_out)

# Tarda una eternidad...
from funciones_practicas import Compute_MLR_CV_2
regre_result_cv, regre_result_full_cv, years_out_reconstruct_full_cv = \
    Compute_MLR_CV_2(predictando=prec, mes_predictando=10,
                     predictores=[n34, dmi, sam],
                     meses_predictores=[8,8,8],
                     predictando_trimestral=True,
                     predictores_trimestral=False,
                     anios_predictores=None,
                     window_years=3)
################################################################################
################################################################################