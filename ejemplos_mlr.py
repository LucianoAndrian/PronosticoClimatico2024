"""
Ejemplos de MLR.
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from funciones_practicas import (PlotContourf_SA, MLR, Compute_MLR_CV, ACC,
                                 CrossAnomaly_1y)
# ---------------------------------------------------------------------------- #
# Pronosticos de precipitacion para SON
mod_cm4 =  xr.open_dataset('~/PronoClim/modelos_seteados/'
                           'prec_CMC-CanCM4i-IC3_SON.nc')
mod_gem =  xr.open_dataset('~/PronoClim/modelos_seteados/'
                           'prec_CMC-GEM5-NEMO_SON.nc')

# precipitacion observada
prec = xr.open_dataset('~/PronoClim/obs_seteadas/prec_monthly_nmme_cpc_sa.nc')

# indices
n34 = xr.open_dataset('~/PronoClim/indices_nc/nino34_mmean.nc')
n34 = n34 - n34.mean('time') # este no está sin anomalias
dmi = xr.open_dataset('~/PronoClim/indices_nc/iod_mmean.nc')
sam = xr.open_dataset('~/PronoClim/indices_nc/sam_mmean.nc')

# Selección de fechas que me importan.
# En este caso (arbitrario) mismo mes que los pronosticos.
# Agosto --> SON.
dmi = dmi.sel(time=dmi.time.dt.year.isin(mod_cm4.time.dt.year))
dmi = dmi.sel(time=dmi.time.dt.month.isin(mod_cm4.time.dt.month))
n34 = n34.sel(time=n34.time.dt.year.isin(mod_cm4.time.dt.year))
n34 = n34.sel(time=n34.time.dt.month.isin(mod_cm4.time.dt.month))
sam = sam.sel(time=sam.time.dt.year.isin(mod_cm4.time.dt.year))
sam = sam.sel(time=sam.time.dt.month.isin(mod_cm4.time.dt.month))
# Selecciono de esta manera ya que la codificación de tiempos es ligeramente
# diferente

# Anomalia y selección de la season SON observada
prec = prec.rolling(time=3, center=True).mean()
prec = prec.sel(time=prec.time.dt.month.isin(10))
prec = CrossAnomaly_1y(prec, norm=True)


################################################################################
# Ejemplos:
# ---------------------------------------------------------------------------- #
# Uso de la clase MLR y sus metodos. ----------------------------------------- #
# ---------------------------------------------------------------------------- #
# Suponiendo que queremos hacer una regresión lineal multiple usando los
# indices nino34, dmi y sam: pp = b1*n34 + b2*dmi + b3*sam + c

predictores = [n34/n34.std(), dmi/dmi.std(), sam/sam.std()]

# se inicia la clase mlr
mlr = MLR(predictores)

# se realiza la regresión multiple a cada punto de reticula por separado
# En este calculo va usar los predictores que usamos para iniciar la clase
prec_regre = mlr.compute_MLR(prec.prec)

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
                prec_regre.sel(coef='nino34_1'),
                scale=np.arange(-1,1.2,0.2),
                cmap='BrBG', title='Coef. Niño3.4 - PP Observada SON')

# Tambien podemos aplicar MLR al MODELO con los mismos predictores
# A la media del ensamble del modelo CanCM4i (probar con GEM5-NEMO)
# Tomamos anomalia con validacion cruzada
aux_mod_cm4 = mod_cm4.mean('r')
aux_mod_cm4 = CrossAnomaly_1y(aux_mod_cm4, norm=True)

regre_result_cm4 = mlr.compute_MLR(aux_mod_cm4.prec)

PlotContourf_SA(regre_result_cm4,
                regre_result_cm4.sel(coef='nino34_1'),
                scale=np.arange(-1,1.2,0.2),
                cmap='BrBG', title='Coef. Niño3.4 - PP CanCM4i-MME SON')


# La clase MLR tambien maneja los modelos con todos sus miembros de ensamble
# TOMA MAS TIEMPO! aprox 1:30 min
aux_mod_cm4 = CrossAnomaly_1y(mod_cm4, norm=True, r=True)


regre_result_cm4_r = mlr.compute_MLR(aux_mod_cm4.prec)

# ahora el resultado es igual que antes pero con una dimensión más.
print(regre_result_cm4_r.dims)

# Podemos acceder a los coeficientes de cada miembro
PlotContourf_SA(regre_result_cm4_r,
                regre_result_cm4_r.sel(coef='nino34_1', r=1),
                scale=np.arange(-1,1.2,0.2),
                cmap='BrBG', title='Coef. Niño3.4 - PP CanCM4i-r1 SON')

PlotContourf_SA(regre_result_cm4_r,
                regre_result_cm4_r.sel(coef='nino34_1', r=5),
                scale=np.arange(-1,1.2,0.2),
                cmap='BrBG', title='Coef. Niño3.4 - PP CanCM4i-r5 SON')

# Y operar con ellos
# ploteamos la media del ensamble del coeficiente
PlotContourf_SA(regre_result_cm4_r,
                regre_result_cm4_r.sel(coef='nino34_1').mean('r'),
                scale=np.arange(-1,1.2,0.2),
                cmap='BrBG', title='Coef. Niño3.4 - PP CanCM4i SON')

################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo usando los coeficientes en
# un ejemplo con periodos de training y testing
# ---------------------------------------------------------------------------- #
# training de 1983-2010, los ultimos años de testing
# Con el dataset observado
training = prec.sel(time=slice('1983-10-01', '2010-10-01'))
testing = prec.drop_sel(time=training.time.values)

# predictores
predictores_training = []
predictores_testing = []
for p in predictores:
    aux_p = p.sel(time=slice('1983-08-01', '2010-08-01'))

    predictores_training.append(aux_p)
    predictores_testing.append(p.drop_sel(time=aux_p.time.values))

# MLR en trainging solamente
mlr = MLR(predictores_training)
gem5_nemo_regremodel = mlr.compute_MLR(training.prec)

predictores_name = list(gem5_nemo_regremodel.coef.values) # nombre de los coef

# Pronostico sobre testing
# usando los valores que adquieren los predictores en ese periodo con los
# coeficientes obtenidos del mlr

# y = a + b1*x1 + b2*x2 + b3*x3 + ... + bnxn

# El siguiente for se podria hacer "a mano". Acá un ejemplo para cualquiera sea
# la cantidad de predictores
predict = gem5_nemo_regremodel.sel(coef=predictores_name[0]).drop_vars('coef')

for c, p in enumerate(predictores_testing):

    # Selección del indice y los valores del predictor en testing
    aux = gem5_nemo_regremodel.sel(
        coef=predictores_name[c+1])*p[list(p.data_vars)[0]]

    aux = aux.drop_vars('coef') # no deja sumar sino

    predict = predict + aux

# Podemos ver como resulto en ciertas fechas:
# PROBAR LOS AÑOS: 2015, 2019, 2020
# Pronostico MLR
PlotContourf_SA(predict,
                predict.sel(time='2015-08-01'),
                scale=np.arange(-2, 2.2, 0.2),
                cmap='BrBG', title='MLR Forecast')

PlotContourf_SA(prec,
                prec.sel(time='2015-10-01').prec[0,:,:],
                scale=np.arange(-2, 2.2, 0.2),
                cmap='BrBG', title='Observado')

# Podemos operar con ellos también: ------------------------------------------ #

# La función requiere que sean xr.DataSet (puede ser facilmente implementado
# dentro de la función)
ds_predict =  xr.Dataset(
    data_vars={'prec': (('lat', 'lon', 'time'), predict.values)},
    coords={'lon':predict.lon.values,
            'lat':predict.lat.values,
            'time':predict.time.values}
)
prec_sel = prec.sel(time=slice('2011-10-01','2020-10-01'))

acc_result = ACC(ds_predict, prec_sel)

PlotContourf_SA(acc_result,
                acc_result.prec,
                scale=np.arange(-1, 1.2, 0.2),
                cmap='BrBG', title='ACC - MLR Forecast - 2011-2020')

################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo de MLR con validación cruzada.
# ---------------------------------------------------------------------------- #
# usando el modelo gem5-nemo
aux_mod_gem = mod_gem.mean('r')
aux_mod_gem = CrossAnomaly_1y(aux_mod_gem, norm=True)

# Está función usa la clase MLR. Puede no ser la manera mas efectiva ni
# practica.
# la manera en la que se devuelven los resultados busca satisfacer
# varias utilidades posteriores de ellos pero puede no ser la mas adecuada.

mlr = MLR(predictores)
# Tarda aprox 5min
regre_result_gem_cv, mod_gem_years_out, predictores_years_out = (
    Compute_MLR_CV(aux_mod_gem, predictores, window_years=3, intercept=True))

# Salidas de esta funcion:
#     1. array de dimensiones [k, lon, lat, coef.] o [k, r, lon, lat, coef]
#     - "k" son los "k-fold": años seleccionados para el modelo lineal
#     - "coef" coeficientes del modelo lineal en cada punto de grilla en orden:
#      constante, b1, b2..., bn
#      2. idem 1. sin "coef" y en "k" estan los años omitidos en cada k-fold
#      3. dict. Diccionario donde cada elemento es una lista de xr.dataarry con
#      los predictores en los años omitods en cada k-fold

# Ejemplo de uso de las salidas de la funcion anterior
k=15 # los años omitidos en el k-fold 15
coefmodel = regre_result_gem_cv.sel(k=k) # coef. del modelo del k-fold
predictores_name = list(coefmodel.coef.values) # nombre de los coef

indices = predictores_years_out[f"k{k}"] # valor predictores en los años omitdos

# Lo mismo que antes
predict = coefmodel.sel(coef=predictores_name[0]).drop_vars('coef')
for c, p in enumerate(indices):
    aux = coefmodel.sel(coef=predictores_name[c+1])*p[list(p.data_vars)[0]]
    aux = aux.drop_vars('coef')
    predict = predict + aux

# Media del pronostico para los años omitods
PlotContourf_SA(predict,
                predict.mean('time'),
                scale=np.arange(-1, 1.2, 0.2),
                cmap='BrBG', title='GEM5-NEMO - MLR Forecast k-fold=15')

# Años omitidos
mod_gem_years_out.sel(k=k).mean('time').prec
PlotContourf_SA(mod_gem_years_out,
                mod_gem_years_out.sel(k=k).mean('time').prec,
                scale=np.arange(-1, 1.2, 0.2),
                cmap='BrBG', title='GEM5-NEMO - k-fold=15')

################################################################################
################################################################################
################################################################################