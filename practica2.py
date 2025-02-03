"""
Ejemplos Practica 2.
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np

from funciones_practicas import (PlotContourf_SA, ACC, Compute_MLR,
                                 Compute_MLR_training_testing,
                                 Compute_MLR_CV)
# ---------------------------------------------------------------------------- #
ruta = '~/PronoClim/obs_seteadas/'

# precipitacion observada - Sudamerica
prec = xr.open_dataset('~/PronoClim/obs_seteadas/prec_monthly_nmme_cpc_sa.nc')

# SST observada - Hemisferio sur
sst = xr.open_dataset(f'{ruta}sst_ERSSTv5_1980-2020.nc')

# SLP observada - hemisferio sur
slp = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')

# indices
n34 = xr.open_dataset('~/PronoClim/indices_nc/nino34_mmean.nc')
n34 = n34 - n34.mean('time') # este no está sin anomalias
dmi = xr.open_dataset('~/PronoClim/indices_nc/iod_mmean.nc')
sam = xr.open_dataset('~/PronoClim/indices_nc/sam_mmean.nc')

################################################################################
# Regresion Lineal Multiple (MLR)
################################################################################
# Ejemplos:
# Suponiendo que queremos hacer una regresión lineal multiple usando los
# indices nino34, dmi y sam: pp = b1*n34 + b2*dmi + b3*sam + c

# predictores en agosto para precipitacion en Septiembre-Octubre-Noviembre (SON)
prec_regresion_son, prec_regresion_anomalias_son, prec_reconstruccion_son, \
    prec_to_verif_son = Compute_MLR(
    predictando=prec,
    mes_predictando=10, # Octubre
    predictores = [n34, dmi, sam], # Normalizados dentro de la funcion
    meses_predictores = [8, 8, 8], # Agosto
    predictando_trimestral=True, # Octubre --> SON
    predictores_trimestral=False) # queda en agosto

print(prec_regresion_son.dims)
# La dimensión "coef" contiene los coeficientes del modelo de regresión para
# CADA PUNTO DE RETICULA

print(prec_regresion_son.coef)
# En este caso:
# -prec_1: ordenada al origen
# -nino34_1: el coeficiente de regresión asociadao al indice dmi. (b1*n34)
# -dmi_2: el coeficiente de regresión asociadao al indice dmi. (b2*dmi)
# -sam_3: etc...
# Si los predictores son solo listas los coeficientes seran "coef_1", "coef_2"

print(prec_reconstruccion_son)
# es igual a prec pero reconstruido a partir de la regresión

# Ploteamos los valores del coeficiente asociado al n34. (probar los otros)
PlotContourf_SA(data=prec_regresion_son,
                data_var=prec_regresion_son.sel(coef='nino34_1').prec,
                scale=np.arange(0,180, 20), cmap='YlGnBu',
                title='Coef. Niño3.4 - Prec. SON',
                mask_andes=True, # enmascara la coordillera
                mask_ocean=False) # enmascara el oceano
# En anomalía
PlotContourf_SA(data=prec_regresion_anomalias_son,
                data_var=prec_regresion_anomalias_son.sel(coef='nino34_1'),
                scale=np.arange(-1,1.2,0.2), cmap='BrBG',
                title='Coef. Niño3.4 - Prec. anomalía SON',
                mask_andes=True, mask_ocean=False)

# Podemos ver la reconstruccion y la observada
PlotContourf_SA(data=prec,
                data_var=prec.sel(
                    time='2015-10-01').prec[0,:,:], # esto [0,:,:] puede no ser siempre necesario
                scale=np.arange(0,200, 20), cmap='YlGnBu',
                title='Prec. observada - SON-2015',
                mask_andes=True, mask_ocean=False)

PlotContourf_SA(data=prec_reconstruccion_son,
                data_var=prec_reconstruccion_son.sel(time='2015-10-01').prec[0,:,:],
                scale=np.arange(0,200, 20), cmap='YlGnBu',
                title='Reconstrucción MLR  Prec. - SON-2015',
                mask_andes=True, mask_ocean=False)

# ---------------------------------------------------------------------------- #
# Ejemplo con lags que requieren predictores en AÑO DIFERENTE al predictando
# EJEMPLO: DMI en octubre, N34 en noviembre, SAM en enero y Prec. en FMA

anios_predictores = [np.arange(1982,2020), # DMI un año antes que prec
                     np.arange(1982,2020), # N34 un año antes que prec
                     prec.time.dt.year]  # SAM en el mismo año que prec

prec_regresion_fma, prec_regresion_anomalias_fma, prec_reconstruccion_fma, \
    prec_to_verif_mlr_fma = Compute_MLR(
        predictando=prec,
        mes_predictando=3,  # Marzo (mes central de FMA)
        predictores=[n34, dmi, sam],
        meses_predictores=[10, 11, 1],  # meses de cada indice
        predictando_trimestral=True,  # Mar --> FMA
        predictores_trimestral=False,
        anios_predictores=anios_predictores)  # <==== ANIOS PARA CADA PREDICTOR

PlotContourf_SA(data=prec,
                data_var=prec.sel(time='2020-03-01').squeeze().prec,
                scale=np.arange(0,200, 20), cmap='YlGnBu',
                title='Prec. observada - SON-2015',
                mask_andes=True, mask_ocean=False)

PlotContourf_SA(data=prec_reconstruccion_fma,
                data_var=prec_reconstruccion_fma.sel(
                    time='2020-03-01').squeeze().prec,
                scale=np.arange(0,200, 20), cmap='YlGnBu',
                title='Reconstrucción MLR  Prec. - SON-2015',
                mask_andes=True, mask_ocean=False)

################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo con periodos de training y testing
# ---------------------------------------------------------------------------- #
# Se le deben dar a la funcion los anios del comienzo y final de los periodos
# que van a ser usados como training y testing
prec_regresion_tt, prec_regresion_anomalias_tt, prec_reconstruccion_tt, \
    prec_to_verif_tt = Compute_MLR_training_testing(
    predictando=prec, mes_predictando=10, # Oct
    predictores=[n34, dmi, sam],
    meses_predictores=[8, 8, 8],  # Ago
    anios_training=[1983, 2000],  # Training
    anios_testing=[2001, 2020],  # Testing
    predictando_trimestral=True,  # Oct --> SON
    predictores_trimestral=False,
    anios_predictores_testing=None,  # ***
    anios_predictores_training=None,  # ***
    reconstruct_full=False)  # ****

# Los resultados tienen la misma forma que en Compute_MLR

# Ejemplo graficar un año de los reconstruidos
PlotContourf_SA(data=prec_reconstruccion_tt,
                data_var=prec_reconstruccion_tt.sel(
                    time='2015-10-01').squeeze().prec,
                scale=np.arange(0, 200, 20),
                cmap='YlGnBu',
                title='Reconstrucción MLR testing Prec. - SON-2015',
                mask_andes=True, mask_ocean=False)

# Si queremos ver ese año pero en anomalia
from funciones_practicas import SingleAnomalia_CV

pp_mlr_2015 = SingleAnomalia_CV(prec_reconstruccion_tt, 2015)

PlotContourf_SA(data=pp_mlr_2015,
                data_var=pp_mlr_2015,
                scale=np.arange(-50, 55, 5),
                cmap='BrBG',
                title='Anomalia Prec. Reconstrucción MLR testing - SON-2015',
                mask_andes=True, mask_ocean=False)

# Anomalías observadas
aux = prec.rolling(time=3, center=True).mean()
aux = aux.sel(time=aux.time.dt.month.isin(10))
pp_obs_2015 = SingleAnomalia_CV(aux, 2015)
del aux

PlotContourf_SA(data=pp_obs_2015,
                data_var=pp_obs_2015,
                scale=np.arange(-50, 55, 5),
                cmap='BrBG',
                title='Anomalia Prec. observada - SON-2015',
                mask_andes=True, mask_ocean=False)

# ****
# Si usamos reconstruct_full = True
# Va a usar: training para computar la MLR y luego reconstruir testing
# y ademas va intercambiar los periodos.
# Va usar tambien testing para computar MLR y luego reconstruir training
# De esta forma reconstruye el periodo completo

# ***
# ademas, en anios_predictores_testing/training podemos especificar diferentes
# años para cada predictor, al igual que con Compute_MLR
# DEBE SER CONSISTENTE ENTRE meses_predictores, anios_training y anios_testing

# Un ejemplo combianando las dos cosas:
prec_regresion_tt2, prec_regresion_anomalias_tt2, prec_reconstruccion_tt_full, \
    prec_to_verif_tt_full  = Compute_MLR_training_testing(
    predictando=prec, mes_predictando=3,
    predictores=[n34, dmi, sam],
    meses_predictores=[10, 11, 1],
    anios_training=[1983, 2000],
    anios_testing=[2001, 2020],
    predictando_trimestral=True,
    predictores_trimestral=False,

    # Modificamos estos argumentos:
    anios_predictores_testing=[[2000, 2019],
                               [2000, 2019],
                               [2001, 2020]],
    anios_predictores_training=[[1982, 1999],
                                [1982, 1999],
                                [1983, 2000]],
    reconstruct_full=True)

# Con todos estos resultados anteriores tambien se puede operar

acc_result = ACC(prec_to_verif_tt_full, prec_reconstruccion_tt_full, cvanomaly=True,
                 reference_time_period=[1983,2020])

PlotContourf_SA(acc_result,
                acc_result.prec,
                scale=np.arange(-1, 1.2, 0.2),
                cmap='BrBG',
                title='ACC - Prec. observada vs reconstruccion MLR',
                mask_andes=True, mask_ocean=False)

################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo de MLR con validación cruzada.
# Los resultados tienen la misma forma que los anteriores
# TOMA UNOS 5-6 MINUTOS
# ---------------------------------------------------------------------------- #
prec_regresion_cv, prec_regresion_anomalias_cv, prec_reconstruccion_cv, \
    prec_to_verif_cv = Compute_MLR_CV(predictando=prec, mes_predictando=10,
                   predictores=[n34, dmi, sam],
                   meses_predictores=[8,8,8],
                   predictando_trimestral=True,
                   predictores_trimestral=False,
                   anios_predictores=None,
                   window_years=3) # Ventana de anios a considerar en la cv

PlotContourf_SA(data=prec_regresion_cv,
                data_var=prec_regresion_cv.sel(coef='nino34_1'),
                scale=np.arange(-1,1.2,0.2),
                cmap='BrBG', title='Coef. Niño3.4 - Prec. anomalía SON',
                mask_andes=True, mask_ocean=False)

PlotContourf_SA(prec_reconstruccion_cv,
                prec_reconstruccion_cv.sel(time='2015-10-01').squeeze().prec,
                scale=np.arange(0, 200, 20),
                cmap='YlGnBu', title='Reconstrucción MLR CV Prec. - SON-2015',
                mask_andes=True, mask_ocean=False)

################################################################################
# Correlacion canonica (CCA)
from funciones_practicas import CCA_training_testing, CCA_mod_CV, Compute_CCA
################################################################################
# Funcion auxiliar para plotear
def Plot(data, data_var, scale, cmap, title = ''):

    import matplotlib.pyplot as plt
    import numpy as np
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    crs_latlon = ccrs.PlateCarree()

    ratio = len(data.lat)/len(data.lon)

    fig = plt.figure(figsize=(6.5, 7*ratio), dpi=100)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([data.lon.values[0],
                   data.lon.values[-1],
                   min(data.lat.values),
                   max(data.lat.values)], crs=crs_latlon)

    # Contornos
    im = ax.contourf(data.lon,
                     data.lat,
                     data_var,
                     levels=scale,
                     transform=crs_latlon, cmap=cmap, extend='both')

    # barra de colores
    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)

    ax.coastlines(color='k', linestyle='-', alpha=1)

    ax.set_xticks(np.arange(data.lon.values[0], data.lon.values[-1], 20),
                  crs=crs_latlon)
    ax.set_yticks(np.arange(min(data.lat.values), max(data.lat.values), 10),
                  crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.tick_params(labelsize=10)

    plt.title(title, fontsize=12)

    plt.tight_layout()
    plt.show()

# Variables ----------- #
# Es IMPORTANTE, y no da lo mismo, el dominio espacial y temporal
# con el que se va a trabajar.

# Sumamos SST y SLP
ruta = '~/PronoClim/obs_seteadas/'

# SST
sst = xr.open_dataset(f'{ruta}sst_ERSSTv5_1980-2020.nc')
sst = sst.sel(lon=slice(150, 280), lat=slice(15,-15))

# SLP
slp = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')
slp = slp.sel(lon=slice(210,320), lat=slice(-20,-75))

prec_sel = prec.sel(lon=slice(280, 320), lat=slice(-45,-20))
################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo basico de CCA:
# ---------------------------------------------------------------------------- #
P, Q, A, B, S = Compute_CCA(
    X=slp,
    Y=prec,
    var_exp=0.7, # varianza que queremos retener
    X_mes=8, # Agosto para X (slp)
    Y_mes=10, # Octubre para Y (prec)
    X_trimestral=False, # Ago. --> Ago.
    Y_trimestral=True, # Oct. --> SON
    X_anios=[1983, 2020], # periodo para X (util cuando un lag es del anio anterior)
    Y_anios=[1983, 2020]) # periodo para Y, deben tener la misma longitud

# Devuelve:
# P y Q mapas canonicos de X e Y, respectivamente
# S: valores singulares/corPrelacion entre A y B
# A y B: vectores canonicos de X e Y, respectivamente

Plot(P, P.sel(modo=1), np.arange(-1, 1.2, .2), cmap='RdBu_r',
     title='SLP Modo=1')
Plot(P, P.sel(modo=2), np.arange(-1, 1.2, .2), cmap='RdBu_r',
     title='SLP Modo=2')

Plot(Q, Q.sel(modo=1), np.arange(-1, 1.2, .2), cmap='BrBG',
     title='SLP Modo=1')
Plot(Q, Q.sel(modo=2), np.arange(-1, 1.2, .2), cmap='BrBG',
     title='SLP Modo=2')

# Podemos ver la evolución temporal de estos campos en el tiempo
import matplotlib.pyplot as plt
plt.plot(A[:,0], label='Vector Canonico de SLP')
plt.plot(B[:,0], label='Vector Canonico de PP')
plt.legend()
plt.title(f'Modo 0 - r={S[0]:.3g}')
plt.xlabel('tiempo')
plt.show()
# S indica la correlacion entre los vectores canonicos
# podemos verlo:
print(f"Correlación A y B modo 0: {np.corrcoef(A[:,0], B[:,0])[0,1]}")
print(f"Valor de S[0] = {S[0]}")

# ---------------------------------------------------------------------------- #
# Con SST
P, Q, A, B, S = Compute_CCA(
    X=sst, # <----
    Y=prec,
    var_exp=0.7, # varianza que queremos retener
    X_mes=8,
    Y_mes=10,
    X_trimestral=False,
    Y_trimestral=True,
    X_anios=[1983, 2020],
    Y_anios=[1983, 2020])

# Devuelve:
# P y Q mapas canonicos de X e Y, respectivamente
# S: valores singulares/correlacion entre A y B
# A y B: vectores canonicos de X e Y, respectivamente

Plot(P, P.sel(modo=1), np.arange(-1, 1.2, .2), cmap='RdBu_r',
     title='SLP Modo=1')

Plot(Q, Q.sel(modo=1), np.arange(-1, 1.2, .2), cmap='BrBG',
     title='Prec Modo=1')

################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo con periodos de training y testing
# ---------------------------------------------------------------------------- #
prec_anom_cca_tt, prec_anom_to_verif_tt  = CCA_training_testing(
    X=slp,
    Y=prec_sel, # usando una subseleccion de Sudamerica
    var_exp=0.7,
    X_mes=7,
    Y_mes=10,
    X_trimestral=True, Y_trimestral=True,
    X_anios=[1983, 2020], Y_anios=[1983, 2020],
    anios_training=[1983, 2000], # testing
    anios_testing=[2001, 2020], # training
    reconstruct_full=True) # Similar al de MLR, training <-> training

print(prec_anom_cca_tt)
# Es la reconstruccion del campo de anomalias a partir de CCA
print(prec_anom_to_verif_tt)
# Es la anomalia de la precipitacion observada en el mismo dominio temporal
# de prec_anom_cca, sea cual sea

# Reconstruido
Plot(data=prec_anom_cca_tt,
     data_var=prec_anom_cca_tt.sel(time='2015-10-01').squeeze().prec,
     scale=np.arange(0, 200, 20), cmap='YlGnBu',
     title='Anomalia Prec. Reconstrucción CCA-tt - SON-2015')

# Observado
Plot(data=prec_anom_to_verif_tt,
     data_var=prec_anom_to_verif_tt.sel(time='2015-10-01').squeeze().prec,
     scale=np.arange(0, 200, 20), cmap='YlGnBu',
     title='Anomalia Prec. observada - SON-2015')

################################################################################
# ---------------------------------------------------------------------------- #
# Ejemplo con validacion cruzada
# ---------------------------------------------------------------------------- #
pp_anom_cca_cv, prec_anom_to_verif_cv  = CCA_mod_CV(X=slp, Y=prec_sel,
                                                    var_exp=0.7,
                                                    X_mes=7, Y_mes=10,
                                                    X_trimestral=True,
                                                    Y_trimestral=True,
                                                    X_anios=[1983, 2020],
                                                    Y_anios=[1983, 2020],
                                                    window_years=3)

Plot(data=pp_anom_cca_cv,
     data_var=pp_anom_cca_cv.sel(time='2015-10-01').squeeze().prec,
     scale=np.arange(0, 200, 20), cmap='YlGnBu',
     title='Anomalia Prec. Reconstrucción CCA-cv - OND-2015')

# Observado
Plot(data=prec_anom_to_verif_cv,
     data_var=prec_anom_to_verif_cv.sel(time='2015-10-01').squeeze().prec,
     scale=np.arange(0, 200, 20), cmap='YlGnBu',
     title='Anomalia Prec. observada - OND-2015')

################################################################################
################################################################################