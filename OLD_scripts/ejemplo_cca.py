"""
Ejemplo de uso de funciones de CCA.

"""
# ---------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
from funciones_practicas import (CrossAnomaly_1y, CCA, CCA_training_testing,
                                 CCA_calibracion_training_testing, CCA_mod_CV,
                                 CCA_calibracion_CV)

# Funcion auxiliar para plotear ---------------------------------------------- #
# (adaptada de PlotContourf_SA)
def Plot(data, data_var, scale, cmap, title = ''):

    import matplotlib.pyplot as plt
    import numpy as np
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    crs_latlon = ccrs.PlateCarree()

    ratio = len(data.lat)/len(data.lon)

    fig = plt.figure(figsize=(6, 6*ratio), dpi=100)

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
# ---------------------------------------------------------------------------- #
ruta = '~/PronoClim/obs_seteadas/'

# SST - ASO
ds1 = xr.open_dataset(f'{ruta}sst_ERSSTv5_1980-2020.nc')
ds1 = ds1.rolling(time=3, center=True).mean('time')
ds1 = ds1.sel(time=ds1.time.dt.year.isin(np.arange(1984,2020)))
ds1 = ds1.sel(lon=slice(150, 280), lat=slice(15,-15))
sst_or = ds1.sel(time=ds1.time.dt.month.isin(9))

# SLP - ASO
ds3 = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')
ds3 = ds3.rolling(time=3, center=True).mean('time')
ds3 = ds3.sel(time=ds3.time.dt.year.isin(np.arange(1984,2020)))
ds3 = ds3.sel(lon=slice(210,320), lat=slice(-20,-75))
slp_or = ds3.sel(time=ds3.time.dt.month.isin(9))

# PP - OND
ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')
ds2 = ds2.rolling(time=3, center=True).mean('time')
ds2 = ds2.sel(time=ds2.time.dt.year.isin(np.arange(1984,2020)))
pp_or = ds2.sel(time=ds2.time.dt.month.isin(11))

# Anomalia cv y normalizo
sst = CrossAnomaly_1y(sst_or, norm=True).sst
pp = CrossAnomaly_1y(pp_or, norm=False).prec
slp = CrossAnomaly_1y(slp_or, norm=False).slp

# ---------------------------------------------------------------------------- #
# Uso de CCA ----------------------------------------------------------------- #
# var_exp=0.7 va seleccionar las componentes principales necesarias para
# explicar el 70% de la varianza. Eso lo hace de forma separada para X e Y
P, Q, heP, heQ, S, A, B = CCA(X=slp, Y=pp, var_exp=0.7)

# Obtengo np.ndarrays:
# P y Q mapas canonicos de X e Y, respectivamente
# heP, heQ: varianzas heterogeneas
# S: valores singulares/correlacion entre A y B
# A y B: vectores canonicos de X e Y, respectivamente

# Los mapas canonicos tienen dimencion (t, n) donde n=lon*lat
# Depende de que se quiera hacer con ellos luego podemos necesitar recuperarlos
# en las dimensiones (t, lon, lat)

print(P.shape)
# Hay 4 vectores canonicos, indicando que el metodo selecciono 2 componentes
# principales

# Para P, que es el mapa canonico de sst
aux = P.reshape(len(slp.lat.values), len(slp.lon.values), P.shape[1])
P_da = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                    coords={'lat': slp.lat.values,
                            'lon': slp.lon.values})

Plot(P_da, P_da.sel(cca=0), np.arange(-1, 1.2, .2), cmap='RdBu_r',
     title='SLP Modo=0')

Plot(P_da, P_da.sel(cca=1), np.arange(-1, 1.2, .2), cmap='RdBu_r',
     title='SLP Modo=1')

# idem con Q
aux = Q.reshape(len(pp.lat.values), len(pp.lon.values), Q.shape[1])
Q_da = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                    coords={'lat': pp.lat.values,
                            'lon': pp.lon.values})
Plot(Q_da, Q_da.sel(cca=0), np.arange(-1, 1.2, .2), cmap='BrBG',
     title='PP Modo=0')

Plot(Q_da, Q_da.sel(cca=1), np.arange(-1, 1.2, .2), cmap='BrBG',
     title='PP Modo=1')

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
# Probemos con sst y pp
P, Q, heP, heQ, S, A, B = CCA(X=sst, Y=pp, var_exp=0.7)
aux = P.reshape(len(sst.lat.values), len(sst.lon.values), P.shape[1])
P_da = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                    coords={'lat': sst.lat.values,
                            'lon': sst.lon.values})

Plot(P_da.sel(cca=0), P_da.sel(cca=0), np.arange(-1, 1.2, .2), cmap='RdBu_r')

aux = Q.reshape(len(pp.lat.values), len(pp.lon.values), Q.shape[1])
Q_da = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                    coords={'lat': pp.lat.values,
                            'lon': pp.lon.values})
Plot(Q_da.sel(cca=0), Q_da.sel(cca=0), np.arange(-1, 1.2, .2), cmap='BrBG')

# ---------------------------------------------------------------------------- #
# Uso de CCA_mod ------------------------------------------------------------- #
# Similar a CCA(), pero permite reconstruir/pronosticar a partir de un predictor
# por medio de CCA.

# AHORA --> CCA_training_testing (CCA_mode, está se usa dentro de esta funcion)
from funciones_practicas import CCA_training_testing_OLD
cca_pp, pp_to_verif = CCA_training_testing_OLD(X=slp_or, Y=pp_or, var_exp=0.7,
                                           anios_training=[1983, 2011],
                                           anios_testing=[2011, 2020],
                                           plus_anio=0)



ds3 = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')
ds3 = ds3.sel(lon=slice(210,320), lat=slice(-20,-75))
ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')
cca_pp, pp_to_verif  = CCA_training_testing(X=ds3, Y=ds2, var_exp=0.7,
                                            X_mes=9, Y_mes=11,
                                            X_trim=True, Y_trim=True,
                                            X_anios=[1983, 2020],
                                            Y_anios=[1983, 2020],
                                            anios_training=[1983, 2010],
                                            anios_testing=[2011, 2019],
                                            reconstruct_full=True)
# plus_anio es unicamente en la situacion que tengamos X en un año y Y en otro
# algo que puede suceder si trabajamos con DJF o MAM
# La idea es que si Y está en otro año plus_anio = 1

# Reconstruido
Plot(cca_pp, cca_pp.sel(time='2015-11-01').prec[0,:,:],
     np.arange(-120, 120, 20), cmap='BrBG',
     title='PP - Reconstruido: OND 2015')

# Observado
Plot(pp_to_verif, pp_to_verif.sel(time='2015-11-01').prec[0,:,:],
     np.arange(-100, 120, 20), cmap='BrBG',
     title='PP - Observado: OND 2015')

# ---------------------------------------------------------------------------- #
# Uso de CCA con validacion cruzada
from funciones_practicas import CCA_mod_CV_OLD
cca_pp_from_slp, pp_to_verif  = CCA_mod_CV(
    X=slp_or, Y=pp_or, var_exp=0.7, window_years=3)


ds3 = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')
ds3 = ds3.sel(lon=slice(210,320), lat=slice(-20,-75))
ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')

cca_pp_from_slp, pp_to_verif  = CCA_mod_CV(X=ds3, Y=ds2, var_exp=0.7,
                                           X_mes=9, Y_mes=11,
                                           X_trim=True, Y_trim=True,
                                           X_anios=[1983, 2020],
                                           Y_anios=[1983, 2020],
                                           window_years=3,
                                           X_test=None)


Plot(cca_pp_from_slp, cca_pp_from_slp.sel(time='2015-11-01').prec[0,:,:],
     np.arange(-90, 100, 10), cmap='BrBG',
     title='PP - Pronostico CCA-SLP SON 2015')

Plot(pp_to_verif, pp_to_verif.sel(time='2015-11-01').prec[0,:,:],
     np.arange(-90, 100, 10), cmap='BrBG',
     title='PP - Observado SON 2015')

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Calibración de Pronostico
# ---------------------------------------------------------------------------- #
ruta_mod = '~/PronoClim/modelos_seteados/'
mod_gem = xr.open_dataset(ruta_mod + 'prec_CMC-CanCM4i-IC3_SON.nc')*(92/3)
mod_gem = mod_gem.sel(time=mod_gem.time.dt.year.isin(np.arange(1984,2020)))
mod_gem_em = mod_gem.mean('r') # media del ensamble

ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')

pp_or_aux = ds2.sel(time=ds2.time.dt.month.isin(10)) # pp observada en DJF

mod_gem_cal_pp, pp_to_verif = (
    CCA_calibracion_training_testing_OLD(mod_gem, pp_or_aux, var_exp=0.7,
                                     anios_training=[1984, 2011],
                                     anios_testing=[2011, 2020]))


ruta_mod = '~/PronoClim/modelos_seteados/'
mod_gem = xr.open_dataset(ruta_mod + 'prec_CMC-CanCM4i-IC3_SON.nc')*(92/3)
ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')

adj_f, data_to_verif_f = CCA_calibracion_training_testing(
    X_modelo=mod_gem,
    Y_observacion=ds2,
    var_exp=0.7,
    Y_mes=10,
    Y_trim=True,
    X_anios=[1983, 2020],  # periodo para X
    Y_anios=[1983, 2020],  # periodo para Y, deben tener la misma longitud
    anios_training=[1983, 2000],  # testing
    anios_testing=[2001, 2020],  # training
    reconstruct_full=True)

Plot(mod_gem,
     mod_gem.sel(time='2015-08-01').mean('r').prec[0,:,:] - mod_gem_em.mean('time').prec,
     np.arange(-60, 65, 5), cmap='BrBG',
     title='PP - Pronostico SON 2015')

Plot(mod_gem_cal_pp, mod_gem_cal_pp.sel(time='2015-08-01').mean('r').prec[0,:,:],
     np.arange(-60, 65, 5), cmap='BrBG',
     title='PP - Pronostico Calibrado SON 2015')

Plot(pp_to_verif,
     pp_to_verif.sel(time='2015-10-01').prec[0,:,:],
     np.arange(-60, 65, 5), cmap='BrBG',
     title='PP - Observado SON 2015')

# ---------------------------------------------------------------------------- #
# Uso de CCA para calibracion del modelo y sus miembros de ensamble con
# validacion cruzada

ruta_mod = '~/PronoClim/modelos_seteados/'
mod_gem = xr.open_dataset(ruta_mod + 'prec_CMC-CanCM4i-IC3_SON.nc')*(92/3)
ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')

mod_gem_cal_pp_cv, pp_to_verif = CCA_calibracion_CV_OLD(X_modelo_full=mod_gem,
                                                    Y=pp_or,
                                                    var_exp=0.7,
                                                    window_years=3)

mod_gem_cal_pp_cv, pp_to_verif = CCA_calibracion_CV(
    X_modelo=mod_gem, Y_observacion=ds2, var_exp=0.7,
    Y_mes=10, Y_trim=True,
    X_anios=[1983, 2020],  # periodo para X
    Y_anios=[1983, 2020],  # periodo para Y, deben tener la misma longitud
    window_years=3)

Plot(mod_gem_cal_pp_cv, mod_gem_cal_pp_cv.mean('r').sel(time='2015-11-01').prec[0,:,:],
     np.arange(-60, 65, 5), cmap='BrBG',
     title='PP - Pronostico calibrado CV SON 2015')

Plot(pp_to_verif, pp_to_verif.sel(time='2015-11-01').prec[0,:,:],
     np.arange(-60, 65, 5), cmap='BrBG',
     title='PP - Observado SON 2015')

################################################################################