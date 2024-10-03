"""Funciones para las practicas"""
# import --------------------------------------------------------------------- #
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

# ---------------------------------------------------------------------------- #
# funciones auxiliares
def verbose_fun(message, verbose=True):
    if verbose:
        print(message)

# ---------------------------------------------------------------------------- #
# Funciones ------------------------------------------------------------------ #
def Signal(data):
    """
    Calcula la señal para cada punto de reticula como la varianza de la
    media del ensamble

    Parametros:
    data (xr.Dataset): con dimenciones lon, lat, time y r (orden no importa)

    return (xr.Dataset): campo 2D lon, lat
    """
    try:
        signal = data.mean('r').var('time')

    except ValueError as e:
        print(f"Error: {e}")
        signal = None

    return signal

def Noise(data):
    """
    Calcula el ruido para cada punto de reticula como la varianza de los
    miembros de ensamble respecto a la media del ensamble

    Parametros:
    data (xr.Dataset): con dimenciones lon, lat, time y r

    return (xr.Dataset): campo 2D lon, lat
    """
    try:
        len_r = len(data.r)
        len_t = len(data.time)

        ensemble_mean = data.mean('r')
        aux = (data - ensemble_mean)**2
        noise = aux.sum(['time', 'r']) / (len_r*len_t)

    except ValueError as e:
        print(f"Error: {e}")
        noise = None

    return noise
# ---------------------------------------------------------------------------- #
def CrossAnomaly_1y(data):
    """
    Toma anomalias cruzadas con ventana de un año

    Parametros:
    data (xr.Dataset): con dimenciones lon, lat, time
    return (xr.Dataset): campo 2D lon, lat, time
    """
    for t in data.time.values:
        data_t = data.sel(time=t)
        data_no_t = data.where(data.time != t, drop=True)

        if t is data.time.values[0]:
            data_anom = data_t - data_no_t.mean(['time'])
        else:
            aux_anom = data_t - data_no_t.mean(['time'])
            data_anom = xr.concat([data_anom, aux_anom], dim='time')

    return data_anom

# ---------------------------------------------------------------------------- #
def ACC(data1, data2, crossanomaly=False, reference_time_period=None,
        verbose=True):
    """
    Anomaly correlation coefficient

    parametros:
    data1 (xr.Dataset): con dimenciones lon, lat, time (orden no importa)
    data2 (xr.Dataset): con dimenciones lon, lat, time (orden no importa)
    crossanomaly (bool): default False, True toma la anomalia cruzada de un año
    reference_time_period (list, opcional): [año_inicial, año_final]
    para el cálculo de anomalías. Si es None y crossanomaly es False,
    no se calculan anomalías.
    verbose (bool): Si es True, imprime mensajes de procesamiento en terminal.

    - si 'crossanomaly' es True no se tiene en cuenta 'reference_time_period'
    - si 'crossanomaly' es False y 'reference_time_period' es None,
    se asume que data1 y data2 ya contienen anomalías.
    - si uno de los data tiene una dimención más el ACC se va devolver tambien
    para esa dimencion. Ej. data1 [lon, lat, time, r], data2 [lon, lat, time]
    el resultado va ser acc [lon, lat, r]

    Ejemplo de uso:
    ACC(data1, data2, crossanomaly=False, reference_time_period=[1985, 2010])
    """
    # ------------------------------------------------------------------------ #
    # Controles -------------------------------------------------------------- #
    dim_time_ok = 'time' in data1.dims and 'time' in data2.dims
    data_time_ok = len(data1.time) == len(data2.time) if dim_time_ok else False

    if not dim_time_ok:
        print("Error: data1 y data2 deben tener una variable 'time'")

    if list(data1.data_vars) == list(data2.data_vars):
        name_variable_ok = True
    else:
        name_variable_ok = False

    # ------------------------------------------------------------------------ #
    # Anomalias -------------------------------------------------------------- #
    rtp = reference_time_period
    if crossanomaly:
        verbose_fun('Anomalía cruzada con ventana de 1 año', verbose=verbose)

        data1 = CrossAnomaly_1y(data1)
        data2 = CrossAnomaly_1y(data2)

    elif isinstance(rtp, list) and len(rtp) == 2:
        verbose_fun(f"Anomalía en base al periodo {rtp[0]} - {rtp[1]}",
                    verbose=verbose)

        periodo = np.arange(rtp[0], rtp[1] + 1)

        data1_clim = (data1.sel(time=data1.time.dt.year.isin(periodo))
                      .mean(['time']))
        data1 = data1 - data1_clim

        data2_clim = (data2.sel(time=data2.time.dt.year.isin(periodo))
                      .mean('time'))
        data2 = data2 - data2_clim
    else:
        verbose_fun('No se calcularán anomalías \nSe asume que data1 y data2 '
                    'ya contienen anomalías', verbose=verbose)

    # ------------------------------------------------------------------------ #
    # Calculo ---------------------------------------------------------------- #
    if dim_time_ok and data_time_ok and name_variable_ok:

        # numerador del ACC. 1/T sum t->T data1*data2
        for t in range(0, len(data1.time.values)):
            data1_t = data1.sel(time=data1.time.values[t])
            data2_t = data2.sel(time=data2.time.values[t])

            if t == 0:
                num = data1_t * data2_t
            else:
                aux_num = data1_t * data2_t
                num = xr.concat([num, aux_num], dim='time')


        num = num.sum(['time'])/len(data1.time.values)

        # denominador del ACC 1/T [sum t->T data1**2 * sum t->T data2**2]
        data1_sum_sq = data1**2
        data1_sum_sq = data1_sum_sq.sum('time')

        data2_sum_sq = data2**2
        data2_sum_sq = data2_sum_sq.sum('time')

        den = np.sqrt((data1_sum_sq*data2_sum_sq))/len(data1.time.values)

        # acc
        acc = num/den

    else:
        if data_time_ok is False:
            print("Error: 'time' en data1 y data2 debe tener la misma longitud")
        elif name_variable_ok is False:
            print("Error: la variable en data1 y data2 debe tener el mismo "
                  "nombre")

        acc = None

    return acc

# ---------------------------------------------------------------------------- #
def ACC_Teorico(data):
    """
    Anomaly correlation coefficient teorico
    promedio del ACC entre MME y cada miembro de ensamble

    parametros:
    data (xr.Dataset): con dimenciones lon, lat, time, r (no importa el orden)

    return (xr.Dataset): campo 2D lon, lat.
    """
    # Controles -------------------------------------------------------------- #
    required_dims = ['lon', 'lat', 'time', 'r']
    dims_ok = False
    if not all(dim in data.dims for dim in required_dims):
        print(f"Error: data debe tener las dimensiones: "
              f"{', '.join(required_dims)}")
    else:
        dims_ok = True

    if dims_ok is True:
        # Calculo ------------------------------------------------------------ #
        for r in data.r.values:
            data_r = data.sel(r=r)
            data_r = data_r - data_r.mean('time')

            data_mme_no_r = data.where(data.r != r, drop=True).mean('r')
            data_mme_no_r = data_mme_no_r - data_mme_no_r.mean('time')

            if r == data.r.values[0]:
                r_theo_acc = ACC(data_mme_no_r, data_r, verbose=False)
            else:
                aux_r_theo_acc = ACC(data_mme_no_r, data_r, verbose=False)
                r_theo_acc = xr.concat([r_theo_acc, aux_r_theo_acc], dim='r')

        theo_acc = r_theo_acc.mean('r')
    else:
        theo_acc = None

    return theo_acc

# ---------------------------------------------------------------------------- #
def PlotContourf_SA(data, data_var, scale, cmap, title):
    """
    Funcion de ejemplo de ploteo de datos georeferenciados

    Parametros:
    data (xr.Dataset): del cual se van a tomar los valores de lon y lat
    data_var (xr.Dataarray): variable a graficar
    scale (array): escala para plotear contornos
    cmap (str): nombre de paleta de colores de matplotlib
    title (str): título del grafico
    """
    crs_latlon = ccrs.PlateCarree()

    fig = plt.figure(figsize=(5,6), dpi=100)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([275, 330, -60, 20], crs=crs_latlon)

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

    ax.set_xticks(np.arange(275, 330, 10), crs=crs_latlon)
    ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
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