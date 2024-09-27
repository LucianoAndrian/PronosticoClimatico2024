"""Funciones para setear archivos de modelos"""
# import --------------------------------------------------------------------- #
import xarray as xr
import numpy as np
# ---------------------------------------------------------------------------- #
def Signal(data):
    """Calcula la señal para cada punto de reticula
    como la varianza de la media del ensamble

    Parametros:
    data (xr.Dataset): con dimenciones lon, lat, time y r

    return (xr.Dataset): campo 2D lon, lat
    """
    try:
        signal = data.mean('r').var('time')

        # esto es igual a esto:
        # len_r = len(data.r)
        # len_t = len(data.time)
        #
        # ensemble_mean = data.mean('r')
        # aux = (ensemble_mean - data.mean(['time', 'r']))**2
        # signal = aux.sum(['time']) / (len_t)

    except ValueError as e:
        print(f"Error: {e}")
        signal = None

    return signal

def Noise(data):
    """Calcula el ruido para cada punto de reticula
    como la varianza de los miembros de ensamble respecto
    a la media del ensamble

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
    """Toma anomalias cruzadas con ventana de un año

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

def ACC(data1, data2):
    """Anomaly correlation coefficient

    parametros:
    data1 (xr.Dataset): con dimenciones lon, lat, time (no importa el orden)
    data2 (xr.Dataset): con dimenciones lon, lat, time (no importa el orden)
    """
    # Controles -------- #
    try:
        aux = data1.time
        aux = data2.time
        dim_time_ok = True
        data_time_ok = len(data1.time.values) == len(data2.time.values)
    except:
        dim_time_ok = False
        print("Error: data1 y data2 deben tener una variable 'time'")


    if list(data1.data_vars) == list(data2.data_vars):
        name_variable_ok = True
    else:
        name_variable_ok = False


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
            print("Error: la variable en data1 y data2 debe tener el mismo nombre")

        acc = None

    return acc


def ACC_Teorico(data):
    """Anomaly correlation coefficient teorico
    promedio del ACC entre MME y cada miembro de ensamble

    parametros:
    data (xr.Dataset): con dimenciones lon, lat, time (no importa el orden)

    return (xr.Dataset): campo 2D lon, lat.
    """
    try:
        aux = data.r
        aux = data.time
        dims_ok = True
    except:
        dims_ok = False
        print("Error: data debe tener dims lon, lat, time, r")

    if dims_ok is True:

        for r in data.r.values:
            data_r = data.sel(r=r)
            data_mme_no_r = data.where(data.r != r, drop=True).mean('r')

            if r == data.r.values[0]:
                r_theo_acc = ACC(data_mme_no_r, data_r)
            else:
                aux_r_theo_acc = ACC(data_mme_no_r, data_r)
                r_theo_acc = xr.concat([r_theo_acc, aux_r_theo_acc], dim='r')

        theo_acc = r_theo_acc.mean('r')
    else:
        theo_acc = None

    return theo_acc

# ---------------------------------------------------------------------------- #