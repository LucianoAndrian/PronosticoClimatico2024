"""Funciones para setear archivos de modelos"""
# import --------------------------------------------------------------------- #
import xarray as xr
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
