"""Funciones para las practicas"""
# import --------------------------------------------------------------------- #
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import pandas as pd
import statsmodels.api as sm
from climpred import HindcastEnsemble
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
    data (xr.Dataset): con dimensiones lon, lat, time y r (orden no importa)

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
    data (xr.Dataset): con dimensiones lon, lat, time y r

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
def CrossAnomaly_1y(data, norm=False, r=False, return_mean_sd = False):
    """
    Toma anomalias cruzadas con ventana de un año

    Parametros:
    data (xr.Dataset): con dimensiones lon, lat, time
    norm (bool): True normaliza
    r (bool): True considera la dimensión r
    return (xr.Dataset): campo 2D lon, lat, time
    """
    if r:
        for r_e in data.r.values:
            for t in data.time.values:
                data_r_t = data.sel(time=t, r=r_e)

                aux = data.where(data.r != r_e, drop=True)
                data_no_r_t = aux.where(aux.time != t, drop=True)

                aux_anom = data_r_t - data_no_r_t.mean(['time', 'r'])
                if norm:
                    aux_anom = aux_anom / data_no_r_t.std(['time', 'r'])

                if t == data.time.values[0]:
                    data_anom_t = aux_anom
                else:
                    data_anom_t = xr.concat([data_anom_t, aux_anom], dim='time')

            if int(r_e) == int(data.r.values[0]):
                data_anom = data_anom_t
            else:
                data_anom = xr.concat([data_anom, data_anom_t], dim='r')

        var_name = list(data_anom.data_vars.keys())[0]
        aux_data_anom = data_anom[var_name].transpose('time', 'r', 'lat', 'lon')

        # Asignar el DataArray reordenado de nuevo al Dataset
        data_anom = data_anom.assign(**{var_name: aux_data_anom})

    else:
        for t in data.time.values:
            data_t = data.sel(time=t)
            data_no_t = data.where(data.time != t, drop=True)

            aux_anom = data_t - data_no_t.mean(['time'])
            if norm:
                aux_anom = aux_anom / data_no_t.std(['time'])

            if t == data.time.values[0]:
                data_anom = aux_anom
                data_sd =  data_no_t.std(['time'])
                data_mean = data_no_t.mean(['time'])
            else:
                data_anom = xr.concat([data_anom, aux_anom], dim='time')
                data_sd = xr.concat([data_sd, data_no_t.std(['time'])],
                                    dim='time')
                data_mean = xr.concat([data_mean, data_no_t.mean(['time'])],
                                      dim='time')
    if return_mean_sd is True:
        data_sd = data_sd.mean('time')
        data_mean = data_mean.mean('time')
        return data_anom, data_mean, data_sd
    else:
        return data_anom


def SingleAnomalia_CV(data, anio):
    """

    :param data:
    :param anio:
    :return:
    """
    data_var = list(data.data_vars)[0]
    data_anio = data.sel(time=data.time.dt.year.isin(anio))
    media_cv = (data.where(data.time != data_anio.time.values, drop=True)
                .mean('time', skipna=True))
    anom =  data_anio - media_cv

    return anom[data_var].squeeze('time')
# ---------------------------------------------------------------------------- #
def ACC(data1, data2, cvanomaly=False, reference_time_period=None,
        verbose=True):
    """
    Anomaly correlation coefficient

    parametros:
    data1 (xr.Dataset): con dimensiones lon, lat, time (orden no importa)
    data2 (xr.Dataset): con dimensiones lon, lat, time (orden no importa)
    cvanomaly (bool): default False, True toma la anomalia cruzada de un año
    reference_time_period (list, opcional): [año_inicial, año_final]
    para el cálculo de anomalías. Si es None y cvanomaly es False,
    no se calculan anomalías.
    verbose (bool): Si es True, imprime mensajes de procesamiento en terminal.

    - si 'cvanomaly' es True no se tiene en cuenta 'reference_time_period'
    - si 'cvanomaly' es False y 'reference_time_period' es None,
    se asume que data1 y data2 ya contienen anomalías.
    - si uno de los data tiene una dimensión más el ACC se va devolver tambien
    para esa dimension. Ej. data1 [lon, lat, time, r], data2 [lon, lat, time]
    el resultado va ser acc [lon, lat, r]

    Ejemplo de uso:
    ACC(data1, data2, cvanomaly=False, reference_time_period=[1985, 2010])
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
    if cvanomaly:
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
    data (xr.Dataset): con dimensiones lon, lat, time, r (no importa el orden)

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
def PlotContourf_SA(data, data_var, scale, cmap, title, mask_ocean=False,
                    mask_andes=False, save=False, out_dir='~/', name_fig='fig'):
    """
    Funcion de ejemplo de ploteo de datos georeferenciados

    Parametros:
    data (xr.Dataset): del cual se van a tomar los valores de lon y lat
    data_var (xr.Dataarray): variable a graficar
    scale (array): escala para plotear contornos
    cmap (str): nombre de paleta de colores de matplotlib
    title (str): título del grafico
    mask_ocean (bool): mascara del oceano
    mask_andes (bool): mascara de los andes
    save (bool): guardar la figura
    out_dir (str): ruta del directorio de salida
    name_fig (str): nombre de la figura
    """
    crs_latlon = ccrs.PlateCarree()

    fig = plt.figure(figsize=(5,6), dpi=100)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([275, 330, -60, 20], crs=crs_latlon)

    if mask_ocean is True:
        try:
            mask_ocean = MakeMask(data_var)
            data_var = data_var * mask_ocean.mask
        except:
            print('regionmask no instalado, no se ensmacarará el océano')
            print('se puede instalar en el entorno con pip install regionmask')

    # Contornos
    im = ax.contourf(data.lon,
                     data.lat,
                     data_var,
                     levels=scale,
                     transform=crs_latlon, cmap=cmap, extend='both')

    # barra de colores
    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)

    if mask_andes is True:
        from SetCodes.descarga_topografia import compute
        topografia = compute()

        from matplotlib import colors
        andes_cmap = colors.ListedColormap(
            ['k'])  # una palenta de colores todo negro

        # contorno que va enmascarar el relieve superior a mask_level
        mask_level = 1300  # metros
        ax.contourf(topografia.lon, topografia.lat, topografia.topo,
                    levels=[mask_level, 666666],
                    cmap=andes_cmap, transform=crs_latlon)

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

    if save is True:
        print(f'Guardado en: {out_dir}{name_fig}.jpg')
        plt.savefig(f'{out_dir}{name_fig}.jpg', dpi=150)


# ---------------------------------------------------------------------------- #
class MLR:
    def __init__(self, predictores):
        self.predictores = predictores
        """
        Inicializa MLR

        Parametros:
        perdictores (list): con series temporales a usar como predictores
        los elementos de las listas pueden ser xr.DataSet, xr.DataArray o 
        np.ndarray 
        """

    def mlr_1d(self, series, intercept=True):
        """Funcion de regresión lineal multiple en una dimensión

        Parametros:
        series (dict): diccionario con nombre y datos de las series temporales
        siendo la primera el predictando y el resto predictores.
        intercept (bool): True, ordenada al orgine (recomendado usar siempre)

        return (ndarray): coeficientes del modelo de regresion en orden:
        constante, b1, b2, b3, ..., bn
        """
        df = pd.DataFrame(series)
        if intercept:
            X = sm.add_constant(df[df.columns[1:]])
        else:
            X = df[df.columns[1:]]

        y = df[df.columns[0]]
        model = sm.OLS(y, X).fit()
        coefs_results = model.params.values

        return coefs_results

    def aux_QueEsEsto(self, serie, recursive=False, count=1):
        """Funcion auxiliar para manejar tres tipos de datos de entrada

        Parametros:
        serie (xr.Dataset, xr.DataArray, np.ndarray o lista de estas variables):
        series temporales
        recursive (bool): True, cuenta las iteraciones
        count (int): numero de iteraciones

        return (np.ndarray) valores de cada serie
        si recursive = True ademas devuelve el nombre que se le asigna a la
        serie. En el caso de ser xr.Dataset usa el nombre de la variable
        """
        resultados = []

        if isinstance(serie, list):
            for elem in serie:
                sub_resultados, count = self.aux_QueEsEsto(elem, True, count)
                resultados.extend(sub_resultados)

        elif isinstance(serie, xr.Dataset):
            name = list(serie.keys())[0]
            resultados.append((f'{name}_{count}', serie[name].values))
            count += 1

        elif isinstance(serie, xr.DataArray):
            name = serie.name if serie.name is not None else 'coef'
            resultados.append((f'{name}_{count}', serie.values))
            count += 1

        elif isinstance(serie, np.ndarray):
            resultados.append((f'coef_{count}', serie))
            count += 1

        else:
            resultados.append((None, None))

        if recursive:
            return resultados, count
        else:
            return resultados

    def make_series(self, serie_target):
        """Incorpora a "series" a partir de "serie_target" y "predictores"

        Parametros:
        serie_target (np.ndarray): serie temporal, predictando

        return (dict): diccionario con nombre y datos de las series temporales
        """
        name_serie_target, data_serie_target = (
            self.aux_QueEsEsto(serie_target))[0]

        len_serie_target = len(data_serie_target)

        series = {name_serie_target: data_serie_target}

        # predictores
        name_n_data_predictores = self.aux_QueEsEsto(self.predictores)

        len_ok = True  # control de longitudes de las series
        for e in name_n_data_predictores:
            if len(e[1]) == len_serie_target:
                series[e[0]] = e[1]
            else:
                len_ok = False

        return series, len_ok

    def pre_mlr_1d(self, serie_target, intercept=True):
        """
        Función auxiliar para usar mlr_1d() desde compute_MLR()

        Parametros
        serie_target (np.ndarray): serie temporal, predictando
        intercept (bool): True, ordenada al orgine (recomendado usar siempre)
        """
        series, len_ok = self.make_series(serie_target)

        if len_ok:
            result = self.mlr_1d(series, intercept)
        else:
            result = 0

        return result

    def compute_MLR(self, xrda, intercept=True):
        """
        Aplica mlr_1d a cada punto de grilla de un xr.DataArray utilizando
        "predictores".
        Esta forma es mas eficiente que el uso de ciclos para recorrer todas
        las dimensiones del array.

        Parametros:
        xrda (xr.DataArray): array dim [time, lon, lat] o [time, lon, lat, r]
        (no importa el orden, cuanto mas dimensiones mas tiempo va tardar)
        intercept (bool): True, ordenada al orgine (recomendado usar siempre)

        return (xr.DataArray): array de dimensiones [lon, lat, coef.] o
        [r, lon, lat, coef] donde cada valor en "coef" es un coeficiente del
        modelo lineal en cada punto de grilla en orden: constante, b1, b2..., bn
        """

        # El calculo se hace acá --------------------------------------------- #
        coef_dataset = xr.apply_ufunc(
            self.pre_mlr_1d, xrda, intercept,
            input_core_dims=[['time'], []],
            output_core_dims=[[]],
            output_dtypes=[list],
            vectorize=True)
        # -------------------------------------------------------------------- #

        # Acomodando los datos en funcion de si existe o no la dimension 'r'
        if "r" in coef_dataset.dims:

            aux = np.array(
                [[[np.array(item) for item in row] for row in r_slice] for
                 r_slice in coef_dataset.values]
            )

            coef_f = xr.DataArray(
                data=aux,
                dims=["r", "lat", "lon", "coef"],
                coords={"r": coef_dataset.r.values,
                        "lat": coef_dataset.lat.values,
                        "lon": coef_dataset.lon.values,
                        "coef": list(self.make_series(xrda)[0].keys())}
            )

        else:
            aux = np.array(
                [[np.array(item) for item in row] for row in
                 coef_dataset.values])

            coef_f = xr.DataArray(
                data=aux,
                dims=["lat", "lon", "coef"],
                coords={"lat": coef_dataset.lat.values,
                        "lon": coef_dataset.lon.values,
                        "coef":
                            list(self.make_series(xrda)[0].keys())}
            )

        return coef_f
# ---------------------------------------------------------------------------- #
def Compute_MLR(predictando=None, mes_predictando=None,
                predictores=None, meses_predictores=None,
                predictando_trimestral = True,
                predictores_trimestral=False,
                anios_predictores = None):
    """
    Computa MLR en todos los puntos de grilla de un predictando  a partir
    series temporales de predictores.

    predictando (xr.dataset)
    mes_predictando (int): numero del mes que se va usar en predictando
    predictores (list): lista de predictores
    mes_predictores (list): lista de int de meses para CADA predictor
    predictando_trimestral (bool): True, promedio movil de 3 meses
    predictores_trimestral (bool): True, promedio movil de 3 meses
    anios_predictores (list): lista de anios de predictores

    return:
    output0 (xr.dataset): coeficientes de regresion
    output1 (xr.dataset): coeficientes de regresion en anomalias
    output2 (xr.dataset): reconstruccion del campo original a partir de MLR
    output3 (xr.dataset): campo original en el mismo periodo que output2
    """
    output0 = None
    output1 = None
    output2 = None
    output3 = None

    if (isinstance(predictando, xr.Dataset)
            and isinstance(predictores, list)
            and isinstance(mes_predictando, int)
            and isinstance(meses_predictores, list)):

        variable_predictando = list(predictando.data_vars)[0]

        predictores_set = []

        iter = zip(predictores, meses_predictores,
                   range(len(predictores)))
        check_anios_predictores = False
        if anios_predictores is not None:
            if (isinstance(anios_predictores, list) and
                    len(anios_predictores) == len(predictores)):
                iter = zip(predictores, meses_predictores, anios_predictores)
                check_anios_predictores = True
            else:
                print('Error: "anios_predictores debe ser una lista con tantos '
                      'iterables como predictores')
                print(
                    'Los predictores se setaron con los anios del predictando')

        for p, mp, ap in iter:
            if predictores_trimestral:
                p = p.rolling(time=3, center=True).mean()

            if check_anios_predictores:
                aux_p = p.sel(
                    time=p.time.dt.year.isin(ap))
            else:
                aux_p = p.sel(
                    time=p.time.dt.year.isin(predictando.time.dt.year))

            aux_p = aux_p.sel(time=aux_p.time.dt.month.isin(mp))

            predictores_set.append(aux_p / aux_p.std())

        if predictando_trimestral:
            predictando = predictando.rolling(time=3, center=True).mean()

        predictando = predictando.sel(
            time=predictando.time.dt.month.isin(mes_predictando))

        #promedio_mes_predictando = predictando.mean('time')

        predictando_anom, promedio_mes_predictando, predictando_sd = (
            CrossAnomaly_1y(predictando, norm=True, return_mean_sd=True))

        mlr = MLR(predictores_set)

        regre = mlr.compute_MLR(predictando_anom[variable_predictando])

        output0 = ((regre*predictando_sd) +
                   promedio_mes_predictando[variable_predictando])
        output1 = regre

        reconstruc = MLR_pronostico(data=predictando_anom,
                                    regre_result=regre,
                                    predictores=predictores_set)[1]

        reconstruc_anom = (reconstruc*predictando_sd)

        reconstruc = reconstruc_anom + promedio_mes_predictando

        output2 = reconstruc
        output3 = predictando
    else:
        print('Error en formato de variables de entradas')

    return output0, output1, output2, output3


def Compute_MLR_training_testing(predictando=None, mes_predictando=None,
                                 predictores=None, meses_predictores=None,
                                 anios_training=[1983, 2010],
                                 anios_testing=[2011, 2020],
                                 predictando_trimestral = True,
                                 predictores_trimestral=False,
                                 anios_predictores_testing = None,
                                 anios_predictores_training = None,
                                 reconstruct_full = False):
    """
    Computa MLR en el periodo de testing en todos los puntos de grilla
    de un predictando a partir series temporales de predictores en training s

    predictando (xr.dataset)
    mes_predictando (int): numero del mes que se va usar en predictando
    predictores (list): lista de predictores
    mes_predictores (list): lista de int de meses para CADA predictor
    anios_training (list): extremos del periodo de anios testing
    anios_testing (list): extremos del periodo de anios testing
    predictando_trimestral (bool): si es True, promedio movil de 3 meses
    predictores_trimestral (bool): si es True, promedio movil de 3 meses
    anios_predictores (list): lista de anios de predictores
    reconstruct_full (bool): si es True, reconstruccion del periodo completo.
    usando training para reconstruir testing y viceversa

    return:
    output0 (xr.dataset): coeficientes de regresion
    output1 (xr.dataset): coeficientes de regresion en anomalias
    output2 (xr.dataset): reconstruccion del campo original a partir de MLR
    output3 (xr.dataset): campo original en el mismo periodo que output2
    """
    output0 = None
    output1 = None
    output2 = None
    output3 = None

    if (isinstance(predictando, xr.Dataset)
            and isinstance(predictores, list)
            and isinstance(mes_predictando, int)
            and isinstance(meses_predictores, list)
            and isinstance(anios_training, list)
            and isinstance(anios_testing, list)):

        variable_predictando = list(predictando.data_vars)[0]

        year_training = np.arange(anios_training[0], anios_training[-1]+1)
        year_testing = np.arange(anios_testing[0], anios_testing[-1]+1)

        iter = zip(predictores, meses_predictores,
                   [year_training, year_training, year_training],
                   [year_testing, year_testing, year_testing])

        chek_anios_predictores_training = False
        if (anios_predictores_testing is not None
            and anios_predictores_training is not None):
            if isinstance(anios_predictores_training, list) \
                and isinstance(anios_predictores_testing, list):

                aux_year_training = []
                aux_year_testing = []
                for atr, att in zip(anios_predictores_training,
                                    anios_predictores_testing):

                    aux_year_training.append(np.arange(atr[0], atr[-1]+1))
                    aux_year_testing.append(np.arange(att[0], att[-1]+1))

                iter = zip(predictores, meses_predictores,
                           aux_year_training,
                           aux_year_testing)
                chek_anios_predictores_training = True
            else:
                print(
                    'Error: anios_predictores_testing/training deben ser listas')

        predictores_training = []
        predictores_testing = []
        for p, mp, trp, ttp in iter:
            if predictores_trimestral:
                p = p.rolling(time=3, center=True).mean()

            aux_p = p.sel(time=p.time.dt.month.isin(mp))
            aux_p_training = aux_p.sel(time=aux_p.time.dt.year.isin(trp))
            aux_p_testing = aux_p.sel(time=aux_p.time.dt.year.isin(ttp))

            predictores_training.append(aux_p_training / aux_p_training.std())
            predictores_testing.append(aux_p_testing / aux_p_testing.std())

        if predictando_trimestral is True:
            predictando = predictando.rolling(time=3, center=True).mean()

        # Training
        predictando_training = predictando.sel(
            time=predictando.time.dt.year.isin(year_training))
        predictando_training = predictando_training.sel(
            time=predictando_training.time.dt.month.isin(mes_predictando))

        predictando_training_anom, promedio_mes_predictando_training, \
            sd_mes_predictando_training = CrossAnomaly_1y(
            predictando_training, norm=True, return_mean_sd=True)

        # Testing
        predictando_testing = predictando.sel(
            time=predictando.time.dt.year.isin(year_testing))
        predictando_testing = predictando_testing.sel(
            time=predictando_testing.time.dt.month.isin(mes_predictando))

        predictando_testing_anom, promedio_mes_predictando_testing, \
            sd_mes_predictando_testing = CrossAnomaly_1y(
            predictando_testing, norm=True, return_mean_sd=True)

        mlr = MLR(predictores_training)

        training_regre =  mlr.compute_MLR(
            predictando_training_anom[variable_predictando])

        training_regre_full = ((training_regre*sd_mes_predictando_training) +
                               promedio_mes_predictando_training[
                                   variable_predictando])

        testing_reconstruc = MLR_pronostico(predictando_testing_anom,
                                            training_regre,
                                            predictores_testing)[1]

        testing_reconstruc_anom = testing_reconstruc * \
                                  sd_mes_predictando_testing

        testing_reconstruc_full = testing_reconstruc_anom + \
                                  promedio_mes_predictando_testing


        if reconstruct_full:
            # MLR
            mlr = MLR(predictores_testing)
            testing_regre = mlr.compute_MLR(
                predictando_testing_anom[variable_predictando])

            testing_regre_full = ((testing_regre*sd_mes_predictando_testing) +
                                  promedio_mes_predictando_testing[
                                      variable_predictando])

            training_reconstruc = MLR_pronostico(predictando_training_anom,
                                                 testing_regre,
                                                 predictores_training)[1]

            training_reconstruc_anom = training_reconstruc * \
                                       sd_mes_predictando_training

            training_reconstruc_full = training_reconstruc_anom + \
                                        promedio_mes_predictando_training

            forecast_full = xr.concat([training_reconstruc_full,
                                       testing_reconstruc_full], dim = 'time')

            data_to_verif = xr.concat([predictando_training,
                                       predictando_testing], dim= 'time')

            output0 = (training_regre_full + testing_regre_full)/2
            output1 = (training_regre + testing_regre)/2
            output2 = forecast_full
            output3 = data_to_verif

        else:
            output0 = training_regre_full
            output1 = training_regre
            output2 = testing_reconstruc_full
            output3 = predictando_testing

    else:
        print('Error en formato de variables de entradas')

    return output0, output1, output2, output3

# ---------------------------------------------------------------------------- #
def Compute_MLR_CV(predictando, mes_predictando=None,
                   predictores=None, meses_predictores=None,
                   predictando_trimestral=True,
                   predictores_trimestral=False,
                   anios_predictores=None,
                   window_years=3):
    """
    Computa MLR con validacion cruzada en todos los puntos de grilla
    de un predictando a partir series temporales de predictores

    predictando (xr.dataset)
    mes_predictando (int): numero del mes que se va usar en predictando
    predictores (list): lista de predictores
    mes_predictores (list): lista de int de meses para CADA predictor
    predictando_trimestral (bool): si es True, promedio movil de 3 meses
    predictores_trimestral (bool): si es True, promedio movil de 3 meses
    anios_predictores (list): lista de anios de predictores
    window_years (int): ventana de anios a usar para la validacion cruzada

    return:
    output0 (xr.dataset): coeficientes de regresion
    output1 (xr.dataset): coeficientes de regresion en anomalias
    output2 (xr.dataset): reconstruccion del campo original a partir de MLR
    output3 (xr.dataset): campo original en el mismo periodo que output2
    """

    intercept = True
    xrds = predictando
    if (isinstance(predictando, xr.Dataset)
            and isinstance(predictores, list)
            and isinstance(window_years, int)
            and isinstance(intercept, bool)
            and isinstance(mes_predictando, int)
            and isinstance(meses_predictores, list)):

        predictores_years_out = {}
        variable_predictando = list(predictando.data_vars)[0]

        # Seteo de tiempos y meses igual a Compute_MLR
        iter = zip(predictores, meses_predictores,
                   range(len(predictores)))
        check_anios_predictores = False
        if anios_predictores is not None:
            if (isinstance(anios_predictores, list) and
                    len(anios_predictores) == len(predictores)):
                iter = zip(predictores, meses_predictores, anios_predictores)
                check_anios_predictores = True
            else:
                print('Error: "anios_predictores debe ser una lista con tantos '
                      'iterables como predictores')
                print(
                    'Los predictores se setaron con los anios del predictando')

        predictores_set = []
        for p, mp, ap in iter:
            if predictores_trimestral:
                p = p.rolling(time=3, center=True).mean()

            if check_anios_predictores:
                aux_p = p.sel(
                    time=p.time.dt.year.isin(ap))
            else:
                aux_p = p.sel(
                    time=p.time.dt.year.isin(predictando.time.dt.year))

            aux_p = aux_p.sel(time=aux_p.time.dt.month.isin(mp))

            predictores_set.append(aux_p)

        if predictando_trimestral:
            predictando = predictando.rolling(time=3, center=True).mean()

        predictando = predictando.sel(
            time=predictando.time.dt.month.isin(mes_predictando))

        # CV ----------------------------------------------------------------- #
        total_tiempos = len(predictando.time)
        for i in range(total_tiempos - window_years + 1):

            per10 = 10 * round((i / total_tiempos) * 10)
            if i == 0 or per10 != 10 * round(((i - 1) / total_tiempos) * 10):
                print(f"{per10}% completado")

            # posicion de tiempos a omitir
            tiempos_a_omitir = range(i, i + window_years)

            # selección de tiempos
            ds_cv = predictando.drop_sel(
                time=predictando.time.values[
                     tiempos_a_omitir[0]:tiempos_a_omitir[-1] + 1])
            ds_out_years = predictando.drop_sel(time=ds_cv.time.values)

            if 'r' in ds_cv.dims:
                ds_cv_media = ds_cv.mean(['r', 'time'], skipna=True)
                ds_cv_std =  ds_cv.std(['r', 'time'], skipna=True)
                ds_cv = ds_cv - ds_cv_media
                ds_cv = ds_cv / ds_cv_std

                ds_out_years = ds_out_years - ds_cv_media
                ds_out_years = ds_out_years / ds_cv_std
            else:
                ds_cv_media = ds_cv.mean('time', skipna=True)
                ds_cv_std = ds_cv.std('time', skipna=True)
                ds_cv = ds_cv - ds_cv_media
                ds_cv = ds_cv / ds_cv_std

                ds_out_years = ds_out_years - ds_cv_std
                ds_out_years = ds_out_years / ds_cv_std

            predictores_cv = []
            aux_predictores_out_years = []
            for p in predictores_set:
                aux_predictor = p.drop_sel(
                    time=p.time.values[
                         tiempos_a_omitir[0]:tiempos_a_omitir[-1] + 1])
                aux_p = p.drop_sel(time=aux_predictor.time.values)
                aux_p = aux_p - aux_predictor.mean('time', skipna=True)
                aux_p = aux_p / aux_predictor.std('time', skipna=True)
                aux_predictores_out_years.append(aux_p)

                predictores_cv.append(aux_predictor)

            predictores_years_out[f"k{i}"] = aux_predictores_out_years

            # MLR
            mlr = MLR(predictores_cv)
            regre_result = mlr.compute_MLR(ds_cv[list(ds_cv.data_vars)[0]],
                                           intercept=intercept)

            regre_result_full = (regre_result*ds_cv_std) + ds_cv_media

            years_out_reconstruct = MLR_pronostico(ds_out_years,
                                                   regre_result,
                                                   predictores_years_out[f"k{i}"])[1]

            years_out_reconstruct = years_out_reconstruct.sel(
                time=years_out_reconstruct.time.values[int(window_years / 2)])

            years_out_reconstruct_full = (years_out_reconstruct*ds_cv_std) + \
                                         ds_cv_media

            if i == 0:
                regre_result_cv = regre_result
                regre_result_full_cv = regre_result_full
                # ds_out_years_cv = ds_out_years
                years_out_reconstruct_cv = years_out_reconstruct
                years_out_reconstruct_full_cv = years_out_reconstruct_full
            else:
                regre_result_cv = xr.concat([regre_result_cv, regre_result],
                                            dim='k')

                regre_result_full_cv = xr.concat([regre_result_full_cv,
                                                  regre_result_full], dim='k')
                # ds_out_years_cv = xr.concat([ds_out_years_cv, ds_out_years],
                #                             dim='k')

                years_out_reconstruct_cv = xr.concat([years_out_reconstruct_cv,
                                                      years_out_reconstruct],
                                                     dim='time')

                years_out_reconstruct_full_cv = (
                    xr.concat([years_out_reconstruct_full_cv,
                               years_out_reconstruct_full], dim='time'))

    regre_result_cv = regre_result_cv.mean('k')
    regre_result_full_cv = regre_result_full_cv.mean('k')
    predictando = predictando.sel(time=years_out_reconstruct_full_cv.time.values)

    return regre_result_cv, regre_result_full_cv, \
        years_out_reconstruct_full_cv, predictando


def MLR_pronostico(data, regre_result, predictores):

    """
    FUNCION INTERNA para reconstruir campos a partir de salidas de MLR
    """

    compute = True
    if isinstance(predictores, dict): # salida de CV
        len_time_predictores = len(predictores)
        # Como tomamos ventanas de años (tiempos) nos van a quedar un numero
        # de años sin considerar en cada extremo. Si la ventana fue de 3 años,
        # debemos sacar uno de cada lado
        remove_time_ext = int((len(data.time) - len_time_predictores) / 2)
        data = data.isel(time=slice(remove_time_ext, -remove_time_ext))
    elif isinstance(predictores, list): # Salida comun
        if len(predictores[0].time.values) == len(data.time.values):
            pass
        else:
            print('Error: "data" debe tener la misma CANTIDAD de tiempos que '
                  'los predictores')
            predict_da = None
            compute = False
    else:
        print('ERROR: predictores debe ser list o dict')
        predict_da = None
        compute = False

    var_name = list(data.data_vars)[0]

    if compute is True:
        if 'k' in regre_result.dims:
            pos_med = int((len(predictores['k0'][0].time) - 1) / 2)
            predict_k = []
            for k_int, k in enumerate(predictores.keys()):
                coefmodel = regre_result.sel(k=k_int)
                predictores_name = list(coefmodel.coef.values)

                indices = predictores[k]

                # modelo
                predict = coefmodel.sel(coef=predictores_name[0]).drop_vars(
                    'coef')

                for c, p in enumerate(indices):
                    p = p.sel(time=p.time[pos_med])
                    aux = coefmodel.sel(coef=predictores_name[c + 1]) * p[
                        list(p.data_vars)[0]]
                    aux = aux.drop_vars('coef')
                    aux = aux.drop_vars('time')
                    predict = predict + aux

                predict_k.append(predict)

            predict_da = xr.concat(predict_k, dim='time')
            predict_da['time'] = data.time.values
            predict_da = predict_da.to_dataset(name=var_name)

        else:
            predictores_name = list(regre_result.coef.values)
            predict = regre_result.sel(coef=predictores_name[0]).drop_vars(
                'coef')
            for c, p in enumerate(predictores):
                # Selección del indice y los valores del predictor en testing
                aux = regre_result.sel(coef=predictores_name[c + 1]) * p[
                    list(p.data_vars)[0]]

                aux = aux.drop_vars('coef')  # no deja sumar sino
                aux = aux.drop_vars('time')

                predict = predict + aux

            predict_da = predict.to_dataset(name=var_name)
            predict_da['time'] = data.time.values

    return data, predict_da

# ---------------------------------------------------------------------------- #
# CCA
# ---------------------------------------------------------------------------- #
def Weights(data, lon_name='lon', lat_name='lat'):
    """
    FUNCION INTERNA

    Normaliza por pesando por el coseno de la latitud

    parametros:
    data xr.DataArray o xr.DataSet
    lon_name str nombre de la dimension longitud
    lat_name str nombre de la dimension latitud

    return
    data xr.DataArray o xr.DataSet pesado
    """

    weights = np.transpose(np.tile(np.cos(data[lat_name] * np.pi / 180),
                                   (len(data[lon_name]), 1)))
    data_w = data * weights
    return data_w


def normalize_and_fill(data):
    """
    FUNCION INTERNA

    Normaliza los datos y rellena los NaN con el valor medio.

    Parameters:
    data (xarray.DataArray/xarray.DataSet): El campo de datos a normalizar.

    Returns:
    xarray.DataArray: Datos normalizados y sin NaNs.
    """
    try:
        data = data[list(data.data_vars)[-1]]
    except:
        pass

    lon_name = next((name for name in ['lon', 'longitude', 'longitud']
                     if name in data.dims), None)
    lat_name = next((name for name in ['lat', 'latitude', 'latitud']
                     if name in data.dims), None)

    data = Weights(data, lon_name=lon_name, lat_name=lat_name)

    # Calcular el valor medio ignorando NaNs
    mean_val = data.mean(dim='time', skipna=True)
    data_filled = data.fillna(mean_val)

    # Normaliza
    mean = data_filled.mean(dim='time')
    std = data_filled.std(dim='time')
    var = data_filled.var(dim='time')

    data_normalized = (data_filled - mean) / std

    # Aveces aparecen NaN otra vez despues del paso anterior
    data_normalized = data_normalized.fillna(0)

    # Transformamos la data a (tiempo, spatial) donde spatial = lat*lon
    data_normalized = data_normalized.stack(spatial=(lat_name, lon_name))

    return data_normalized, mean, std, var


def AutoVec_Val_EOF(m, var_exp):
    """
    FUNCION INTERNA

    Calcula los autovectores y autovalores de la matriz m.
    Si sklearn está disponible, usa PCA para calcular los valores propios
    explicados hasta var_exp. Si no, usa numpy (mas lento).

    Parameters:
    - m: xarray.DataArray o numpy.array con los datos de entrada.
    - var_exp: Proporción de varianza explicada deseada (en sklearn) o el número
      de componentes deseado en el cálculo de numpy.

    Returns:
    - eigvecs: Autovectores de m.
    - eigvals: Autovalores de m.
    """

    sklearn_check = False
    try:
        from sklearn.decomposition import PCA
        sklearn_check = True
    except ImportError:
        pass

    if sklearn_check:
        pca = PCA(n_components=var_exp, svd_solver='full')
        pca.fit(m.values)

        eigvals = pca.explained_variance_
        eigvecs = pca.components_.T

    else:
        print('Sklearn no encontrado')
        print('Se utilizará numpy: es equivalente pero mucho mas lento para'
              ' dominios grandes')
        print(
            'Algunas componentes pueden diferir en signo entre numpy y sklearn')

        try:
            cov_matrix = np.cov(m.values, rowvar=False)
        except:
            cov_matrix = np.cov(m, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # ordenando de mayor a menor
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Selecciona componentes hasta completar var_exp de varianza
        if isinstance(var_exp, float) and 0 < var_exp <= 1:
            cumulative_var = np.cumsum(eigvals) / np.sum(eigvals)
            n_components = np.searchsorted(cumulative_var, var_exp) + 1
        else:
            n_components = var_exp

        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    return eigvecs, eigvals


def CCA(X, Y, var_exp=0.7, dataarray_pq_outputs=False):
    """
    FUNCION INTERNA

    Correlación canonica entre X e Y
    - Normaliza X, Y
    - prefilstra con EOF el numero de cp para explicar var_exp
    - CCA

    Parametros:
    X (xr.DataArray o Xr.DataSet), de dimension (t, x)
    Y /xr.DataArray o Yr.DataSet), de dimension (t, y)
    var_exp (float) 0<var_exp<=1, default = 0.7 varianza que se quiere retener.
    dataarray_pq_outputs (bool): si es True convierte la salida de CCA xr.dataarray

    return (pueden tener varios nombres...)
    todos numpy.ndarray
    P: mapas canonicos de X
    Q: mapas canonicos de Y
    heP: varianza heterogenea
    heQ: varianza heterogena
    S: valores singulares/correlacion entre A y B
    A: vectores canonicos de X
    B: vectores canonicos de Y
    """
    X_lat = X.lat.values
    X_lon = X.lon.values


    Y_lat = Y.lat.values
    Y_lon = Y.lon.values

    # Normalizado
    X, X_mean, X_std, X_var = normalize_and_fill(X)
    Y, Y_mean, Y_std, X_var = normalize_and_fill(Y)

    tx = X.shape[0]
    ty = Y.shape[0]

    if tx != ty:
        print('La dimensión tiempo de X e Y tiene que tener la misma longitud')
        P = Q = heP = heQ = S = A = B = None

    else:

        if var_exp > 1:
            var_exp = var_exp / 100

        # Eof
        Vx, ux = AutoVec_Val_EOF(X, var_exp)
        Vy, uy = AutoVec_Val_EOF(Y, var_exp)

        try:
            X = X.values @ Vx
        except:
            X = X @ Vx
        X = X - np.mean(X, axis=0)
        X = X / np.sqrt(ux)

        try:
            Y = Y.values @ Vy
        except:
            Y = Y @ Vy
        Y = Y - np.mean(Y, axis=0)
        Y = Y / np.sqrt(uy)

        # CCA
        Cxy = (X.T @ Y) / (X.shape[0] - 1)

        U, S, V = np.linalg.svd(Cxy, full_matrices=False)

        P = (Vx * np.sqrt(ux)) @ U
        Q = (Vy * np.sqrt(uy)) @ V.T

        A = X @ U
        B = Y @ V.T

        heP = P * S
        heQ = Q * S

        if dataarray_pq_outputs is True:
            aux = P.reshape(len(X_lat), len(X_lon),
                            P.shape[1])
            P = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                             coords={'lat': X_lat,
                                     'lon': X_lon})

            aux = Q.reshape(len(Y_lat), len(Y_lon),
                            Q.shape[1])
            Q = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                             coords={'lat': Y_lat,
                                     'lon': Y_lon})

    return P, Q, heP, heQ, S, A, B


def Compute_CCA(X, Y, var_exp=0.7, X_mes=None, Y_mes=None,
                X_trimestral=False, Y_trimestral=False,
                X_anios=None, Y_anios=None):

    """
    Computo basico de CCA:

    Parametros:
    X (xr.DataArray)
    Y /xr.DataArray)
    var_exp (float): 0<var_exp<=1, default = 0.7 varianza que se quiere retener.
    X_mes (int): mes de X a usar
    Y_mes (int): mes de Y a usar
    X_trimestral (bool): si es True toma promedio movil de 3 meses
    Y_trimestral (bool): si es True toma promedio movil de 3 meses
    X_anios (list): extremos de anios para periodo de X
    Y_anios (list): extremos de anios para periodo de X

    Returns:
    P mapa canonico de X
    Q mapa canonico de X
    S valores singulares/corPrelacion entre A y B
    A vector canonico de X
    B vector canonico de Y
    """

    compute = True
    if X_anios is None and Y_anios is None:
        print('X_anios, Y_anios is None')
        print('Se usará el periodo de anios en común mas largo entre X e Y')

        if len(X.time.values) < len(Y.time.values):
            X_anios_values = np.unique(X.time.dt.year.values)
            Y_anios_values = X_anios_values

        elif len(Y.time.values) < len(X.time.values):
            X_anios_values = np.unique(Y.time.dt.year.values)
            Y_anios_values = X_anios_values
    elif X_anios is not None and Y_anios is not None:
        X_anios_values = np.arange(X_anios[0], X_anios[-1] + 1)
        Y_anios_values = np.arange(Y_anios[0], Y_anios[-1] + 1)
        if len(X_anios_values) != len(Y_anios_values):
            print('El rango de X_anios, Y_anios debe contener la misma '
                  'cantidad de años')
            compute = False
            P_da, Q_da, A, B, S = None
    else:
        print('X_anios, Y_anios deben ser None o list, ambos.')
        compute = False
        P_da, Q_da, A, B, S = None


    if compute is True:
        # Set data
        if X_trimestral is True:
            X = X.rolling(time=3, center=True).mean('time')
            # Si se hace promedio movil perdemos los extremos y CCA no admite NaNs
            X = X.sel(time=X.time.isin(X.time.values[1:-1]))

        if Y_trimestral is True:
            Y = Y.rolling(time=3, center=True).mean('time')
            Y = Y.sel(time=Y.time.isin(Y.time.values[1:-1]))

        X = X.sel(time=X.time.dt.year.isin(X_anios_values))
        X = X.sel(time=X.time.dt.month.isin(X_mes))

        Y = Y.sel(time=Y.time.dt.year.isin(Y_anios_values))
        Y = Y.sel(time=Y.time.dt.month.isin(Y_mes))

        P, Q, heP, heQ, S, A, B = CCA(X=X, Y=Y, var_exp=var_exp,
                                      dataarray_pq_outputs=True)

        modos = np.arange(1, len(P.cca)+1)
        P = P.assign_coords({'cca': modos})
        P = P.rename({'cca':'modo'})

        Q = Q.assign_coords({'cca': modos})
        Q = Q.rename({'cca':'modo'})

    return P, Q, A, B, S


def CCA_mod(X, X_test, Y, var_exp=0.7):
    """
    FUNCION INTERNA

    Reconstruye el predictando Y a partir de X en X_test

    Parametros:
    X xr.DataArray o Xr.DataSet, de dimension (t, x)
    Y xr.DataArray o Yr.DataSet, de dimension (t, y)
    var_exp float, default 0.7 varianza que se quiere retener.

    return:
    ajd numpy.ndarray dim=(tiempos_reconstruidos, puntos_reticula, modos)
    B_verif np.ndarray vector canonico para la reconstruccion
    """

    P, Q, heP, heQ, S, A, B = CCA(X, Y, var_exp)

    # Normalizando
    X_training = normalize_and_fill(X)[0]
    X_verif = normalize_and_fill(X_test)[0]

    Y_training = normalize_and_fill(Y)[0]

    # EOF
    Vx, ux = AutoVec_Val_EOF(X_training, P.shape[1])
    Vy, uy = AutoVec_Val_EOF(Y_training, Q.shape[1])

    Tx = X_training.values @ Vx
    Tx = Tx - np.mean(Tx, axis=0)
    Tx = Tx / np.sqrt(ux)

    Txx = X_verif.values @ Vx
    Txx = Txx - np.mean(Tx, axis=0)
    Txx = Txx / np.sqrt(ux)

    Ty = Y_training.values @ Vy
    Ty = Ty - np.mean(Ty, axis=0)
    Ty = Ty / np.sqrt(ux)

    # CCA
    Cxy = (Tx.T @ Ty) / (Tx.shape[0] - 1)
    U, S, V = np.linalg.svd(Cxy, full_matrices=False)

    VV = (Vy * np.sqrt(uy)) @ V.T

    # Esto se usa para algo?
    # UU = (Vx * np.sqrt(ux)) @ U
    # A = Tx @ U
    # B = Ty @ V.T

    B_verif = Txx @ U

    # Acomodando los campos resultantes de forma practica
    adj = np.zeros((B_verif.shape[0], VV.shape[0], B_verif.shape[1]))
    aux = B_verif * S

    for year in range(B_verif.shape[0]):
        for mod in range(B_verif.shape[1]):
            adj[year, :, mod] = np.outer(aux[year, mod], VV[:, mod])

    return adj, B_verif


def CCA_training_testing(X, Y, var_exp,
                         X_mes=None, Y_mes=None,
                         X_trimestral=False, Y_trimestral=False,
                         X_anios=None, Y_anios=None,
                         anios_training=[1983, 2010],
                         anios_testing=[2011, 2020],
                         reconstruct_full = False):
    """
    CCA con training y testing

    Parametros:
    X (xr.DataArray)
    Y /xr.DataArray)
    var_exp (float): 0<var_exp<=1, default = 0.7 varianza que se quiere retener.
    X_mes (int): mes de X a usar
    Y_mes (int): mes de Y a usar
    X_trimestral (bool): si es True toma promedio movil de 3 meses
    Y_trimestral (bool): si es True toma promedio movil de 3 meses
    X_anios (list): extremos de anios para periodo de X
    Y_anios (list): extremos de anios para periodo de X
    anios_training (list): extremos de anios para periodo de training
    anios_testing (list): extremos de anios para periodo de testing
    reconstruct_full (bool): si es True, reconstruccion del periodo completo.
    usando training para reconstruir testing y viceversa

    return:
    output0 (xr.Dataset): reconstruccion a partir de CCA
    outout1 (xr.Dataset): Y original en el periodo de output0
    """
    compute = True
    if X_anios is None and Y_anios is None:
        print('X_anios, Y_anios is None')
        print('Se usará el periodo de anios en común mas largo entre X e Y')

        if len(X.time.values) < len(Y.time.values):
            X_anios_values = np.unique(X.time.dt.year.values)
            Y_anios_values = X_anios_values

        elif len(Y.time.values) < len(X.time.values):
            X_anios_values = np.unique(Y.time.dt.year.values)
            Y_anios_values = X_anios_values

    elif X_anios is not None and Y_anios is not None:
        X_anios_values = np.arange(X_anios[0], X_anios[-1] + 1)
        Y_anios_values = np.arange(Y_anios[0], Y_anios[-1] + 1)

        if len(X_anios_values) != len(Y_anios_values):
            print('El rango de X_anios, Y_anios debe contener la misma '
                  'cantidad de años')
            compute = False
    else:
        print('X_anios, Y_anios deben ser None o list, ambos.')
        compute = False

    if compute is True:
        plus_anio = np.abs(X_anios_values[-1]-Y_anios_values[-1])
        # Set data
        if X_trimestral is True:
            X = X.rolling(time=3, center=True).mean('time')
            # Si se hace promedio movil perdemos los extremos y CCA no admite NaNs
            X = X.sel(time=X.time.isin(X.time.values[1:-1]))

        if Y_trimestral is True:
            Y = Y.rolling(time=3, center=True).mean('time')
            Y = Y.sel(time=Y.time.isin(Y.time.values[1:-1]))

        X = X.sel(time=X.time.dt.year.isin(X_anios_values))
        X = X.sel(time=X.time.dt.month.isin(X_mes))

        Y = Y.sel(time=Y.time.dt.year.isin(Y_anios_values))
        Y = Y.sel(time=Y.time.dt.month.isin(Y_mes))

        # Training - Testing
        anios_training = np.arange(anios_training[0], anios_training[-1] + 1)
        anios_testing = np.arange(anios_testing[0], anios_testing[-1] + 1)

        if plus_anio > 0:
            if anios_testing[-1] > Y_anios[-1]:
                anios_testing = anios_testing[0:-1]

        if reconstruct_full is True:
            iter = [[anios_training, anios_testing],
                    [anios_testing, anios_training]]
        else:
            iter = [[anios_training, anios_testing]]

        for it_n, it in enumerate(iter):
            atr = it[0]
            att = it[1]

            X_anios_training = atr
            X_anios_testing = att

            Y_anios_training = atr + plus_anio
            Y_anios_testing = att + plus_anio


            X_training = X.sel(time=X.time.dt.year.isin(X_anios_training))
            X_testing = X.sel(time=X.time.dt.year.isin(X_anios_testing))

            Y_training = Y.sel(time=Y.time.dt.year.isin(Y_anios_training))
            Y_testing = Y.sel(time=Y.time.dt.year.isin(Y_anios_testing))

            adj, b_verif = CCA_mod(X=X_training, X_test=X_testing,
                                   Y=Y_training, var_exp=var_exp)

            adj_rs = adj.reshape(adj.shape[0],
                                 len(Y.lat.values),
                                 len(Y.lon.values),
                                 adj.shape[2])

            adj_xr = xr.DataArray(adj_rs,
                                  dims=['time', 'lat', 'lon', 'modo'],
                                  coords={'lat': Y.lat.values,
                                          'lon': Y.lon.values,
                                          'time': X_testing.time.values,
                                          'modo': np.arange(0,

                                                           adj_rs.shape[-1])})

            Y_training_mean, Y_training_std, Y_training_var = \
                normalize_and_fill(Y_training)[1:4]

            adj_total = adj_xr.sum('modo')

            try:
                adj_std = adj_total.std('time')
            except:
                adj_std = adj_xr.std()

            adj_xr = (adj_total * Y_training_std / adj_std) + \
                     Y_training_mean

            mod_adj = adj_xr.to_dataset(name=list(Y.data_vars)[0])
            mod_adj['time'] = Y_testing.time.values

            data_to_verif = Y_testing# - Y_training.mean('time')

            if it_n == 0:
                adj_f = mod_adj
                data_to_verif_f = data_to_verif
            else:
                adj_f = xr.concat([mod_adj, adj_f], dim='time')
                data_to_verif_f = xr.concat([data_to_verif, data_to_verif_f],
                                            dim='time')
    else:
        adj_f, data_to_verif_f = None

    return adj_f, data_to_verif_f


def CCA_calibracion_training_testing(X_modelo, Y_observacion, var_exp,
                                     Y_mes=None, Y_trimestral=False,
                                     X_anios=None, Y_anios=None,
                                     anios_training=[1983, 2010],
                                     anios_testing=[2011, 2020],
                                     reconstruct_full = False):
    """
    Calibracion CCA con training y testing

    Parametros:
    X_modelo (xr.DataArray): Modelo
    Y_observacion (xr.DataArray)
    var_exp (float): 0<var_exp<=1, default = 0.7 varianza que se quiere retener.
    Y_mes (int): mes de Y a usar
    X_trimestral (bool): si es True toma promedio movil de 3 meses
    Y_trimestral (bool): si es True toma promedio movil de 3 meses
    X_anios (list): extremos de anios para periodo de X
    Y_anios (list): extremos de anios para periodo de X
    anios_training (list): extremos de anios para periodo de training
    anios_testing (list): extremos de anios para periodo de testing
    reconstruct_full (bool): si es True, reconstruccion del periodo completo.
    usando training para reconstruir testing y viceversa

    return:
    output0 (xr.Dataset): Modelo calibrado a partir de CCA
    outout1 (xr.Dataset): Y original en el periodo de output0
    """

    X = X_modelo
    Y = Y_observacion
    compute = True
    if X_anios is None and Y_anios is None:
        print('X_anios, Y_anios is None')
        print('Se usará el periodo de anios en común mas largo entre X e Y')

        if len(X_modelo.time.values) < len(Y.time.values):
            X_anios_values = np.unique(X.time.dt.year.values)
            Y_anios_values = X_anios_values

        elif len(Y.time.values) < len(X_modelo.time.values):
            X_anios_values = np.unique(Y.time.dt.year.values)
            Y_anios_values = X_anios_values

    elif X_anios is not None and Y_anios is not None:
        X_anios_values = np.arange(X_anios[0], X_anios[-1] + 1)
        Y_anios_values = np.arange(Y_anios[0], Y_anios[-1] + 1)

        if len(X_anios_values) != len(Y_anios_values):
            print('El rango de X_anios, Y_anios debe contener la misma '
                  'cantidad de años')
            compute = False
    else:
        print('X_anios, Y_anios deben ser None o list, ambos.')
        compute = False

    if compute is True:
        plus_anio = np.abs(X_anios_values[-1]-Y_anios_values[-1])
        # Set data
        # if X_trimestral is True:
        #     X = X_modelo.rolling(time=3, center=True).mean('time')
        #     # Si se hace promedio movil perdemos los extremos y CCA no admite NaNs
        #     X = X.sel(time=X.time.isin(X.time.values[1:-1]))

        if Y_trimestral is True:
            Y = Y.rolling(time=3, center=True).mean('time')
            Y = Y.sel(time=Y.time.isin(Y.time.values[1:-1]))


        X = X.sel(time=X.time.dt.year.isin(X_anios_values))
        #X = X.sel(time=X.time.dt.month.isin(X_mes))

        Y = Y.sel(time=Y.time.dt.year.isin(Y_anios_values))
        Y = Y.sel(time=Y.time.dt.month.isin(Y_mes))

        # Training - Testing
        anios_training = np.arange(anios_training[0], anios_training[-1] + 1)
        anios_testing = np.arange(anios_testing[0], anios_testing[-1] + 1)

        if plus_anio > 0:
            if anios_testing[-1] > Y_anios[-1]:
                anios_testing = anios_testing[0:-1]

        if reconstruct_full is True:
            iter = [[anios_training, anios_testing],
                    [anios_testing, anios_training]]
        else:
            iter = [[anios_training, anios_testing]]

        for it_n, it in enumerate(iter):
            atr = it[0]
            att = it[1]

            X_anios_training = atr
            X_anios_testing = att

            Y_anios_training = atr + plus_anio
            Y_anios_testing = att + plus_anio


            X_training = X.sel(time=X.time.dt.year.isin(X_anios_training))
            X_testing = X.sel(time=X.time.dt.year.isin(X_anios_testing))

            Y_training = Y.sel(time=Y.time.dt.year.isin(Y_anios_training))
            Y_testing = Y.sel(time=Y.time.dt.year.isin(Y_anios_testing))


            mod_adj = []
            for r in X.r.values:
                adj, b_verif = CCA_mod(X=X_training.mean('r'),
                                       X_test=X_testing.sel(r=r),
                                       Y=Y_training, var_exp=var_exp)

                adj_rs = adj.reshape(adj.shape[0],
                                     len(Y.lat.values),
                                     len(Y.lon.values),
                                     adj.shape[2])

                adj_xr = xr.DataArray(adj_rs,
                                      dims=['time', 'lat', 'lon', 'modo'],
                                      coords={'lat': Y.lat.values,
                                              'lon': Y.lon.values,
                                              'time': X_testing.time.values,
                                              'modo': np.arange(
                                                  0, adj_rs.shape[-1])})

                # sumamos todos los modos y escalamos para reconstruir los datos

                Y_training_mean, Y_training_std, Y_training_var =  \
                    normalize_and_fill(Y_training)[1:4]

                adj_total = adj_xr.sum('modo')

                adj_xr = (adj_total*Y_training_std/adj_total.std('time')) + \
                          Y_training_mean

                mod_adj.append(adj_xr)

            data_to_verif = Y_testing #

            mod_adj = xr.concat(mod_adj, dim='r')
            mod_adj['r'] = X_testing['r']
            mod_adj = mod_adj.to_dataset(name=list(X.data_vars)[0])

            if it_n == 0:
                adj_f = mod_adj
                data_to_verif_f = data_to_verif
            else:
                adj_f = xr.concat([mod_adj, adj_f], dim='time')
                data_to_verif_f = xr.concat(
                    [data_to_verif, data_to_verif_f], dim='time')

    else:
        adj_f, data_to_verif_f = None

    return adj_f, data_to_verif_f


def CCA_mod_CV(X, Y, var_exp,
                 X_mes=None, Y_mes=None,
                 X_trimestral=False, Y_trimestral=False,
                 X_anios=None, Y_anios=None,
                 window_years=3, X_test=None):

    """
    CCA con validacion cruzada

    Parametros:
    X (xr.DataArray)
    Y (xr.DataArray)
    var_exp (float): 0<var_exp<=1, default = 0.7 varianza que se quiere retener.
    X_mes (int): mes de X a usar
    Y_mes (int): mes de Y a usar
    X_trimestral (bool): si es True toma promedio movil de 3 meses
    Y_trimestral (bool): si es True toma promedio movil de 3 meses
    X_anios (list): extremos de anios para periodo de X
    Y_anios (list): extremos de anios para periodo de X
    window_years (int): ventana de anios a usar para la validacion cruzada
    X_test (xr.dataset): solo para calibracion. INTERNO

    return:
    output0 (xr.Dataset): reconstruccion a partir de CCA
    outout1 (xr.Dataset): Y original en el periodo de output0
    """

    var_name = list(Y.data_vars)[0]
    total_tiempos = len(X.time)
    compute = True
    if X_anios is None and Y_anios is None:
        print('X_anios, Y_anios is None')
        print('Se usará el periodo de anios en común mas largo entre X e Y')

        if len(X.time.values) < len(Y.time.values):
            X_anios_values = np.unique(X.time.dt.year.values)
            Y_anios_values = X_anios_values

        elif len(Y.time.values) < len(X.time.values):
            X_anios_values = np.unique(Y.time.dt.year.values)
            Y_anios_values = X_anios_values

    elif X_anios is not None and Y_anios is not None:
        X_anios_values = np.arange(X_anios[0], X_anios[-1] + 1)
        Y_anios_values = np.arange(Y_anios[0], Y_anios[-1] + 1)

        if len(X_anios_values) != len(Y_anios_values):
            print('El rango de X_anios, Y_anios debe contener la misma '
                  'cantidad de años')
            compute = False
    else:
        print('X_anios, Y_anios deben ser None o list, ambos.')
        compute = False

    if compute is True:
        # Set data
        if X_trimestral is True:
            X = X.rolling(time=3, center=True).mean('time')
            # Si se hace promedio movil perdemos los extremos y CCA no admite NaNs
            X = X.sel(time=X.time.isin(X.time.values[1:-1]))

        if Y_trimestral is True:
            Y = Y.rolling(time=3, center=True).mean('time')
            Y = Y.sel(time=Y.time.isin(Y.time.values[1:-1]))

        X = X.sel(time=X.time.dt.year.isin(X_anios_values))
        if X_mes is not None:
            X = X.sel(time=X.time.dt.month.isin(X_mes))

        Y = Y.sel(time=Y.time.dt.year.isin(Y_anios_values))
        Y = Y.sel(time=Y.time.dt.month.isin(Y_mes))

        if X_test is not None:
            if X_trimestral is True:
                X_test = X_test.rolling(time=3, center=True).mean('time')
                X_test = X_test.sel(time=X_test.time.isin(X.time.values[1:-1]))

            X_test = X_test.sel(time=X.time.dt.year.isin(X_anios_values))
            X_test = X_test.sel(time=X_test.time.dt.month.isin(X_mes))

    total_tiempos = len(X.time)
    mod_adj = []
    fechas_testing = []
    fechas = []
    for i in range(total_tiempos - window_years + 1):

        tiempos_a_omitir = range(i, i + window_years)

        # iteracion 0
        if X_test is not None:
            X_testing = X_test.sel(time=X_test.time.values[tiempos_a_omitir])
        else:
            X_testing = X.sel(time=X.time.values[tiempos_a_omitir])

        Y_testing = Y.sel(time=Y.time.values[tiempos_a_omitir])

        X_training = X.drop_sel(time=X_testing.time.values)
        Y_training = Y.drop_sel(time=Y_testing.time.values)

        pos_med = int((len(X_testing.time) - 1) / 2)

        fechas.append(X_testing.time.values[pos_med])
        fechas_testing.append(Y_testing.time.values[pos_med])

        adj = CCA_mod(X=X_training, X_test=X_testing,
                      Y=Y_training, var_exp=var_exp)[0]
        adj = adj[pos_med,:,:]

        adj_aux = adj.reshape(len(Y.lat.values), len(Y.lon.values),
                              adj.shape[-1])

        adj_xr = xr.DataArray(adj_aux,
                              dims=['lat', 'lon', 'modo'],
                              coords={'lat': Y.lat.values,
                                      'lon': Y.lon.values,
                                      'modo': np.arange(0, adj_aux.shape[-1])})

        Y_training_mean, Y_training_std, Y_training_var = \
            normalize_and_fill(Y_training)[1:4]

        adj_total = adj_xr.sum('modo')

        try:
            adj_std = adj_total.std('time')
        except:
            adj_std = adj_xr.std()

        adj_xr = (adj_total * Y_training_std / adj_std) + \
                 Y_training_mean

        mod_adj.append(adj_xr)

    mod_adj_ct = xr.concat(mod_adj, dim='time')
    if X_test is None:
        mod_adj_ct['time'] = fechas_testing
    else:
        mod_adj_ct['time'] = fechas

    mod_adj_ct = mod_adj_ct.to_dataset(name=var_name)

    Y_to_verif = Y.sel(time=fechas_testing)

    return mod_adj_ct, Y_to_verif#-Y_to_verif.mean('time')


def CCA_calibracion_CV(X_modelo, Y_observacion, var_exp,
                       Y_mes=None, Y_trimestral=False,
                       X_anios=None, Y_anios=None,
                       window_years=3):
    """
    Calibracion CCA con validacion cruzada

    Parametros:
    X_modelo (xr.DataArray): Modelo
    Y_observacion (xr.DataArray)
    var_exp (float): 0<var_exp<=1, default = 0.7 varianza que se quiere retener.
    Y_mes (int): mes de Y a usar
    Y_trimestral (bool): si es True toma promedio movil de 3 meses
    X_anios (list): extremos de anios para periodo de X
    Y_anios (list): extremos de anios para periodo de X
    window_years (int): ventana de anios a usar para la validacion cruzada

    return:
    output0 (xr.Dataset): Modelo calibrado a partir de CCA
    outout1 (xr.Dataset): Y original en el periodo de output0
    """


    X_mes = int(X_modelo.time.dt.month.mean().values)
    mod_adj = []
    for r in X_modelo.r.values:
        adj, Y_to_verif = CCA_mod_CV(X=X_modelo.mean('r'),
                                     Y=Y_observacion,
                                     X_test=X_modelo.sel(r=r),
                                     var_exp=var_exp,
                                     X_mes=X_mes, Y_mes=Y_mes,
                                     X_trimestral=False, Y_trimestral=Y_trimestral,
                                     X_anios=X_anios, Y_anios=Y_anios,
                                     window_years=window_years)
        mod_adj.append(adj)

    mod_adj_xr = xr.concat(mod_adj, dim='r')
    mod_adj_xr['r'] = X_modelo.r.values

    return mod_adj_xr, Y_to_verif
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def Media_Desvio_CV_Observaciones(data):
    """
    FUNCION INTERNA

    Calcula la media y desvio estandar usando validación cruzada de 1 año

    Parametros:
    data xr.dataarray

    return
    mean xr.dataarray: media para cada punto de reticula
    sd xr.dataarray: desvio estandar para cada punto de reticula
    """
    for t in data.time.values:
        data_no_t = data.where(data.time != t, drop=True)

        mean_no_t = data_no_t.mean(['time'])
        sd_no_t = data_no_t.std(['time'])

        if t == data.time.values[0]:
            mean_no_t_final = mean_no_t
            sd_no_t_final = sd_no_t
        else:
            mean_no_t_final = xr.concat(
                [mean_no_t_final, mean_no_t], dim='time')
            sd_no_t_final = xr.concat(
                [sd_no_t_final, sd_no_t], dim='time')

    mean = mean_no_t_final.mean('time')
    sd = sd_no_t_final.mean('time')

    return mean, sd


def Calibracion_MediaSD(X_modelo, Y_observacion,
                        Y_mes, Y_trimestral=False,
                        X_anios=[], Y_anios=[]):
    """
    Calibra removiendo la media y desvio standard del modelo y luego
    multiplicando y sumando el desvio y la media observada, respectivamente

    Parametros:
    X_modelo (xr.dataarray o xr.dataset): modelo
    Y_observacion (xr.dataarray o xr.dataset)
    Y_mes (int): mes de Y a usar
    Y_trimestral (bool): si es True toma promedio movil de 3 meses
    X_anios (list): extremos de anios para periodo de X
    Y_anios (list): extremos de anios para periodo de X

    return:
    output0 (xr.Dataset): Modelo calibrado
    outout1 (xr.Dataset): Y original en el periodo de output0
    """
    X = X_modelo
    Y = Y_observacion

    import warnings
    warnings.simplefilter("ignore", category=RuntimeWarning)

    compute = True
    if X_anios is None and Y_anios is None:
        print('X_anios, Y_anios is None')
        print('Se usará el periodo de anios en común mas largo entre X e Y')

        if len(X.time.values) < len(Y.time.values):
            X_anios_values = np.unique(X.time.dt.year.values)
            Y_anios_values = X_anios_values

        elif len(Y.time.values) < len(X.time.values):
            X_anios_values = np.unique(Y.time.dt.year.values)
            Y_anios_values = X_anios_values

    elif X_anios is not None and Y_anios is not None:
        X_anios_values = np.arange(X_anios[0], X_anios[-1] + 1)
        Y_anios_values = np.arange(Y_anios[0], Y_anios[-1] + 1)

        if len(X_anios_values) != len(Y_anios_values):
            print('El rango de X_anios, Y_anios debe contener la misma '
                  'cantidad de años')
            compute = False
    else:
        print('X_anios, Y_anios deben ser None o list, ambos.')
        compute = False

    if Y_trimestral:
        obs = Y.rolling(time=3, center=True).mean('time')

    obs = obs.sel(time=obs.time.dt.month.isin(Y_mes))
    obs = obs.sel(time=obs.time.dt.year.isin(Y_anios_values))

    # Como en _OLD
    mod = X

    calibrated_t = None
    obs_to_verif = None
    if compute:
        obs_mean, obs_sd = Media_Desvio_CV_Observaciones(obs)

        if 'r' in mod.dims:
            for r_e in mod.r.values:
                for t in mod.time.values:

                    mod_r_t = mod.sel(time=t, r=r_e)

                    aux = mod.where(mod.r != r_e, drop=True)
                    mod_no_r_t = aux.where(aux.time != t, drop=True)

                    mean_no_r_t = mod_no_r_t.mean(['time', 'r'])
                    sd_no_r_t = mod_no_r_t.std(['time', 'r'])

                    mod_r_t_calibrated = (((mod_r_t - mean_no_r_t) / sd_no_r_t)
                                          * obs_sd + obs_mean)

                    if t == mod.time.values[0]:
                        calibrated_r_t = mod_r_t_calibrated
                    else:
                        calibrated_r_t = xr.concat([calibrated_r_t,
                                                    mod_r_t_calibrated],
                                                   dim='time')

                if int(r_e) == int(mod.r.values[0]):
                    calibrated_t = calibrated_r_t
                else:
                    calibrated_t = xr.concat([calibrated_t, calibrated_r_t],
                                             dim='r')
            obs_to_verif = obs
        else:
            calibrated_t = None

    return calibrated_t, obs_to_verif


def Quantile_CV(data, quantiles=[0.33, 0.66]):

    """

    FUNCION INTERNA

    Calcula los quantiles para cada punto de reticula

    Parametros:
    data xr.dataarray o xr.dataset
    quantiles list: valores de los quaniles a calcular, 0.33 y 0.66 por default

    return
    data_q_t xr.dataarray o xr.dataset de los campos para cada quantile
    """

    for t in data.time.values:

        data_no_t = data.where(data.time != t, drop=True)

        data_q = data_no_t.quantile(quantiles, dim='time')

        if t == data.time.values[0]:
            data_q_t = data_q
        else:
            data_q_t = xr.concat([data_q_t, data_q], dim='time')

    data_q_t = data_q_t.mean('time')

    return data_q_t


def Prono_Qt(modelo, fecha_pronostico, obs_referencia=None,
             return_quantiles=False, verbose=True):

    """
    Calcula la categoria mas probable de un pronostico en funcion de los
    terciles climatologicos.

    Parametros
    modelo (xr.dataarray): tdo el modelo, sus años y sus miembros
    de ensamble
    fecha_pronostico (cftime._cftime.Datetime360Day): fecha que se quiere
    pronosticar.
    obs_referencia (xr.dataset o xr.dataarrat); se usaran para calcular los
    terciles climatologicos siempre que este argumento no sea None
    En ese caso, se asume que el modelo no está calibrado y los terciles se
    calculan a partir de la media del ensamble.

    return:
    categorias xr.dataarray con las tres categorias y sus valores

    """
    categorias = None
    error = False

    prono = modelo.sel(time=fecha_pronostico)
    try:
        modelo.time.values == fecha_pronostico
    except:
        modelo = modelo.where(
            ~modelo.time.isin(fecha_pronostico), drop=True)

    # aux = modelo.where(modelo.time != fecha_pronostico, drop=True)
    # modelo = aux.where(aux.time != fecha_pronostico, drop=True)

    if obs_referencia is not None:
        if verbose:
            print('observaciones not None: se asume modelo calibrado')
            print('Se usarán los terciles observados')

        anios_mod = modelo.time.dt.year
        obs = obs_referencia.sel(
            time=obs_referencia.time.dt.year.isin(anios_mod))
        q_33_66 = Quantile_CV(obs, quantiles=[0.33, 0.66])

        var_name = list(obs_referencia.data_vars)[0]

    elif obs_referencia is None:
        if verbose:
            print('observaciones is None: se asume modelo sin calibrar')
            print('Se usarán los terciles del modelo. Toma más tiempo...')

        q_33_66 = Quantile_CV(modelo, quantiles=[0.33, 0.66])
        var_name = list(modelo.data_vars)[0]

    else:
        print('Error Prono_Qt, revisar argumentos de entrada')
        error = True

    if error is False:
        # Extraer los cuantiles
        q_33 = q_33_66[var_name].sel(quantile=0.33)
        q_66 = q_33_66[var_name].sel(quantile=0.66)

        # Comparar cada miembro del ensamble con los cuantiles
        below = (prono[var_name] < q_33).sum(dim='r') / len(prono.r.values)
        normal = ((prono[var_name] >= q_33) & (prono[var_name] <= q_66)).sum(
            dim='r') / len(prono.r.values)
        above = (prono[var_name] > q_66).sum(dim='r') / len(prono.r.values)

        # las tres categorias a una variable
        categorias = xr.concat([below.drop_vars('quantile'),
                                normal,
                                above.drop_vars('quantile')],
                               dim="category")

        categorias = categorias.assign_coords(
            category=["below", "normal", "above"])

    if return_quantiles:
        return categorias, q_33_66
    else:
        return categorias


def Prono_AjustePDF(modelo, fecha_pronostico, obs_referencia=None,
                    return_quantiles=False, verbose=True):

    """
    Calcula la categoria mas probable de un pronostico en funcion de los
    terciles climatologicos derivaddos del ajuste guassiano.

    Parametros:
    modelo (xr.dataarray): tdo el modelo, sus años y sus miembros
    de ensamble
    fecha_pronostico (cftime._cftime.Datetime360Day): fecha que se quiere
    pronosticar.
    obs_referencia (xr.dataset o xr.dataarrat); se usaran para calcular la pdf
    climatologica siempre que este argumento no sea None
    En ese caso, se asume que el modelo no está calibrado y la pdf se
    calculara a partir de la media del ensamble.

    return:
    categorias xr.dataarray con las tres categorias y sus valores
    """
    import scipy.stats as stats

    prono = modelo.sel(time=fecha_pronostico)
    try:
        modelo.time.values == fecha_pronostico
    except:
        modelo = modelo.where(
            ~modelo.time.isin(fecha_pronostico), drop=True)

    anios_mod = modelo.time.dt.year

    if obs_referencia is None:
        if verbose:
            print('terciles modelo')
        data_clim = modelo
        dim_metrics = ['time', 'r']
    else:
        if verbose:
            print('terciles observados')
        data_clim = obs_referencia
        dim_metrics = ['time']
        data_clim = data_clim.sel(time=data_clim.time.dt.year.isin(anios_mod))

    var_name = list(data_clim.data_vars)[0]

    prono_mean = prono.mean('r')[var_name]
    prono_sd = prono.std('r')[var_name]

    tercil_1_f = []
    tercil_2_f = []
    for t in data_clim.time.values:
        data_clim_aux = data_clim.where(data_clim.time != t, drop=True)

        clim_mean = data_clim_aux.mean(dim_metrics)[var_name]
        clim_std = data_clim_aux.std(dim_metrics)[var_name]

        # Terciles de las distribuciones de las pdf
        # no hace falta tener explicitamente la pdf definida
        # Acá calcula los terciles de una distribucion nornmal con le media y
        # el desvio estandar climatologico.
        tercil_1_f.append(stats.norm.ppf(1 / 3, clim_mean, clim_std))
        tercil_2_f.append(stats.norm.ppf(2 / 3, clim_mean, clim_std))

    tercil_1 = xr.DataArray(tercil_1_f).mean('dim_0')
    tercil_2 = xr.DataArray(tercil_2_f).mean('dim_0')


    # probabilidades de que la pdf del pronostico esté en cada categoria
    # definida por los terciles
    below = stats.norm.cdf(tercil_1, prono_mean, prono_sd)
    normal = stats.norm.cdf(tercil_2, prono_mean, prono_sd) - below
    above = 1 - stats.norm.cdf(tercil_2, prono_mean, prono_sd)

    #below.shape (6, 81, 56)

    # las tres categorias a una variable
    if len(below.shape)==3:
        probabilidades_categoria = xr.DataArray(
            data=np.array([below, normal, above]),
            dims=['category', 'time', 'lat', 'lon'],
            coords=dict(
                category=['below', 'normal', 'above'],
                time=fecha_pronostico,
                lon=modelo.lon.values,
                lat=modelo.lat.values)
        )
    else:
        probabilidades_categoria = xr.DataArray(
            data=np.array([below, normal, above]),
            dims=['category', 'lat', 'lon'],
            coords=dict(
                category=['below', 'normal', 'above'],
                lon=modelo.lon.values,
                lat=modelo.lat.values)
        )

    if return_quantiles:
        q_33_66 = xr.concat([tercil_1, tercil_1], dim='quantile')
        q_33_66 = q_33_66.rename({'dim_1': 'lat', 'dim_2': 'lon'})
        q_33_66['lon'] = data_clim.lon.values
        q_33_66['lat'] = data_clim.lat.values
        q_33_66['quantile'] = [0.33, 0.66]
        q_33_66 = q_33_66.to_dataset(name=list(data_clim.data_vars)[0])

        return probabilidades_categoria, q_33_66
    else:
        return probabilidades_categoria


def MakeMask(DataArray, dataname='mask'):

    """
    FUNCION INTERNA
    Usar region mask para enmascarar el oceano

    :param DataArray:
    :param dataname:
    :return:
    """

    import regionmask
    mask=regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(DataArray)
    mask = xr.where(np.isnan(mask), mask, 1)
    mask = mask.to_dataset(name=dataname)
    return mask


def Plot_CategoriaMasProbable(data_categorias, variable,
                              titulo='Categoria más probable',
                              mask_ocean=False,
                              mask_andes = False,
                              save=False,
                              out_dir='~/',
                              name_fig='fig'):
    """
    Plotea la salida de Prono_Qt graficando la en cada punto de reticula
    la categoria mas probable

    Parametros:
    data_categorias (xr.Dataarray): salida de Prono_Qt
    variable (str): nombre de la variable, admite prec, pp, precipitacion,
    temp, tref o temperatura (determina las paletas de colores a usar)
    titulo (str): titulo de la figura
    mask_ocean (bool): mascara del oceano
    mask_andes (bool): mascara de los andes
    save (bool): guardar la figura
    out_dir (str): ruta del directorio de salida
    name_fig (str): nombre de la figura
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    crs_latlon = ccrs.PlateCarree()
    import warnings
    warnings.simplefilter("ignore", category=UserWarning)

    fig = plt.figure(figsize=(5.5,6), dpi=100)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([275, 330, -60, 20], crs=crs_latlon)

    # Paletas de colores para cada categoria
    if variable == 'pp' or variable == 'prec' or variable == 'precipitacion':
        colores_below = ListedColormap(
            plt.cm.YlOrBr(np.linspace(0.2, 0.6, 256)))
        colores_normal = ListedColormap(
            plt.cm.Greys(np.linspace(0.3, 0.8, 256)))
        colores_above = ListedColormap(plt.cm.Blues(np.linspace(0.3, 0.9, 256)))
    elif variable == 'temp' or variable == 'tref' or variable == 'temperatura':
        colores_below = ListedColormap(
            plt.cm.Blues(np.linspace(0.3, 0.9, 256)))
        colores_normal = ListedColormap(
            plt.cm.Greys(np.linspace(0.3, 0.8, 256)))
        colores_above = ListedColormap(plt.cm.Reds(np.linspace(0.2, 0.6, 256)))

    data_categorias = xr.where(np.isnan(data_categorias), -999, data_categorias)

    # Categoria mas probable en cada punto de reticula
    categoria_mas_probable = data_categorias.argmax(dim="category")
    if mask_ocean is True:
        try:
            mask_ocean = MakeMask(categoria_mas_probable)
            categoria_mas_probable = categoria_mas_probable * mask_ocean.mask
        except:
            print('regionmask no instalado, no se ensmacarará el océano')
            print('se puede instalar en el entorno con pip install regionmask')


    for i, (cat, cmap, label) in enumerate(
            zip([0, 1, 2], [colores_below, colores_normal, colores_above],
                ["Below", "Normal", "Above"])):
        mask = categoria_mas_probable == cat
        im = ax.pcolormesh(data_categorias.lon, data_categorias.lat,
                         data_categorias.isel(category=cat).where(mask),
                         vmin=0.4, vmax=1,#=np.arange(0.2, 1.2, 0.2),
                         transform=crs_latlon, cmap=cmap)

        cbar_ax = fig.add_axes([0.83, 0.17 + 0.25 * i, 0.02, 0.2])
        cbar = plt.colorbar(im, cax=cbar_ax, extend='both')
        cbar.set_label(label)
        cbar.ax.set_yticks([0.4, 0.6, 0.8, 1])

    if mask_andes is True:
        from SetCodes.descarga_topografia import compute
        topografia = compute()

        from matplotlib import colors
        andes_cmap = colors.ListedColormap(
            ['k'])  # una palenta de colores todo negro

        # contorno que va enmascarar el relieve superior a mask_level
        mask_level = 1300  # metros
        ax.contourf(topografia.lon, topografia.lat, topografia.topo,
                    levels=[mask_level, 666666],
                    cmap=andes_cmap, transform=crs_latlon)

    ax.coastlines(color='k', linestyle='-', alpha=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.set_xticks(np.arange(275, 330, 10), crs=crs_latlon)
    ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.tick_params(labelsize=10)

    ax.set_title(titulo, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    if save is True:
        print(f'Guardado en: {out_dir}{name_fig}.jpg')
        plt.savefig(f'{out_dir}{name_fig}.jpg', dpi=150)



def PlotPcolormesh_SA(data, data_var, scale, cmap, title,
                      mask_ocean=False, mask_andes=False,
                      save=False, out_dir='~/', name_fig='fig'):
    """
    Funcion de ejemplo de ploteo de datos georeferenciados

    Parametros:
    data (xr.Dataset): del cual se van a tomar los valores de lon y lat
    data_var (xr.Dataarray): variable a graficar
    scale (array): escala para plotear contornos
    cmap (str): nombre de paleta de colores de matplotlib
    title (str): título del grafico
    mask_ocean (bool): mascara del oceano
    mask_andes (bool): mascara de los andes
    save (bool): guardar la figura
    out_dir (str): ruta del directorio de salida
    name_fig (str): nombre de la figura

    """
    crs_latlon = ccrs.PlateCarree()

    if mask_ocean is True:
        try:
            mask_ocean = MakeMask(data_var)
            data_var = data_var * mask_ocean.mask
        except:
            print('regionmask no instalado, no se ensmacarará el océano')
            print('se puede instalar en el entorno con pip install regionmask')


    fig = plt.figure(figsize=(5,6), dpi=100)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([275, 330, -60, 20], crs=crs_latlon)

    # Contornos
    im = ax.pcolormesh(data.lon,
                     data.lat,
                     data_var,
                     vmin=np.min(scale), vmax=np.max(scale),
                     transform=crs_latlon, cmap=cmap)

    # barra de colores
    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)


    if mask_andes is True:
        from SetCodes.descarga_topografia import compute
        topografia = compute()

        from matplotlib import colors
        andes_cmap = colors.ListedColormap(
            ['k'])  # una palenta de colores todo negro

        # contorno que va enmascarar el relieve superior a mask_level
        mask_level = 1300  # metros
        ax.contourf(topografia.lon, topografia.lat, topografia.topo,
                    levels=[mask_level, 666666],
                    cmap=andes_cmap, transform=crs_latlon)

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

    if save is True:
        print(f'Guardado en: {out_dir}{name_fig}.jpg')
        plt.savefig(f'{out_dir}{name_fig}.jpg', dpi=150)


# ---------------------------------------------------------------------------- #
def decode_cf(ds, time_var):
    """
    FUNCION INTERNA
    Decodes time dimension to CFTime standards.
    """
    if ds[time_var].attrs["calendar"] == "360":
        ds[time_var].attrs["calendar"] = "360_day"
    ds = xr.decode_cf(ds, decode_times=True)
    return ds


def SetPronos_Climpred(mod, obs_ref):
    """
    Setea "mod" en funcion de obs_ref para trabajar con climpred

    mod xr.Dataset
    obs_ref xr.Dataset

    return mod seteado
    """
    try:
        try:
            falso_lead = obs_ref.time.values[0].month-mod.time.values[0].month
        except:
            falso_lead = obs_ref.time.dt.month - mod.time.dt.month
            fecha = mod.time.dt.month
    except:
        falso_lead = 0

    try:
        mod = mod.rename({'time':'init'})
    except:
        pass

    try:
        len(mod.init.values)
    except:
        mod = mod.expand_dims(dim='init')
        #mod['init'] = fecha

    mod = mod.expand_dims(dim=['lead'])
    try:
        mod['lead'] = [falso_lead]
    except:
        mod['lead'] = falso_lead.values

    mod["lead"].attrs = {"units": "months"}

    try:
        mod = mod.rename({'r':'member'})
    except:
        pass

    return mod


def Comparar_Qt(data, quantiles):

    """

    FUNCION INTERNA

    (se usa para observaciones)
    Compara valores de data con los quantiles 0.33 y 0.66
    data: xr.Dataset
    quantiles: xr.Dataarray que contengan los quantiles .33 y .66

    return below, nornal, above np.ndarray bnario para cada categoria
    """

    var_name_q = list(quantiles.data_vars)[0]
    var_name = list(data.data_vars)[0]

    q_33 = quantiles[var_name_q].sel(quantile=0.33)
    q_66 = quantiles[var_name_q].sel(quantile=0.66)

    # Comparar cada miembro del ensamble con los cuantiles
    below = (data[var_name] < q_33)
    normal = ((data[var_name] >= q_33) & (data[var_name] <= q_66))
    above = (data[var_name] > q_66)

    # esto asi xq sino aveces corta latitudes entonces x==False, 1, 0
    below = below.where(below==False, 1, 0)
    normal = normal.where(normal==False, 1, 0)
    above = above.where(above==False, 1, 0)

    return below, normal, above


def BSS(modelo, observaciones, fechas_pronostico, calibrado,
        funcion_prono='Prono_AjustePDF', bss_por_categorias=False):

    """
    Brier Skill Score

    modelo (xr.Dataset)
    observaciones (xr.Dataset)
    fechas_pronostico (cftime._cftime.Datetime360Day)
    calibrado (bool)
    bss_por_categorias (bool): si es True devuelve las 3 categorias del BSS

    return
    bss,
    si bss_por_categorias=True
    bss, bss_below, bss_normal, bss_above

    """

    if calibrado:
        if funcion_prono.lower() == 'prono_qt':
            prono, q_33_66 = Prono_Qt(modelo=modelo,
                                      fecha_pronostico=fechas_pronostico,
                                      obs_referencia=observaciones,
                                      return_quantiles=True,
                                      verbose=False)
            q_33_66_obs = q_33_66
        elif funcion_prono.lower() == 'prono_ajustepdf':
            prono, q_33_66 = Prono_AjustePDF(modelo=modelo,
                                             fecha_pronostico=fechas_pronostico,
                                             obs_referencia=observaciones,
                                             return_quantiles=True,
                                             verbose=False)
            q_33_66_obs = q_33_66

            try:
                len(fechas_pronostico.values)  # falla cuando es una sola fecha
            except:
                prono['time'] = fechas_pronostico

        else:
            print("funcion_prono debe ser Prono_Qt o Prono_AjustePDF")
    else:

        if funcion_prono.lower() == 'prono_qt':
            print('Modelo sin calibrar, el computo tarda bastante más...')
            prono, q_33_66 = Prono_Qt(modelo=modelo,
                                      fecha_pronostico=fechas_pronostico,
                                      obs_referencia=None,
                                      return_quantiles=True,
                                      verbose=False)

            q_33_66_obs = Quantile_CV(observaciones)

        elif funcion_prono.lower() == 'prono_ajustepdf':
            prono, q_33_66 = Prono_AjustePDF(modelo=modelo,
                                             fecha_pronostico=fechas_pronostico,
                                             obs_referencia=None,
                                             return_quantiles=True,
                                             verbose=False)

            # Como esto no tarda nada, podemos aplicar otra vez la funcion
            # tomando las observaciones y q_33_66_obs sera el de las
            # observaciones
            q_33_66_obs = Prono_AjustePDF(modelo=modelo,
                                          fecha_pronostico=fechas_pronostico,
                                          obs_referencia=observaciones,
                                          return_quantiles=True,
                                          verbose=False)[1]
            try:
                len(fechas_pronostico.values)  # falla cuando es una sola fecha
            except:
                prono['time'] = fechas_pronostico
        else:
            print("funcion_prono debe ser Prono_Qt o Prono_AjustePDF")

    try:
        prono = prono.expand_dims('time')
    except:
        pass

    try:
        anios_mod = []
        for f in fechas_pronostico:
            anios_mod.append(f.year)
    except:
        anios_mod = fechas_pronostico.year

    obs_below, obs_normal, obs_above = Comparar_Qt(observaciones.sel(
        time=observaciones.time.dt.year.isin(anios_mod)), q_33_66_obs)

    cts = []
    for ct in [obs_below, obs_normal, obs_above]:
        try:
            cts.append(ct.drop_vars('quantile'))
        except:
            cts.append(ct)

    obs_ct = xr.concat(cts, dim='category')
    obs_ct['category'] = ['below', 'normal', 'above']

    clim_prob = (.33333, .33334, .33333)
    BSo = []
    BSf = []

    # que buena idea che...
    BSo_below = []
    BSo_normal = []
    BSo_above = []

    BSf_below = []
    BSf_normal = []
    BSf_above = []


    for f_o, f_f in zip([obs_ct.time.values], [fechas_pronostico]):
        # BSo ---------------------------------------------------------------- #
        below = obs_ct.sel(time=f_o, category='below')
        normal = obs_ct.sel(time=f_o, category='normal')
        above = obs_ct.sel(time=f_o, category='above')

        aux_bs = (clim_prob[0] - below) ** 2 + \
                  (clim_prob[1] - normal) ** 2 + \
                  (clim_prob[2] - above) ** 2

        BSo.append(aux_bs)

        if bss_por_categorias is True:
            BSo_below.append((clim_prob[0] - below) ** 2)
            BSo_normal.append((clim_prob[1] - normal) ** 2)
            BSo_above.append((clim_prob[2] - above) ** 2)

        # BSf ---------------------------------------------------------------- #
        below_f = prono.sel(time=f_f, category='below')
        normal_f = prono.sel(time=f_f, category='normal')
        above_f = prono.sel(time=f_f, category='above')

        try:
            aux_bsf = (below_f - below.values) ** 2 + \
                      (normal_f - normal.values) ** 2 + \
                      (above_f - above.values) ** 2
        except:
            aux_bsf = (below_f - below) ** 2 + \
                      (normal_f - normal) ** 2 + \
                      (above_f - above) ** 2

        BSf.append(aux_bsf)

        if bss_por_categorias is True:
            try:
                BSf_below.append((below_f - below) ** 2)
                BSf_normal.append((normal_f - normal) ** 2)
                BSf_above.append((above_f - above) ** 2)
            except:
                BSf_below.append((below_f - below.values) ** 2)
                BSf_normal.append((normal_f - normal.values) ** 2)
                BSf_above.append((above_f - above.values) ** 2)


    BSo = xr.concat(BSo, dim='time')
    BSf = xr.concat(BSf, dim='time')

    BSo = BSo.drop_vars('category')
    BSf = BSf.drop_vars('category')

    if bss_por_categorias is True:
        BSo_below = xr.concat(BSo_below, dim='time')
        BSo_normal = xr.concat(BSo_normal, dim='time')
        BSo_above = xr.concat(BSo_above, dim='time')

        BSo_below = BSo_below.drop_vars('category')
        BSo_normal = BSo_normal.drop_vars('category')
        BSo_above = BSo_above.drop_vars('category')

        BSf_below = xr.concat(BSf_below, dim='time')
        BSf_normal = xr.concat(BSf_normal, dim='time')
        BSf_above = xr.concat(BSf_above, dim='time')

        BSf_below = BSf_below.drop_vars('category')
        BSf_normal = BSf_normal.drop_vars('category')
        BSf_above = BSf_above.drop_vars('category')

    # BSS -------------------------------------------------------------------- #
    try:
        bss = 1 - BSf/BSo.values

        if bss_por_categorias is True:
            bss_below = 1 - BSf_below / BSo_below.values
            bss_normal = 1 - BSf_normal / BSo_normal.values
            bss_above = 1 - BSf_above / BSo_above.values

    except:
        # cuando sea una sola fecha va caer aca
        bss = 1 - BSf / BSo
        bss['time'] = [fechas_pronostico]

        if bss_por_categorias is True:
            bss_below = 1 - BSf_below / BSo_below
            bss_normal = 1 - BSf_normal / BSo_normal
            bss_above = 1 - BSf_above / BSo_above

            bss_below['time'] = [fechas_pronostico]
            bss_normal['time'] = [fechas_pronostico]
            bss_above['time'] = [fechas_pronostico]

    bss = bss.to_dataset(name='BSS')
    bss = bss.mean('time')

    if bss_por_categorias is True:
        bss_below = bss_below.to_dataset(name='BSS_below')
        bss_below = bss_below.mean('time')

        bss_normal = bss_normal.to_dataset(name='BSS_normal')
        bss_normal = bss_normal.mean('time')

        bss_above = bss_above.to_dataset(name='BSS_above')
        bss_above = bss_above.mean('time')

    if bss_por_categorias is True:
        return bss, bss_below, bss_normal, bss_above
    else:
        return bss


def RPSS(modelo, observaciones, fechas_pronostico, calibrado,
        funcion_prono='Prono_AjustePDF'):
    """
    Ranked probability skill score

    modelo xr.Dataset
    observaciones xr.Dataset
    fechas_pronostico cftime._cftime.Datetime360Day
    calibrado bool

    return
    rpss xr.dataset

    """

    if calibrado:
        if funcion_prono.lower() == 'prono_qt':
            prono, q_33_66 = Prono_Qt(modelo=modelo,
                                      fecha_pronostico=fechas_pronostico,
                                      obs_referencia=observaciones,
                                      return_quantiles=True,
                                      verbose=False)
            q_33_66_obs = q_33_66
        elif funcion_prono.lower() == 'prono_ajustepdf':
            prono, q_33_66 = Prono_AjustePDF(modelo=modelo,
                                             fecha_pronostico=fechas_pronostico,
                                             obs_referencia=observaciones,
                                             return_quantiles=True,
                                             verbose=False)
            q_33_66_obs = q_33_66

            try:
                len(fechas_pronostico.values)  # falla cuando es una sola fecha
            except:
                prono['time'] = fechas_pronostico

        else:
            print("funcion_prono debe ser Prono_Qt o Prono_AjustePDF")
    else:

        if funcion_prono.lower() == 'prono_qt':
            print('Modelo sin calibrar, el computo tarda bastante más...')
            prono, q_33_66 = Prono_Qt(modelo=modelo,
                                      fecha_pronostico=fechas_pronostico,
                                      obs_referencia=None,
                                      return_quantiles=True,
                                      verbose=False)

            q_33_66_obs = Quantile_CV(observaciones)

        elif funcion_prono.lower() == 'prono_ajustepdf':
            prono, q_33_66 = Prono_AjustePDF(modelo=modelo,
                                             fecha_pronostico=fechas_pronostico,
                                             obs_referencia=None,
                                             return_quantiles=True,
                                             verbose=False)
            # Como esto no tarda nada, podemos aplicar otra vez la funcion
            # tomando las observaciones y q_33_66_obs sera el de las
            # observaciones
            q_33_66_obs = Prono_AjustePDF(modelo=modelo,
                                          fecha_pronostico=fechas_pronostico,
                                          obs_referencia=observaciones,
                                          return_quantiles=True,
                                          verbose=False)[1]
            try:
                len(fechas_pronostico.values)  # falla cuando es una sola fecha
            except:
                prono['time'] = fechas_pronostico
        else:
            print("funcion_prono debe ser Prono_Qt o Prono_AjustePDF")

    try:
        prono = prono.expand_dims('time')
    except:
        pass

    try:
        anios_mod = []
        for f in fechas_pronostico:
            anios_mod.append(f.year)
    except:
        anios_mod = fechas_pronostico.year

    obs_below, obs_normal, obs_above = Comparar_Qt(observaciones.sel(
        time=observaciones.time.dt.year.isin(anios_mod)), q_33_66_obs)

    cts = []
    for ct in [obs_below, obs_normal, obs_above]:
        try:
            cts.append(ct.drop_vars('quantile'))
        except:
            cts.append(ct)

    obs_ct = xr.concat(cts, dim='category')
    obs_ct['category'] = ['below', 'normal', 'above']

    clim_prob = (.33333, .33334, .33333)
    RPSo = []
    RPSf = []
    #for f in obs_ct.time.values:
    for f_o, f_f in zip([obs_ct.time.values], [fechas_pronostico]):
        # RPSo --------------------------------------------------------------- #
        below = obs_ct.sel(time=f_o, category='below')
        normal = obs_ct.sel(time=f_o, category='normal')
        above = obs_ct.sel(time=f_o, category='above')

        aux_rpso = (clim_prob[0] - below) ** 2 + \
                  (np.sum(clim_prob[0:2]) - (normal+below)) ** 2 + \
                  (np.sum(clim_prob) - (above+normal+below)) ** 2

        RPSo.append(aux_rpso)

        # RPSf --------------------------------------------------------------- #
        below_f = prono.sel(time=f_f, category='below')
        normal_f = prono.sel(time=f_f, category='normal')
        above_f = prono.sel(time=f_f, category='above')

        try:
            aux_rpsf = (below_f - below.values) ** 2 + \
                       ((normal_f+below_f) -
                        (normal.values+below.values)) ** 2 + \
                       ((above_f+normal_f+below_f) -
                        (above.values+normal.values+below.values)) ** 2
        except:
            aux_rpsf = (below_f - below) ** 2 + \
                       ((normal_f + below_f) - (normal+below)) ** 2 + \
                       ((above_f+normal_f+below_f)-(above+ normal+ below)) ** 2

        RPSf.append(aux_rpsf)

    RPSo = xr.concat(RPSo, dim='time')
    RPSf = xr.concat(RPSf, dim='time')

    RPSo = RPSo.drop_vars('category')
    RPSf = RPSf.drop_vars('category')

    # RPSS ------------------------------------------------------------------- #
    try:
        rpss = 1 - RPSf/RPSo.values
    except:
        rpss = 1 - RPSf/RPSo
        rpss['time'] = [fechas_pronostico]

    rpss = rpss.to_dataset(name='RPSS')
    rpss = rpss.mean('time')

    return rpss


def ROC(modelo, observaciones, fechas_pronostico, calibrado,
        funcion_prono='Prono_AjustePDF'):

    """
    ROC usando la funcion Prono_Qt para establecer probabilidades
    para cada categoria

    modelo xr.Dataset
    observaciones xr.Dataset
    fechas_pronostico cftime._cftime.Datetime360Day
    calibrado bool

    return xr.datarray
    """
    parche_nan = False
    if (True in np.isnan(modelo[list(modelo.data_vars)[0]].values) or
            True in np.isnan(
                observaciones[list(observaciones.data_vars)[0]].values)):
        print('Los datos contienen NaN!')
        print('El resutlado será aproximado/erroneo')

        print('Nota: Reemplazar los NAN por un valor constante '
              'NO SOLUCIONA EL PROBLEMA')

        parche_nan = True

    if calibrado:
        if funcion_prono.lower() == 'prono_qt':
            prono, q_33_66 = Prono_Qt(modelo=modelo,
                                      fecha_pronostico=fechas_pronostico,
                                      obs_referencia=observaciones,
                                      return_quantiles=True,
                                      verbose=False)
            q_33_66_obs = q_33_66
        elif funcion_prono.lower() == 'prono_ajustepdf':
            prono, q_33_66 = Prono_AjustePDF(modelo=modelo,
                                             fecha_pronostico=fechas_pronostico,
                                             obs_referencia=observaciones,
                                             return_quantiles=True,
                                             verbose=False)
            q_33_66_obs = q_33_66

            try:
                len(fechas_pronostico.values)  # falla cuando es una sola fecha
            except:
                prono['time'] = fechas_pronostico

        else:
            print("funcion_prono debe ser Prono_Qt o Prono_AjustePDF")
    else:

        if funcion_prono.lower() == 'prono_qt':
            print('Modelo sin calibrar, el computo tarda bastante más...')
            prono, q_33_66 = Prono_Qt(modelo=modelo,
                                      fecha_pronostico=fechas_pronostico,
                                      obs_referencia=None,
                                      return_quantiles=True,
                                      verbose=False)

            q_33_66_obs = Quantile_CV(observaciones)

        elif funcion_prono.lower() == 'prono_ajustepdf':
            prono, q_33_66 = Prono_AjustePDF(modelo=modelo,
                                             fecha_pronostico=fechas_pronostico,
                                             obs_referencia=None,
                                             return_quantiles=True,
                                             verbose=False)
            # Como esto no tarda nada, podemos aplicar otra vez la funcion
            # tomando las observaciones y q_33_66_obs sera el de las
            # observaciones
            q_33_66_obs = Prono_AjustePDF(modelo=modelo,
                                          fecha_pronostico=fechas_pronostico,
                                          obs_referencia=observaciones,
                                          return_quantiles=True,
                                          verbose=False)[1]
            try:
                len(fechas_pronostico.values)  # falla cuando es una sola fecha
            except:
                prono['time'] = fechas_pronostico
        else:
            print("funcion_prono debe ser Prono_Qt o Prono_AjustePDF")

    try:
        anios_mod = []
        for f in fechas_pronostico:
            anios_mod.append(f.year)
    except:
        anios_mod = fechas_pronostico.year

    obs_below, obs_normal, obs_above = Comparar_Qt(observaciones.sel(
        time=observaciones.time.dt.year.isin(anios_mod)), q_33_66_obs)

    ct_roc_result = []
    for ct, ct_name in zip([obs_below, obs_normal, obs_above],
                                   ['below', 'normal', 'above']):

        ct_set = SetPronos_Climpred(prono.sel(category=ct_name), ct)

        try:
            hindcast = HindcastEnsemble(ct_set)
        except:
            # esto va pasar solo con "Prono_AjustePDF"
            # redondeamos los valores xq la funcion de climpred tarda más sino
            ct_set.name = list(modelo.data_vars)[0]
            hindcast = HindcastEnsemble(np.round(ct_set, 1))

        try:
            hindcast = hindcast.add_observations(ct.expand_dims(dim='time'))
        except:
            hindcast = hindcast.add_observations(ct)

        if parche_nan:
            result = hindcast.verify(metric='roc',
                                     comparison='e2o',
                                     dim=['init'],
                                     alignment='maximize',
                                     bin_edges='continuous',
                                     return_results="all_as_metric_dim")

            result = result.mean(['lon', 'lat'], skipna=True)

        else:
            result = hindcast.verify(metric='roc',
                                     comparison='e2o',
                                     dim=['init', 'lon', 'lat'],
                                     alignment='maximize',
                                     bin_edges='continuous',
                                     return_results="all_as_metric_dim")

        try:
            ct_roc_result.append(result.drop_vars('quantile'))
        except:
            ct_roc_result.append(result)

    roc_result_xr = xr.concat(ct_roc_result, dim='category')

    return roc_result_xr


def REL(modelo, observaciones, fechas_pronostico, calibrado,
        funcion_prono='Prono_AjustePDF'):

    """
    reliability  usando la funcion Prono_Qt para establecer probabilidades
    para cada categoria

    modelo xr.Dataset
    observaciones xr.Dataset
    fechas_pronostico cftime._cftime.Datetime360Day
    calibrado bool

    return xr.datarray, tupla, tupla
    """
    parche_nan = False
    if (True in np.isnan(modelo[list(modelo.data_vars)[0]].values) or
            True in np.isnan(
                observaciones[list(observaciones.data_vars)[0]].values)):
        print('Los datos contienen NaN!')
        print('El resutlado será aproximado/erroneo')

        print('Nota: Reemplazar los NAN por un valor constante '
              'NO SOLUCIONA EL PROBLEMA')

        parche_nan = True

    if calibrado:
        if funcion_prono.lower() == 'prono_qt':
            prono, q_33_66 = Prono_Qt(modelo=modelo,
                                      fecha_pronostico=fechas_pronostico,
                                      obs_referencia=observaciones,
                                      return_quantiles=True,
                                      verbose=False)
            q_33_66_obs = q_33_66
        elif funcion_prono.lower() == 'prono_ajustepdf':
            prono, q_33_66 = Prono_AjustePDF(modelo=modelo,
                                             fecha_pronostico=fechas_pronostico,
                                             obs_referencia=observaciones,
                                             return_quantiles=True,
                                             verbose=False)
            q_33_66_obs = q_33_66

            try:
                len(fechas_pronostico.values)  # falla cuando es una sola fecha
            except:
                prono['time'] = fechas_pronostico

        else:
            print("funcion_prono debe ser Prono_Qt o Prono_AjustePDF")
    else:

        if funcion_prono.lower() == 'prono_qt':
            print('Modelo sin calibrar, el computo tarda bastante más...')
            prono, q_33_66 = Prono_Qt(modelo=modelo,
                                      fecha_pronostico=fechas_pronostico,
                                      obs_referencia=None,
                                      return_quantiles=True,
                                      verbose=False)

            q_33_66_obs = Quantile_CV(observaciones)

        elif funcion_prono.lower() == 'prono_ajustepdf':
            prono, q_33_66 = Prono_AjustePDF(modelo=modelo,
                                             fecha_pronostico=fechas_pronostico,
                                             obs_referencia=None,
                                             return_quantiles=True,
                                             verbose=False)
            # Como esto no tarda nada, podemos aplicar otra vez la funcion
            # tomando las observaciones y q_33_66_obs sera el de las
            # observaciones
            q_33_66_obs = Prono_AjustePDF(modelo=modelo,
                                          fecha_pronostico=fechas_pronostico,
                                          obs_referencia=observaciones,
                                          return_quantiles=True,
                                          verbose=False)[1]
            try:
                len(fechas_pronostico.values)  # falla cuando es una sola fecha
            except:
                prono['time'] = fechas_pronostico
        else:
            print("funcion_prono debe ser Prono_Qt o Prono_AjustePDF")

    try:
        anios_mod = []
        for f in fechas_pronostico:
            anios_mod.append(f.year)
    except:
        anios_mod = fechas_pronostico.year

    obs_below, obs_normal, obs_above = Comparar_Qt(observaciones.sel(
        time=observaciones.time.dt.year.isin(anios_mod)), q_33_66_obs)

    ct_rel_result = []
    for ct, ct_name in zip([obs_below, obs_normal, obs_above],
                                   ['below', 'normal', 'above']):

        ct_set = SetPronos_Climpred(prono.sel(category=ct_name), ct)

        try:
            hindcast = HindcastEnsemble(ct_set)
        except:
            ct_set.name = list(modelo.data_vars)[0]
            hindcast = HindcastEnsemble(ct_set)

        try:
            hindcast = hindcast.add_observations(ct.expand_dims(dim='time'))
        except:
            hindcast = hindcast.add_observations(ct)

        if parche_nan:

            result = hindcast.verify(metric='reliability',
                                     comparison='e2o',
                                     dim=['init'],
                                     alignment='maximize',
                                     probability_bin_edges=np.arange(-0.1, 1.2,
                                                                     0.2))
            result = result.mean(['lon','lat'])

        else:
            result = hindcast.verify(metric='reliability',
                                     comparison='e2o',
                                     dim=['init', 'lon', 'lat'],
                                     alignment='maximize',
                                     probability_bin_edges=np.arange(-0.1, 1.2,
                                                                     0.2))

        try:
            ct_rel_result.append(result.drop_vars('quantile'))
        except:
            ct_rel_result.append(result)

    rel_result_xr = xr.concat(ct_rel_result, dim='category')

    hist_above = np.histogram(prono.sel(category='above'),
                              bins=[-0.1, 0.1, 0.3,0.5,0.7,0.9,1.1])
    aux = hist_above[0] / np.prod(prono.sel(category='above').values.shape)
    hist_above = (aux,[0, 0.2, 0.4, 0.6, 0.8, 1])

    hist_below = np.histogram(prono.sel(category='below'),
                              bins=[-0.1, 0.1, 0.3,0.5,0.7,0.9,1.1])
    aux = hist_below[0] / np.prod(prono.sel(category='below').values.shape)
    hist_below = (aux, [0, 0.2, 0.4, 0.6, 0.8, 1])

    return rel_result_xr, hist_above, hist_below


def PlotROC(roc, save=False, out_dir='~/', name_fig='fig'):
    """
    roc: salida de ROC()

    save (bool): guardar la figura
    out_dir (str): ruta del directorio de salida
    name_fig (str): nombre de la figura
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    var_name = list(roc.data_vars)[0]
    ax.plot(roc.sel(metric='false positive rate', category='above')[var_name],
            roc.sel(metric='true positive rate', category='above')[var_name],
            label='above', color='red', marker='o')
    ax.plot(roc.sel(metric='false positive rate', category='below')[var_name],
            roc.sel(metric='true positive rate', category='below')[var_name],
            label='below', color='blue', marker='o')

    auc_below = roc.sel(metric='area under curve', category='below')[var_name][
        0].values
    auc_above = roc.sel(metric='area under curve', category='above')[var_name][
        0].values

    ax.set_title(f'AUC below: {auc_below:.3}, AUC above {auc_above:.3}')
    plt.legend()
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.grid()
    plt.show()

    if save is True:
        print(f'Guardado en: {out_dir}{name_fig}.jpg')
        plt.savefig(f'{out_dir}{name_fig}.jpg', dpi=150)



def PlotRelDiag(rel, hist_above, hist_below,
                save=False, out_dir='~/', name_fig='fig'):
    """
    rel, hist_above, hist_below salidas de REL()

    save (bool): guardar la figura
    out_dir (str): ruta del directorio de salida
    name_fig (str): nombre de la figura
    """
    import matplotlib.pyplot as plt
    var_name = list(rel.data_vars)[0]
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(rel.sel(category='below').forecast_probability,
            rel.sel(category='below')[var_name],
            label='below', color='blue', marker='o')
    ax.plot(rel.sel(category='above').forecast_probability,
            rel.sel(category='above')[var_name],
            label='above', color='red', marker='o')

    ax.plot(hist_above[1], hist_above[0], color='red',
            linestyle='--', linewidth=0.8)
    ax.plot(hist_below[1], hist_below[0], color='blue',
            linestyle='--', linewidth=0.8)

    ax.set_title(f'Reliability')
    plt.legend()
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Relative Frequency')
    plt.grid()
    plt.show()

    if save is True:
        print(f'Guardado en: {out_dir}{name_fig}.jpg')
        plt.savefig(f'{out_dir}{name_fig}.jpg', dpi=150)


# ---------------------------------------------------------------------------- #
################################################################################