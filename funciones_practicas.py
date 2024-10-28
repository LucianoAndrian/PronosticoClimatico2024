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
def CrossAnomaly_1y(data, norm=False, r=False):
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
            else:
                data_anom = xr.concat([data_anom, aux_anom], dim='time')

    return data_anom


# ---------------------------------------------------------------------------- #
def ACC(data1, data2, crossanomaly=False, reference_time_period=None,
        verbose=True):
    """
    Anomaly correlation coefficient

    parametros:
    data1 (xr.Dataset): con dimensiones lon, lat, time (orden no importa)
    data2 (xr.Dataset): con dimensiones lon, lat, time (orden no importa)
    crossanomaly (bool): default False, True toma la anomalia cruzada de un año
    reference_time_period (list, opcional): [año_inicial, año_final]
    para el cálculo de anomalías. Si es None y crossanomaly es False,
    no se calculan anomalías.
    verbose (bool): Si es True, imprime mensajes de procesamiento en terminal.

    - si 'crossanomaly' es True no se tiene en cuenta 'reference_time_period'
    - si 'crossanomaly' es False y 'reference_time_period' es None,
    se asume que data1 y data2 ya contienen anomalías.
    - si uno de los data tiene una dimensión más el ACC se va devolver tambien
    para esa dimension. Ej. data1 [lon, lat, time, r], data2 [lon, lat, time]
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
def Compute_MLR_CV(xrds, predictores, window_years, intercept=True):
    """
    Funcion de EJEMPLO de MLR con validación cruzada.

    La funcion tiene aspectos por mejorar, por ejemplo:
     - parelización: se podria acelerar el computo de manera considerable
     - return: la manera en la que se devuelven los resultados busca satisfacer
     varias utilidades posteriores de ellos pero puede no ser la mas adecuada.

    Parametros:
    xrds (xr.Dataset): array dim [time, lon, lat] o [time, lon, lat, r]
        (no importa el orden, cuanto mas dimensiones mas tiempo va tardar)
    predictores (list): lista con series temporales a usar como predictores
    window_years (int): ventana de años a usar en la validacion cruzada
    intercept (bool): True, ordenada al orgine (recomendado usar siempre)

    return
    1. array de dimensiones [k, lon, lat, coef.] o [k, r, lon, lat, coef]
    - "k" son los "k-fold": años seleccionados para el modelo lineal
    - "coef" coeficientes del modelo lineal en cada punto de grilla en orden:
     constante, b1, b2..., bn
     2. idem 1. sin "coef" y en "k" estan los años omitidos en cada k-fold
     3. dict. Diccionario donde cada elemento es una lista de xr.dataarry con
     los predictores en los años omitods en cada k-fold

     La idea de estos dos ultimos resultados es poder usarlos para realizar y
     evaluar el pronostico del modelo lineal en cada periodo omitido.
    """

    total_tiempos = len(xrds.time)
    predictores_years_out = {}
    for i in range(total_tiempos - window_years + 1):

        per10 = 10 * round((i / total_tiempos) * 10)
        if i == 0 or per10 != 10 * round(((i - 1) / total_tiempos) * 10):
            print(f"{per10}% completado")

        # posicion de tiempos a omitir
        tiempos_a_omitir = range(i, i + window_years)

        # selección de tiempos
        ds_cv = xrds.drop_sel(
            time=xrds.time.values[tiempos_a_omitir[0]:tiempos_a_omitir[-1]+1])
        ds_out_years = xrds.drop_sel(time=ds_cv.time.values)

        if 'r' in ds_cv.dims:
            ds_cv = ds_cv - ds_cv.mean(['r','time'])
            ds_out_years = ds_out_years - ds_cv.mean(['r','time'])
        else:
            ds_cv = ds_cv - ds_cv.mean('time')
            ds_out_years = ds_out_years - ds_cv.mean(['time'])

        predictores_cv = []
        aux_predictores_out_years = []
        for p in predictores:
            aux_predictor = p.drop_sel(
                time=p.time.values[tiempos_a_omitir[0]:tiempos_a_omitir[-1]+1])
            aux_predictores_out_years.append(
                p.drop_sel(time=aux_predictor.time.values))

            predictores_cv.append(aux_predictor)

        predictores_years_out[f"k{i}"] = aux_predictores_out_years

        # MLR
        mlr = MLR(predictores_cv)
        regre_result = mlr.compute_MLR(ds_cv[list(ds_cv.data_vars)[0]],
                                       intercept=intercept)

        if i == 0:
            regre_result_cv = regre_result
            ds_out_years_cv = ds_out_years
        else:
            regre_result_cv = xr.concat([regre_result_cv, regre_result],
                                        dim='k')
            ds_out_years_cv = xr.concat([ds_out_years_cv, ds_out_years],
                                        dim='k')

    return regre_result_cv, ds_out_years_cv, predictores_years_out

# ---------------------------------------------------------------------------- #

def Weights(data, lon_name='lon', lat_name='lat'):
    """
    Normaliza por pesando por el coseno de la latitud

    parametros:
    data xr.DataArray o xr.DataSet
    lon_name str nombre de la dimension longitud
    lat_name str nombre de la dimension latitud

    return
    data xr.DataArray o xr.DataSet pesado \m/
    """

    weights = np.transpose(np.tile(np.cos(data[lat_name] * np.pi / 180),
                                   (len(data[lon_name]), 1)))
    data_w = data * weights
    return data_w


def normalize_and_fill(data):
    """
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
    data_normalized = ((data_filled - data_filled.mean(dim='time')) /
                       data_filled.std(dim='time'))

    # Aveces aparecen NaN otra vez despues del paso anterior
    data_normalized = data_normalized.fillna(0)

    # Transformamos la data a (tiempo, spatial) donde spatial = lat*lon
    data_normalized = data_normalized.stack(spatial=(lat_name, lon_name))

    return data_normalized


def AutoVec_Val_EOF(m, var_exp):
    """
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
        pca = PCA(n_components=var_exp)
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


def CCA(X, Y, var_exp=0.7):
    """
    Correlación canonica entre X e Y
    - Normaliza X, Y
    - prefilstra con EOF el numero de cp para explicar var_exp
    - CCA

    Parametros:
    X xr.DataArray o Xr.DataSet, de dimension (t, x)
    Y xr.DataArray o Yr.DataSet, de dimension (t, y)
    var_exp float, default = 0.7 varianza que se quiere retener.

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
    # Normalizado
    X = normalize_and_fill(X)
    Y = normalize_and_fill(Y)

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

    return P, Q, heP, heQ, S, A, B

def CCA_mod(X, X_test, Y, var_exp=0.7):
    """
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
    X_training = normalize_and_fill(X)
    X_verif = normalize_and_fill(X_test)

    Y_training = normalize_and_fill(Y)

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

    # Esto lo se usa para algo?
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
# ---------------------------------------------------------------------------- #