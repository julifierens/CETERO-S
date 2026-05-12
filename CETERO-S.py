import os
import io
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import tempfile
from datetime import datetime


# =========================================================
# CONFIGURACIÓN
# =========================================================
CONFIG = {
    "col_cabezal": "Channel_0",
    "col_carga": "Channel_1",
    "col_cmod": "Channel_2",

    # Suavizado
    "ventana_suavizado": 7,

    # Detección de descargas
    "min_puntos_descarga": 18,
    "umbral_derivada_descarga": -0.0015,   # kN/punto
    "caida_minima_carga": 0.08,            # kN
    "corr_min_descarga": -0.85,            # correlación índice-carga

    # 8 ventanas posibles para probar compliance
    "ventanas_analisis": [
        (0.20, 0.80),
        (0.25, 0.75),
        (0.30, 0.70),
        (0.35, 0.65),
        (0.40, 0.60),
        (0.22, 0.68),
        (0.28, 0.72),
        (0.32, 0.78),
    ],

    # Calidad mínima del ajuste
    "r2_min": 0.70,
    "delta_cmod_min": 0.002,
    "fraccion_rango_carga_min": 0.12,
    
    "tolerancia_default_rel": 0.20,

    # Corrección por rotación
    "iter_rotacion": 20,
    "tol_rotacion": 1e-8,
}


# =========================================================
# FUNCIONES DE LECTURA
# =========================================================
def seleccionar_archivo():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    ruta = filedialog.askopenfilename(
        title="Seleccione el archivo .csv del ensayo",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )

    root.destroy()
    return ruta


def detectar_delimitador(linea):
    cant_comas = linea.count(",")
    cant_tabs = linea.count("\t")
    return "\t" if cant_tabs > cant_comas else ","


def generar_nombres_unicos(lista):
    contador = {}
    resultado = []

    for nombre in lista:
        nombre = nombre.strip().strip('"')
        if nombre == "":
            nombre = "col"

        if nombre in contador:
            contador[nombre] += 1
            nombre_nuevo = f"{nombre}_{contador[nombre]}"
        else:
            contador[nombre] = 0
            nombre_nuevo = nombre

        resultado.append(nombre_nuevo)

    return resultado


def leer_csv_maquina(ruta):
    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        lineas = f.readlines()

    if len(lineas) < 3:
        raise ValueError("El archivo no tiene el formato esperado.")

    delimitador = detectar_delimitador(lineas[1])
    lector = csv.reader(lineas, delimiter=delimitador)
    filas = list(lector)

    encabezado_real = filas[1]
    nombres_columnas = generar_nombres_unicos(encabezado_real)

    datos = "\n".join(delimitador.join(fila) for fila in filas[2:])

    df = pd.read_csv(
        io.StringIO(datos),
        sep=delimitador,
        header=None,
        names=nombres_columnas,
        engine="python"
    )

    return df


def convertir_a_numerico(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================================================
# PREPARACIÓN DE DATOS
# =========================================================
def elegir_columnas(df):
    col_cabezal = CONFIG["col_cabezal"]
    col_carga = CONFIG["col_carga"]
    col_cmod = CONFIG["col_cmod"]

    for col in [col_cabezal, col_carga, col_cmod]:
        if col not in df.columns:
            raise ValueError(f"No encuentro la columna requerida: {col}")

    return col_cabezal, col_carga, col_cmod


def preparar_datos(df, col_x, col_y):
    datos = df[[col_x, col_y]].dropna().copy()

    x = datos[col_x].to_numpy(dtype=float)
    y = datos[col_y].to_numpy(dtype=float)

    if np.nanmean(x) < 0:
        x = -x
    if np.nanmean(y) < 0:
        y = -y

    return x, y


# =========================================================
# UTILIDADES
# =========================================================
def suavizar(y, ventana=7):
    if ventana < 3 or len(y) < ventana:
        return y.copy()

    kernel = np.ones(ventana) / ventana
    return np.convolve(y, kernel, mode="same")


def extraer_ventana_porcentaje(x, y, p_ini, p_fin):
    n = len(x)
    i0 = int(np.floor(n * p_ini))
    i1 = int(np.ceil(n * p_fin))

    i0 = max(0, i0)
    i1 = min(n, i1)

    if i1 - i0 < 3:
        return None, None

    return x[i0:i1], y[i0:i1]


def ajuste_lineal_filtrado(x, y, max_iter=5, tol_resid=2.5):
    """
    Ajuste lineal robusto con eliminación iterativa de outliers.
    Ajusta: y = m*x + b
    """
    if len(x) < 5:
        return np.nan, np.nan, np.nan, None, None

    mask = np.ones(len(x), dtype=bool)

    for _ in range(max_iter):
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 5:
            break

        m, b = np.polyfit(x_fit, y_fit, 1)
        y_pred = m * x + b

        resid = y - y_pred
        std = np.std(resid[mask])

        if std < 1e-12:
            break

        new_mask = np.abs(resid) < tol_resid * std

        if np.all(new_mask == mask):
            break

        mask = new_mask

    x_final = x[mask]
    y_final = y[mask]

    if len(x_final) < 5:
        return np.nan, np.nan, np.nan, None, None

    m, b = np.polyfit(x_final, y_final, 1)
    y_pred = m * x_final + b

    ss_res = np.sum((y_final - y_pred) ** 2)
    ss_tot = np.sum((y_final - np.mean(y_final)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return m, b, r2, x_final, y_final

def calcular_u_desde_compliance(C, B, BN, E, nu):
    """
    Calcula el parámetro u a partir de la compliance C.
    """
    Be = B - ((B - BN) ** 2) / B
    Eprima = E / (1.0 - nu ** 2)

    C_N = C / 1000.0   # pasar de mm/kN a mm/N
    valor = Be * Eprima * C_N    
    
    if valor <= 0:
        return np.nan

    u = 1.0 / (np.sqrt(valor) + 1.0)
    return u

def calcular_a0i_desde_u_ct(u, W):
    """
    Calcula a0i/W y a0i para geometría C(T), según polinomio de calibración.
    """
    if np.isnan(u):
        return np.nan, np.nan

    aW = (
        1.000196
        - 4.06319 * u
        + 11.242 * u**2
        - 106.043 * u**3
        + 464.335 * u**4
        - 650.677 * u**5
    )

    a0i = aW * W
    return aW, a0i

def determinar_a0q_por_compliance_corregida(resultados_seleccionados, W, B, BN, E, nu, n_primeras=None):
    """
    Calcula a0i para cada descarga activa usando la compliance corregida por rotación (Ci_final),
    luego a0q promedio, y verifica el criterio |a0i - a0q| <= 0.002 W.

    Si n_primeras no es None, usa solo las primeras n_primeras descargas activas corregidas.
    """
    activos = [
        r for r in resultados_seleccionados
        if r["seleccionada"]
        and not r.get("eliminada", False)
        and not r.get("eliminada_corregida", False)
        and not np.isnan(r["Ci_final"])
    ]

    activos = sorted(activos, key=lambda r: r["descarga"])

    if n_primeras is not None:
        activos = activos[:n_primeras]

    filas = []

    for r in activos:
        Ccorr = r["Ci_final"]
        u = calcular_u_desde_compliance(Ccorr, B, BN, E, nu)
        aW, a0i = calcular_a0i_desde_u_ct(u, W)

        filas.append({
            "descarga": r["descarga"],
            "Ci_final": Ccorr,
            "u": u,
            "a0i_W": aW,
            "a0i_mm": a0i,
        })

    if len(filas) < 3:
        return {
            "tabla": pd.DataFrame(filas),
            "a0q": np.nan,
            "cumple": False,
            "motivo": "Hay menos de 3 descargas activas corregidas. No alcanza para evaluar a0q.",
        }

    tabla = pd.DataFrame(filas)

    a0q = tabla["a0i_mm"].mean()
    tolerancia = 0.002 * W

    tabla["desvio_abs_mm"] = np.abs(tabla["a0i_mm"] - a0q)
    tabla["cumple_individual"] = tabla["desvio_abs_mm"] <= tolerancia

    cumple = bool(tabla["cumple_individual"].all())

    return {
        "tabla": tabla,
        "a0q": a0q,
        "cumple": cumple,
        "tolerancia_mm": tolerancia,
        "motivo": "ok" if cumple else "Uno o más valores individuales no cumplen |a0i - a0q| <= 0.002 W",
    }

def crear_carpeta_reporte(nombre_archivo_base):
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = os.path.join(
        os.path.dirname(nombre_archivo_base),
        f"reporte_figuras_{marca_tiempo}"
    )
    os.makedirs(carpeta, exist_ok=True)
    return carpeta

def guardar_o_mostrar_figura(fig, ruta_guardado=None, mostrar=True):
    if ruta_guardado is not None:
        fig.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
        print(f"Figura guardada en: {ruta_guardado}")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

# =========================================================
# DETECCIÓN DE DESCARGAS
# =========================================================
def detectar_descargas(
    carga,
    min_puntos=18,
    umbral_derivada=-0.0015,
    caida_minima=0.08,
    corr_min=-0.85,
    ventana_suavizado=7
):
    """
    Detecta tramos de descarga sostenida.
    Devuelve lista de tuplas (i_inicio, i_fin).
    """
    carga_s = suavizar(carga, ventana=ventana_suavizado)
    dP = np.diff(carga_s)

    descargando = dP < umbral_derivada
    segmentos = []

    en_segmento = False
    inicio = None

    for i, flag in enumerate(descargando):
        if flag and not en_segmento:
            inicio = i
            en_segmento = True
        elif not flag and en_segmento:
            fin = i + 1

            if fin - inicio >= min_puntos:
                sub = carga_s[inicio:fin]
                if len(sub) >= min_puntos:
                    caida = sub[0] - sub[-1]

                    idx = np.arange(len(sub))
                    corr = np.corrcoef(idx, sub)[0, 1] if len(sub) > 2 else 0.0

                    if np.isnan(corr):
                        corr = 0.0

                    if caida >= caida_minima and corr <= corr_min:
                        segmentos.append((inicio, fin))

            en_segmento = False
            inicio = None

    if en_segmento and inicio is not None:
        fin = len(carga_s)
        if fin - inicio >= min_puntos:
            sub = carga_s[inicio:fin]
            caida = sub[0] - sub[-1]

            idx = np.arange(len(sub))
            corr = np.corrcoef(idx, sub)[0, 1] if len(sub) > 2 else 0.0

            if np.isnan(corr):
                corr = 0.0

            if caida >= caida_minima and corr <= corr_min:
                segmentos.append((inicio, fin))

    return segmentos


# =========================================================
# CANDIDATOS DE COMPLIANCE
# =========================================================
def generar_candidatos_compliance(
    carga_seg,
    cmod_seg,
    numero_descarga=None,
    n_submuestras=400,
    n_puntos_sub=18,
    seed=1234
):
    """
    Para cada ventana fija genera candidatos de compliance.

    Regla:
    - En las primeras 6 descargas: SIEMPRE intenta devolver opciones,
      aunque no cumplan todos los criterios.
    - Desde la descarga 7: mantiene criterio estricto.
    """

    candidatos = []

    if len(carga_seg) < 8:
        return candidatos

    carga_seg = np.asarray(carga_seg, dtype=float)
    cmod_seg = np.asarray(cmod_seg, dtype=float)

    rango_total = np.max(carga_seg) - np.min(carga_seg)
    if rango_total <= 0:
        return candidatos

    rng = np.random.default_rng(seed)
    
    # Relajar criterios en primeras descargas
    modo_relajado = (numero_descarga is not None and numero_descarga <= 6)

    def evaluar_candidato(xw, yw, etiqueta_opcion, tipo, ventana, n_usados):
        if xw is None or yw is None or len(xw) < 5:
            return None

        xw = np.asarray(xw, dtype=float)
        yw = np.asarray(yw, dtype=float)

        orden = np.argsort(xw)
        xw = xw[orden]
        yw = yw[orden]

        pendiente, intercepto, r2, xw_filtrado, yw_filtrado = ajuste_lineal_filtrado(xw, yw)

        if xw_filtrado is None or yw_filtrado is None:
            return None
        if len(xw_filtrado) < 5:
            return None
        if np.isnan(pendiente) or pendiente <= 0:
            return None

        rango_w = np.max(xw_filtrado) - np.min(xw_filtrado)
        frac_rango = rango_w / rango_total if rango_total > 0 else 0.0
        delta_cmod = np.max(yw_filtrado) - np.min(yw_filtrado)

        cumple = (
            r2 >= CONFIG["r2_min"]
            and frac_rango >= CONFIG["fraccion_rango_carga_min"]
            and delta_cmod >= CONFIG["delta_cmod_min"]
            and len(xw_filtrado) >= n_puntos_sub
        )

        # Score continuo de calidad, para poder rankear aunque no cumpla
        score = (
            4.0 * max(0.0, min(r2, 1.0)) +
            2.5 * min(frac_rango / max(CONFIG["fraccion_rango_carga_min"], 1e-12), 2.0) +
            2.0 * min(delta_cmod / max(CONFIG["delta_cmod_min"], 1e-12), 2.0) +
            1.0 * min(len(xw_filtrado) / max(n_puntos_sub, 1), 2.0)
        )

        return {
            "opcion": etiqueta_opcion,
            "tipo": tipo,
            "Ci": float(pendiente),
            "intercepto": float(intercepto),
            "R2": float(r2),
            "ventana": ventana,
            "xw": xw_filtrado,
            "yw": yw_filtrado,
            "n_puntos": int(len(xw_filtrado)),
            "n_puntos_originales": int(n_usados),
            "rango_carga": float(rango_w),
            "fraccion_rango_carga": float(frac_rango),
            "delta_cmod": float(delta_cmod),
            "carga_media": float(np.mean(xw_filtrado)),
            "cmod_medio": float(np.mean(yw_filtrado)),
            "cumple": bool(cumple),
            "score": float(score),
        }

    for idx, (p_ini, p_fin) in enumerate(CONFIG["ventanas_analisis"], start=1):
        xw, yw = extraer_ventana_porcentaje(carga_seg, cmod_seg, p_ini, p_fin)
        if xw is None or yw is None:
            continue

        xw = np.asarray(xw, dtype=float)
        yw = np.asarray(yw, dtype=float)

        if len(xw) < 8:
            continue

        candidatos_ventana = []

        # A) todos los puntos de la ventana
        cand_full = evaluar_candidato(
            xw, yw,
            etiqueta_opcion=f"{idx}A",
            tipo="full_window",
            ventana=(p_ini, p_fin),
            n_usados=len(xw)
        )
        if cand_full is not None:
            candidatos_ventana.append(cand_full)

        # B) submuestras buscando extremos y buenos scores
        n_sel_base = min(n_puntos_sub, len(xw))

        for _ in range(n_submuestras):
            if len(xw) <= n_sel_base:
                ind = np.arange(len(xw))
            else:
                ind = np.sort(rng.choice(len(xw), size=n_sel_base, replace=False))

            x_sub = xw[ind]
            y_sub = yw[ind]

            cand = evaluar_candidato(
                x_sub, y_sub,
                etiqueta_opcion="tmp",
                tipo="submuestra_18pts",
                ventana=(p_ini, p_fin),
                n_usados=len(ind)
            )

            if cand is not None:
                candidatos_ventana.append(cand)

        if len(candidatos_ventana) == 0:
            continue

        # Desde descarga 7, quedarse solo con los estrictamente válidos
        if not modo_relajado:
            candidatos_ventana = [c for c in candidatos_ventana if c["cumple"]]
            if len(candidatos_ventana) == 0:
                continue

        # Elegir 3 representantes por ventana:
        # mejor score, menor Ci y mayor Ci
        mejor_score = max(candidatos_ventana, key=lambda c: c["score"])
        menor_ci = min(candidatos_ventana, key=lambda c: c["Ci"])
        mayor_ci = max(candidatos_ventana, key=lambda c: c["Ci"])

        mejor_score["opcion"] = f"{idx}A"
        mejor_score["tipo"] = "mejor_score"

        menor_ci["opcion"] = f"{idx}B"
        menor_ci["tipo"] = "min_Ci"

        mayor_ci["opcion"] = f"{idx}C"
        mayor_ci["tipo"] = "max_Ci"

        candidatos.extend([mejor_score, menor_ci, mayor_ci])

    # Limpiar duplicados
    unicos = []
    usados = set()
    for c in candidatos:
        clave = (
            round(c["Ci"], 10),
            round(c["R2"], 6),
            tuple(c["ventana"]),
            c["tipo"]
        )
        if clave not in usados:
            unicos.append(c)
            usados.add(clave)

    # Orden final:
    # primeras descargas -> por score
    # resto -> por Ci
    if modo_relajado:
        unicos = sorted(unicos, key=lambda c: (-c["score"], c["Ci"]))
    else:
        unicos = sorted(unicos, key=lambda c: c["Ci"])

    return unicos[:6]

def seleccionar_extremos_compliance(candidatos, n_extremos=3):
    """
    De todos los candidatos calculados, devuelve:
    - n_extremos con Ci más baja
    - n_extremos con Ci más alta

    Evita duplicados exactos por opción/ventana/tipo.
    """
    if not candidatos:
        return []

    # Filtrar candidatos válidos numéricamente
    candidatos_ok = [
        c for c in candidatos
        if c is not None and np.isfinite(c["Ci"]) and c["Ci"] > 0
    ]

    if not candidatos_ok:
        return []

    # Orden global por compliance
    ordenados = sorted(candidatos_ok, key=lambda c: c["Ci"])

    bajos = ordenados[:n_extremos]
    altos = ordenados[-n_extremos:]

    # unir evitando duplicados
    seleccionados = []
    claves = set()

    for c in bajos + altos:
        clave = (
            tuple(c["ventana"]),
            c["tipo"],
            round(c["Ci"], 12),
            c["n_puntos_originales"]
        )
        if clave not in claves:
            seleccionados.append(c)
            claves.add(clave)

    # opcional: ordenar por Ci antes de devolver
    seleccionados = sorted(seleccionados, key=lambda c: c["Ci"])

    return seleccionados

def seleccionar_6_extremos(candidatos_todos, n_extremos=3):
    """
    Devuelve hasta 6 candidatos:
    - 3 con menor Ci
    - 3 con mayor Ci
    """
    candidatos_ok = [
        c for c in candidatos_todos
        if c is not None and np.isfinite(c["Ci"]) and c["Ci"] > 0
    ]

    if not candidatos_ok:
        return []

    candidatos_ordenados = sorted(candidatos_ok, key=lambda c: c["Ci"])

    bajos = candidatos_ordenados[:n_extremos]
    altos = candidatos_ordenados[-n_extremos:]

    seleccionados = bajos + altos

    # eliminar duplicados si se superponen
    unicos = []
    usados = set()

    for c in seleccionados:
        clave = (
            round(c["Ci"], 12),
            tuple(c["ventana"]),
            c["tipo"]
        )
        if clave not in usados:
            unicos.append(c)
            usados.add(clave)

    return sorted(unicos, key=lambda c: c["Ci"])

def resumir_candidatos_descarga(numero_descarga, candidatos_seleccionados, n_cols=6):
    """
    Devuelve un diccionario tipo fila para tabla:
    Descarga, C1 ... C6
    """
    fila = {"Descarga": numero_descarga}

    for i in range(n_cols):
        if i < len(candidatos_seleccionados):
            fila[f"C{i+1}"] = candidatos_seleccionados[i]["Ci"]
        else:
            fila[f"C{i+1}"] = np.nan

    return fila

def mapear_opciones_descarga(candidatos_seleccionados, n_cols=6):
    """
    Devuelve un diccionario:
    {
        "C1": candidato_dict,
        ...
        "C6": candidato_dict
    }
    """
    mapa = {}

    for i in range(n_cols):
        clave = f"C{i+1}"
        if i < len(candidatos_seleccionados):
            mapa[clave] = candidatos_seleccionados[i]
        else:
            mapa[clave] = None

    return mapa

# =========================================================
# TABLA SUPERPUESTA
# =========================================================
def agregar_tabla_superpuesta(ax, seleccionadas_previas):
    """
    Agrega una tabla con las compliances previamente elegidas
    dentro del gráfico actual.
    """
    previas_validas = [
        r for r in seleccionadas_previas
        if r.get("seleccionada", False) and np.isfinite(r.get("Ci", np.nan))
    ]

    if len(previas_validas) == 0:
        return

    ultimas = previas_validas[-8:]  # últimas 8 para no tapar demasiado

    cell_text = []
    for r in ultimas:
        cell_text.append([
            str(r["descarga"]),
            str(r.get("opcion_elegida", "")),
            f"{r['Ci']:.6f}",
            f"{r['R2']:.4f}",
        ])

    col_labels = ["Desc.", "Opción", "Ci", "R²"]

    tabla = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        bbox=[0.60, 0.02, 0.38, 0.28],  # x, y, ancho, alto
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)

    for (fila, col), cell in tabla.get_celld().items():
        cell.set_alpha(0.9)
        if fila == 0:
            cell.set_text_props(weight="bold")

# =========================================================
# GRÁFICOS
# =========================================================
def graficar_curva(x, y, nombre_archivo, etiqueta_x, etiqueta_y, titulo, ruta_guardado=None, mostrar=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=1.5)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.set_title(f"{titulo}\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    guardar_o_mostrar_figura(fig, ruta_guardado=ruta_guardado, mostrar=mostrar)


def graficar_descarga_con_opciones(carga_seg, cmod_seg, opciones_dict, numero_descarga, nombre_archivo, seleccionadas_previas):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Puntos experimentales de toda la descarga
    ax.scatter(
        cmod_seg,
        carga_seg,
        s=16,
        c="black",
        alpha=0.75,
        label="Puntos experimentales de la descarga"
    )

    # Rectas candidatas C1-C6
    for k in range(1, 7):
        clave = f"C{k}"
        cand = opciones_dict.get(clave, None)

        if cand is None:
            continue

        xw = np.asarray(cand["xw"], dtype=float)   # carga
        yw = np.asarray(cand["yw"], dtype=float)   # CMOD

        if len(xw) < 2 or len(yw) < 2:
            continue

        m = cand["Ci"]
        b = cand["intercepto"]

        # Ajuste original: CMOD = Ci * Carga + b
        xfit = np.linspace(np.min(xw), np.max(xw), 100)  # carga
        yfit = m * xfit + b                              # CMOD

        ax.plot(
            yfit,
            xfit,
            "--",
            linewidth=2.0,
            label=f"{clave}: Ci={cand['Ci']:.6f} | R²={cand['R2']:.4f}"
        )

        # Puntos usados en ese ajuste
        ax.plot(
            yw,
            xw,
            "o",
            markersize=5,
            alpha=0.9
        )

        # Etiqueta C1, C2, etc. en el centro de la recta
        ax.text(
            np.mean(yfit),
            np.mean(xfit),
            clave,
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85)
        )

    agregar_tabla_superpuesta(ax, seleccionadas_previas)

    ax.set_xlabel("CMOD [mm]")
    ax.set_ylabel("Carga [kN]")
    ax.set_title(f"Descarga {numero_descarga} - Selección supervisada de compliance\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.show(block=True)

def graficar_compliance_vs_cmod(cmod_medios, compliances, descargas, nombre_archivo, mostrar_numeros=True, ruta_guardado=None, mostrar=True):
    pares = [
        (x, y, d)
        for x, y, d in zip(cmod_medios, compliances, descargas)
        if not np.isnan(x) and not np.isnan(y)
    ]

    if len(pares) == 0:
        print("\nNo hay datos seleccionados para graficar Compliance vs CMOD.")
        return

    pares = sorted(pares, key=lambda t: t[0])

    xv = [p[0] for p in pares]
    yv = [p[1] for p in pares]
    dv = [p[2] for p in pares]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xv, yv, s=45)

    if mostrar_numeros:
        for x, y, d in zip(xv, yv, dv):
            ax.text(x, y, str(d), fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("CMOD al inicio de descarga [mm]")
    ax.set_ylabel("Compliance, C = d(CMOD)/dP [mm/kN]")
    ax.set_title(f"Compliance vs CMOD\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    guardar_o_mostrar_figura(fig, ruta_guardado=ruta_guardado, mostrar=mostrar)

def graficar_delta_a_vs_cmod(resultados_seleccionados, nombre_archivo, ruta_guardado=None, mostrar=True):
    pares = [
        (r["x_descarga"], r["delta_a_mm"], r["descarga"])
        for r in resultados_seleccionados
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not np.isnan(r.get("delta_a_mm", np.nan))
        )
    ]

    if len(pares) == 0:
        print("No hay datos para graficar Δa vs CMOD.")
        return

    pares = sorted(pares, key=lambda t: t[0])

    xv = [p[0] for p in pares]
    yv = [p[1] for p in pares]
    dv = [p[2] for p in pares]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xv, yv, s=45)

    for x, y, d in zip(xv, yv, dv):
        ax.text(x, y, str(d), fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("CMOD al inicio de descarga [mm]")
    ax.set_ylabel("Δa [mm]")
    ax.set_title(f"Δa vs CMOD\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    guardar_o_mostrar_figura(fig, ruta_guardado=ruta_guardado, mostrar=mostrar)

def loop_eliminacion_interactiva(resultados_seleccionados, nombre_archivo):
    """
    Permite eliminar puntos manualmente de la curva final.
    El usuario escribe el número de descarga a eliminar.
    Cuando escribe 0, termina y se muestra el gráfico final sin numeritos.
    """
    while True:
        activos = [
            r for r in resultados_seleccionados
            if r["seleccionada"] and not r.get("eliminada", False)
            and not np.isnan(r["Ci"]) and not np.isnan(r["cmod_medio"])
        ]

        if len(activos) == 0:
            print("\nNo quedan puntos activos para graficar.")
            return

        cmod_medios = [r["x_descarga"] for r in activos]
        compliances = [r["Ci"] for r in activos]
        descargas = [r["descarga"] for r in activos]

        graficar_compliance_vs_cmod(
            cmod_medios,
            compliances,
            descargas,
            nombre_archivo,
            mostrar_numeros=True
        )

        print("\nPuntos actualmente mostrados en la gráfica:")
        print(", ".join(str(d) for d in descargas))
        print("[0] Terminar eliminación y mostrar gráfico final")

        entrada = input("Ingrese el número de descarga que desea eliminar: ").strip()

        try:
            opcion = int(entrada)
        except ValueError:
            print("Entrada inválida. Ingrese un número.")
            continue

        if opcion == 0:
            activos_finales = [
                r for r in resultados_seleccionados
                if r["seleccionada"] and not r.get("eliminada", False)
                and not np.isnan(r["Ci"]) and not np.isnan(r["cmod_medio"])
            ]

            cmod_medios_final = [r["x_descarga"] for r in activos_finales]
            compliances_final = [r["Ci"] for r in activos_finales]
            descargas_final = [r["descarga"] for r in activos_finales]

            graficar_compliance_vs_cmod(
                cmod_medios_final,
                compliances_final,
                descargas_final,
                nombre_archivo,
                mostrar_numeros=False
            )
            return

        encontrados = [
            r for r in resultados_seleccionados
            if r["descarga"] == opcion and r["seleccionada"] and not r.get("eliminada", False)
        ]

        if len(encontrados) == 0:
            print(f"No existe un punto activo con número de descarga {opcion}.")
            continue

        encontrados[0]["eliminada"] = True
        print(f"Se eliminó de la gráfica el punto correspondiente a la descarga {opcion}.")

def graficar_J_vs_delta_a(resultados_seleccionados, nombre_archivo, ruta_guardado=None, mostrar=True):
    pares = []

    for r in resultados_seleccionados:
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not r.get("eliminada_jr", False)
        ):
            delta_a = r.get("delta_a_mm", np.nan)
            if not np.isfinite(delta_a):
                delta_a = r.get("delta_a_debug_mm", np.nan)

            J = r.get("J_total_kJ_m2", np.nan)

            if np.isfinite(delta_a) and np.isfinite(J):
                pares.append((delta_a, J, r["descarga"]))

    if len(pares) == 0:
        print("No hay puntos graficables para J vs Δa.")
        return

    pares = sorted(pares, key=lambda t: t[0])

    xv = [p[0] for p in pares]
    yv = [p[1] for p in pares]
    dv = [p[2] for p in pares]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xv, yv, s=45, label="Puntos J-R")

    for x, y, d in zip(xv, yv, dv):
        ax.text(x, y, str(d), fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("Δa [mm]")
    ax.set_ylabel("J [kJ/m²]")
    ax.set_title(f"Curva J-R: J vs Δa\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    guardar_o_mostrar_figura(fig, ruta_guardado=ruta_guardado, mostrar=mostrar)
    

def filtrar_puntos_astm(pares, sigma_y_MPa, b0_mm):
    """
    Filtra y etiqueta puntos ASTM de forma consistente con enriquecer_puntos_astm.

    Devuelve:
    - lista de puntos enriquecidos
    - J_limit
    - Δa_limit
    """
    return enriquecer_puntos_astm(
        pares=pares,
        sigma_y_MPa=sigma_y_MPa,
        b0_mm=b0_mm
    )


def encontrar_jq(c1, c2, sigma_y_MPa, k_mm=1.0, da_min=1e-6, da_max=3.0, n=5000):
    """
    Intersección entre la curva ajustada y la offset line a 0.2 mm:
    J = 2 sigma_y (Δa - 0.2)
    """
    da_grid = np.linspace(da_min, da_max, n)
    jr = evaluar_potencia_jr(da_grid, c1, c2, k_mm=k_mm)
    offset = 2.0 * sigma_y_MPa * (da_grid - 0.2)

    diff = jr - offset
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    if len(idx) == 0:
        return np.nan, np.nan

    i = idx[0]
    x1, x2 = da_grid[i], da_grid[i + 1]
    y1, y2 = diff[i], diff[i + 1]

    da_q = x1 - y1 * (x2 - x1) / (y2 - y1)
    J_q = c1 * (da_q / k_mm) ** c2
    return da_q, J_q


def evaluar_validez_jq(Jq_kJ_m2, B_mm, b0_mm, sigma_y_MPa, da_q_mm):
    """
    Criterio simplificado según tu PDF:
    B, b0 > 10 * Jq / sigma_y
    """
    if np.isnan(Jq_kJ_m2):
        return {"cumple_tamano": False, "B_req": np.nan, "tipo": "sin_interseccion"}

    B_req = 10.0 * Jq_kJ_m2 / sigma_y_MPa
    cumple_tamano = (B_mm > B_req) and (b0_mm > B_req)

    limite_transicion = 0.2 + Jq_kJ_m2 / (2.0 * sigma_y_MPa)

    if da_q_mm < limite_transicion:
        tipo = "Jc"
    else:
        tipo = "Ju"

    return {
        "cumple_tamano": cumple_tamano,
        "B_req": B_req,
        "limite_transicion_mm": limite_transicion,
        "tipo": tipo
    }

def graficar_analisis_astm_jr(
    pares_filtrados,
    c1,
    c2,
    sigma_y_MPa,
    j_limit,
    da_limit,
    da_q,
    J_q,
    nombre_archivo,
    ruta_guardado=None,
    mostrar=True
):
    fig, ax = plt.subplots(figsize=(9, 6))

    todos_x = np.array([p.get("delta_a_astm_mm", p.get("delta_a_mm", np.nan)) for p in pares_filtrados], dtype=float)
    todos_y = np.array([p.get("J_kJ_m2", np.nan) for p in pares_filtrados], dtype=float)

    # usar SIEMPRE el mismo campo lógico
    validos = [p for p in pares_filtrados if p.get("valido_astm", False)]
    invalidos = [p for p in pares_filtrados if not p.get("valido_astm", False)]

    x_validos = np.array([p.get("delta_a_astm_mm", p.get("delta_a_mm", np.nan)) for p in validos], dtype=float)
    y_validos = np.array([p.get("J_kJ_m2", np.nan) for p in validos], dtype=float)

    x_invalidos = np.array([p.get("delta_a_astm_mm", p.get("delta_a_mm", np.nan)) for p in invalidos], dtype=float)
    y_invalidos = np.array([p.get("J_kJ_m2", np.nan) for p in invalidos], dtype=float)

    mask_v = np.isfinite(x_validos) & np.isfinite(y_validos)
    x_validos = x_validos[mask_v]
    y_validos = y_validos[mask_v]

    mask_i = np.isfinite(x_invalidos) & np.isfinite(y_invalidos)
    x_invalidos = x_invalidos[mask_i]
    y_invalidos = y_invalidos[mask_i]

    if len(x_validos) > 0:
        ax.scatter(
            x_validos,
            y_validos,
            s=55,
            label="Puntos válidos ASTM",
            zorder=3
        )

    if len(x_invalidos) > 0:
        ax.scatter(
            x_invalidos,
            y_invalidos,
            s=55,
            marker="x",
            color="orange",
            label="Puntos excluidos",
            zorder=3
        )

    for p in pares_filtrados:
        x_txt = p.get("delta_a_astm_mm", p.get("delta_a_mm", np.nan))
        y_txt = p.get("J_kJ_m2", np.nan)

        if np.isfinite(x_txt) and np.isfinite(y_txt):
            ax.text(
                x_txt,
                y_txt,
                f' {p["descarga"]}',
                fontsize=9,
                va="bottom"
            )

    x_ref = []
    if np.any(np.isfinite(todos_x)):
        x_ref.append(np.nanmax(todos_x))
    if np.isfinite(da_limit):
        x_ref.append(da_limit)
    if np.isfinite(da_q):
        x_ref.append(da_q)

    da_max_plot = max(x_ref) * 1.10 if len(x_ref) > 0 else 1.6
    da_max_plot = max(da_max_plot, 1.6)

    da_grid = np.linspace(0.0, da_max_plot, 500)

    blunting = 2.0 * sigma_y_MPa * da_grid
    excl_015 = 2.0 * sigma_y_MPa * (da_grid - 0.15)
    excl_150 = 2.0 * sigma_y_MPa * (da_grid - 1.50)
    offset_02 = 2.0 * sigma_y_MPa * (da_grid - 0.2)

    ax.plot(
        da_grid, blunting,
        linestyle="--", linewidth=1.2,
        label="Construction / blunting line",
        zorder=1
    )
    ax.plot(
        da_grid, excl_015,
        linestyle="--", linewidth=1.0,
        label="Exclusion line 0.15 mm",
        zorder=1
    )
    ax.plot(
        da_grid, excl_150,
        linestyle="--", linewidth=1.0,
        label="Exclusion line 1.50 mm",
        zorder=1
    )
    ax.plot(
        da_grid, offset_02,
        linestyle=":", linewidth=1.5,
        label="Offset line 0.2 mm",
        zorder=1
    )

    # Ajuste solo con puntos válidos
    if np.isfinite(c1) and np.isfinite(c2) and len(x_validos) >= 2:
        x_fit_base = x_validos[np.isfinite(x_validos) & (x_validos > 0)]

        if len(x_fit_base) >= 2:
            da_fit_min = np.nanmin(x_fit_base)
            da_fit_max = np.nanmax(x_fit_base)

            if da_fit_max > da_fit_min:
                da_fit_grid = np.linspace(da_fit_min, da_fit_max, 300)
                jr_fit = evaluar_potencia_jr(da_fit_grid, c1, c2)

                ax.plot(
                    da_fit_grid,
                    jr_fit,
                    linewidth=2.2,
                    color="red",
                    label=f"Ajuste potencia: J = {c1:.3f}(Δa)^{c2:.3f}",
                    zorder=2
                )

    if np.isfinite(j_limit):
        ax.axhline(
            j_limit,
            linestyle="--",
            linewidth=1.0,
            label="J_limit"
        )

    if np.isfinite(da_limit):
        ax.axvline(
            da_limit,
            linestyle="--",
            linewidth=1.0,
            label="Δa_limit"
        )

    if np.isfinite(da_q) and np.isfinite(J_q):
        ax.scatter(
            [da_q], [J_q],
            s=85,
            marker="s",
            label=f"JQ = {J_q:.1f} kJ/m²",
            zorder=4
        )
        ax.text(da_q, J_q, f"  JQ={J_q:.1f}", va="bottom")

    ax.set_xlabel("Δa [mm]")
    ax.set_ylabel("J [kJ/m²]")
    ax.set_title(f"Análisis ASTM posterior de curva J-R\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)

    y_candidatos = []
    if np.any(np.isfinite(todos_y)):
        y_candidatos.append(np.nanmax(todos_y))
    if np.isfinite(j_limit):
        y_candidatos.append(j_limit)
    if np.isfinite(J_q):
        y_candidatos.append(J_q)

    if len(y_candidatos) > 0:
        y_min_plot = min(0.0, np.nanmin(todos_y) * 1.10) if np.any(np.isfinite(todos_y)) else 0.0
        y_max_plot = max(y_candidatos) * 1.20
        ax.set_ylim(y_min_plot, y_max_plot)

    ax.legend(fontsize=8)
    fig.tight_layout()

    guardar_o_mostrar_figura(fig, ruta_guardado=ruta_guardado, mostrar=mostrar)
    
def analisis_astm_post_jr(resultados_seleccionados, nombre_archivo, W, a0, B, sigma_y_MPa,
                          ruta_guardado=None, mostrar=True):
    """
    Postproceso ASTM sobre la curva J-R ya calculada.

    - Siempre intenta graficar todos los puntos graficables.
    - Marca cuáles quedan válidos ASTM y cuáles no.
    - Sólo ajusta ley de potencia si hay al menos 2 válidos ASTM.
    """
    b0 = W - a0
    if b0 <= 0:
        print("Error: b0 <= 0. Revise W y a0.")
        return None

    pares_graficables = obtener_pares_jr_graficables(resultados_seleccionados)

    if len(pares_graficables) == 0:
        print("No hay puntos J-R graficables para análisis ASTM.")
        return None

    pares_filtrados, j_limit, da_limit = filtrar_puntos_astm(pares_graficables, sigma_y_MPa, b0)
    validos = [p for p in pares_filtrados if p.get("valido_astm", False)]

    print("\n" + "=" * 100)
    print("ANÁLISIS ASTM POSTERIOR A LA CURVA J-R")
    print("=" * 100)
    print(f"b0 = W - a0 = {b0:.4f} mm")
    print(f"J_limit = {j_limit:.4f} kJ/m²")
    print(f"Δa_limit = {da_limit:.4f} mm")
    print(f"Cantidad total de puntos J-R graficables = {len(pares_filtrados)}")
    print(f"Cantidad de puntos válidos ASTM = {len(validos)}")

    tabla = pd.DataFrame(pares_filtrados)
    print("\nTabla ASTM de puntos:")
    print(tabla.to_string(index=False))

    if len(validos) >= 5:
        delta_a_valid = np.array([p["delta_a_mm"] for p in validos], dtype=float)
        J_valid = np.array([p["J_kJ_m2"] for p in validos], dtype=float)

        c1, c2, r2 = ajustar_potencia_jr(delta_a_valid, J_valid, k_mm=1.0)
        da_q, J_q = encontrar_jq(
            c1, c2, sigma_y_MPa, k_mm=1.0,
            da_max=max(3.0, da_limit * 1.5 if np.isfinite(da_limit) else 3.0)
        )
        validez = evaluar_validez_jq(J_q, B, b0, sigma_y_MPa, da_q)

        print(f"Ajuste potencia: J = {c1:.4f} * (Δa/1.0)^{c2:.4f}")
        print(f"R² ajuste log-log = {r2:.5f}")

        if not np.isnan(J_q):
            print(f"Δa_q = {da_q:.4f} mm")
            print(f"J_Q = {J_q:.4f} kJ/m²")
            print(f"B requerido = {validez['B_req']:.4f} mm")
            print(f"¿Cumple tamaño? {'Sí' if validez['cumple_tamano'] else 'No'}")
            print(f"Clasificación tentativa = {validez['tipo']}")
        else:
            print("No se encontró intersección entre curva ajustada y offset line de 0.2 mm.")
    else:
        c1 = np.nan
        c2 = np.nan
        r2 = np.nan
        da_q = np.nan
        J_q = np.nan
        validez = {
            "cumple_tamano": False,
            "B_req": np.nan,
            "tipo": "sin_ajuste",
            "limite_transicion_mm": np.nan
        }
        print("No quedaron suficientes puntos válidos ASTM para ajustar la curva J-R.")
        print("Se grafica igualmente el conjunto total, distinguiendo válidos y excluidos.")

    graficar_analisis_astm_jr(
        pares_filtrados=pares_filtrados,
        c1=c1,
        c2=c2,
        sigma_y_MPa=sigma_y_MPa,
        j_limit=j_limit,
        da_limit=da_limit,
        da_q=da_q,
        J_q=J_q,
        nombre_archivo=nombre_archivo,
        ruta_guardado=ruta_guardado,
        mostrar=mostrar
    )

    return {
        "tabla": tabla,
        "b0_mm": b0,
        "J_limit": j_limit,
        "da_limit": da_limit,
        "c1": c1,
        "c2": c2,
        "r2": r2,
        "da_q_mm": da_q,
        "J_q_kJ_m2": J_q,
        "validez": validez
    }
def preguntar_correccion_rotacion():
    while True:
        entrada = input("\n¿Desea aplicar corrección por rotación para mini C(T)? [s/n]: ").strip().lower()

        if entrada in ["n", "no"]:
            return {"usar": False}

        if entrada in ["s", "si", "sí"]:
            break

        print("Respuesta inválida. Escriba 's' o 'n'.")

    print("\nIngrese los datos geométricos para la corrección por rotación:")
    print("(Use unidades coherentes; por ejemplo, mm y MPa)")

    while True:
        try:
            W = float(input("W = ancho de la probeta [mm]: ").strip())
            B = float(input("B = espesor de la probeta [mm]: ").strip())

            txt_bn = input("BN = espesor neto [mm] (Enter si no hay side-grooves): ").strip()
            if txt_bn == "":
                BN = B
            else:
                BN = float(txt_bn)

            H_ast = float(input("H* = mitad de la distancia entre centros de pernos [mm]: ").strip())
            d = float(input("d = distancia entre puntos de medición del clip gauge [mm]: ").strip())
            E = float(input("E = módulo elástico [MPa]: ").strip())
            nu = float(input("nu = coeficiente de Poisson [-]: ").strip())

            return {
                "usar": True,
                "W": W,
                "B": B,
                "BN": BN,
                "H_ast": H_ast,
                "d": d,
                "E": E,
                "nu": nu,
            }

        except ValueError:
            print("Valor inválido. Ingrese números.")


def calcular_be(B, BN):
    return B - ((B - BN) ** 2) / B


def calcular_ai_desde_compliance_ct(Cc, W, B, BN, E, nu):
    """
    Estima a_i a partir de compliance corregida para C(T).
    Devuelve: ai [mm], aW [-], u [-]
    """
    Be = calcular_be(B, BN)
    Eprima = E / (1.0 - nu ** 2)

    Cc_N = Cc / 1000.0   # mm/kN -> mm/N
    valor = Be * Eprima * Cc_N

    if valor <= 0:
        return np.nan, np.nan, np.nan

    u = 1.0 / (np.sqrt(valor) + 1.0)

    aW = (
        1.000196
        - 4.06319 * u
        + 11.242 * u**2
        - 106.043 * u**3
        + 464.335 * u**4
        - 650.677 * u**5
    )

    ai = aW * W
    return ai, aW, u


def corregir_rotacion_ct_iterativo(Ci, v_m, W, B, BN, H_ast, d, E, nu,
                                   max_iter=20, tol=1e-8):
    """
    Corrección iterativa por rotación para mini C(T).
    Devuelve:
        Cc, ai, aW, Ri, theta_rad, theta_deg, u
    """
    if np.isnan(Ci) or Ci <= 0 or np.isnan(v_m):
        return {
            "Cc": np.nan,
            "ai": np.nan,
            "aW": np.nan,
            "Ri": np.nan,
            "theta_rad": np.nan,
            "theta_deg": np.nan,
            "u": np.nan,
        }

    Cc = Ci
    D = d / 2.0

    for _ in range(max_iter):
        ai, aW, u = calcular_ai_desde_compliance_ct(Cc, W, B, BN, E, nu)

        if np.isnan(ai):
            return {
                "Cc": np.nan,
                "ai": np.nan,
                "aW": np.nan,
                "Ri": np.nan,
                "theta_rad": np.nan,
                "theta_deg": np.nan,
                "u": np.nan,
            }

        Ri = (W + ai) / 2.0

        denom = 2.0 * np.sqrt(D**2 + Ri**2)
        if denom <= 0:
            return {
                "Cc": np.nan,
                "ai": np.nan,
                "aW": np.nan,
                "Ri": np.nan,
                "theta_rad": np.nan,
                "theta_deg": np.nan,
                "u": np.nan,
            }

        arg = np.clip(v_m / denom, -1.0, 1.0)
        theta = np.arcsin(arg) - np.arctan(D / Ri)

        f1 = (H_ast / Ri) * np.sin(theta) - np.cos(theta)
        f2 = (D / Ri) * np.sin(theta) - np.cos(theta)
        factor = f1 * f2

        if abs(factor) < 1e-14:
            return {
                "Cc": np.nan,
                "ai": np.nan,
                "aW": np.nan,
                "Ri": np.nan,
                "theta_rad": np.nan,
                "theta_deg": np.nan,
                "u": np.nan,
            }

        Cc_nueva = Ci / factor

        if np.isnan(Cc_nueva) or Cc_nueva <= 0:
            return {
                "Cc": np.nan,
                "ai": np.nan,
                "aW": np.nan,
                "Ri": np.nan,
                "theta_rad": np.nan,
                "theta_deg": np.nan,
                "u": np.nan,
            }

        if abs(Cc_nueva - Cc) < tol:
            Cc = Cc_nueva
            break

        Cc = Cc_nueva

    ai, aW, u = calcular_ai_desde_compliance_ct(Cc, W, B, BN, E, nu)
    Ri = (W + ai) / 2.0

    denom = 2.0 * np.sqrt(D**2 + Ri**2)
    arg = np.clip(v_m / denom, -1.0, 1.0)
    theta = np.arcsin(arg) - np.arctan(D / Ri)

    return {
        "Cc": Cc,
        "ai": ai,
        "aW": aW,
        "Ri": Ri,
        "theta_rad": theta,
        "theta_deg": np.degrees(theta),
        "u": u,
    }


def aplicar_correccion_rotacion_a_resultados(resultados_seleccionados, rot_cfg):
    """
    Aplica la corrección por rotación SOLO a los puntos
    que sobrevivieron a la revisión manual de la curva original.
    """
    for r in resultados_seleccionados:
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not np.isnan(r["Ci"])
            and not np.isnan(r["v_inicio_descarga"])
        ):
            rot = corregir_rotacion_ct_iterativo(
                Ci=r["Ci"],
                v_m=r["v_inicio_descarga"],
                W=rot_cfg["W"],
                B=rot_cfg["B"],
                BN=rot_cfg["BN"],
                H_ast=rot_cfg["H_ast"],
                d=rot_cfg["d"],
                E=rot_cfg["E"],
                nu=rot_cfg["nu"],
                max_iter=CONFIG["iter_rotacion"],
                tol=CONFIG["tol_rotacion"]
            )

            r["Ci_final"] = rot["Cc"]
            r["a_i_mm"] = rot["ai"]
            r["a_i_W"] = rot["aW"]
            r["R_i_mm"] = rot["Ri"]
            r["theta_deg"] = rot["theta_deg"]
            r["u"] = rot["u"]
        else:
            r["Ci_final"] = np.nan
            r["a_i_mm"] = np.nan
            r["a_i_W"] = np.nan
            r["R_i_mm"] = np.nan
            r["theta_deg"] = np.nan
            r["u"] = np.nan


def loop_eliminacion_interactiva_corregida(resultados_seleccionados, nombre_archivo):
    """
    Revisión manual de la curva corregida por rotación.
    """
    while True:
        activos = [
            r for r in resultados_seleccionados
            if r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not np.isnan(r["Ci_final"])
            and not np.isnan(r["cmod_medio"])
        ]

        if len(activos) == 0:
            print("\nNo quedan puntos activos para graficar la curva corregida.")
            return

        cmod_medios = [r["x_descarga"] for r in activos]
        compliances = [r["Ci_final"] for r in activos]
        descargas = [r["descarga"] for r in activos]

        graficar_compliance_vs_cmod(
            cmod_medios,
            compliances,
            descargas,
            f"{nombre_archivo} - corregida por rotación",
            mostrar_numeros=True
        )

        print("\nPuntos actualmente mostrados en la gráfica corregida:")
        print(", ".join(str(d) for d in descargas))
        print("[0] Terminar eliminación y mostrar gráfico final corregido")

        entrada = input("Ingrese el número de descarga que desea eliminar de la curva corregida: ").strip()

        try:
            opcion = int(entrada)
        except ValueError:
            print("Entrada inválida. Ingrese un número.")
            continue

        if opcion == 0:
            activos_finales = [
                r for r in resultados_seleccionados
                if r["seleccionada"]
                and not r.get("eliminada", False)
                and not r.get("eliminada_corregida", False)
                and not np.isnan(r["Ci_final"])
                and not np.isnan(r["cmod_medio"])
            ]

            cmod_medios_final = [r["x_descarga"] for r in activos_finales]
            compliances_final = [r["Ci_final"] for r in activos_finales]
            descargas_final = [r["descarga"] for r in activos_finales]

            graficar_compliance_vs_cmod(
                cmod_medios_final,
                compliances_final,
                descargas_final,
                f"{nombre_archivo} - corregida por rotación",
                mostrar_numeros=False
            )
            return

        encontrados = [
            r for r in resultados_seleccionados
            if r["descarga"] == opcion
            and r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
        ]

        if len(encontrados) == 0:
            print(f"No existe un punto activo con número de descarga {opcion} en la curva corregida.")
            continue

        encontrados[0]["eliminada_corregida"] = True
        print(f"Se eliminó de la gráfica corregida el punto correspondiente a la descarga {opcion}.")

def mostrar_tabla_delta_a(resultados_seleccionados, a0q):
    filas = []

    for r in resultados_seleccionados:
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not np.isnan(r.get("a_i_mm", np.nan))
        ):
            filas.append({
                "descarga": r["descarga"],
                "x_descarga_mm": r["x_descarga"],
                "Ci_final": r["Ci_final"],
                "u": r["u"],
                "a_i_mm": r["a_i_mm"],
                "a_i_W": r["a_i_W"],
                "Δa_mm": r["delta_a_mm"],
            })

    if len(filas) == 0:
        print("No hay datos válidos para tabla Δa.")
        return

    tabla = pd.DataFrame(filas)

    print("\n" + "=" * 100)
    print("TABLA FINAL: GRIETA Y CRECIMIENTO")
    print("=" * 100)
    print(tabla.to_string(index=False))

    print(f"\na0q utilizado = {a0q:.6f} mm")

# =========================================================
# INTERACCIÓN CON EL USUARIO
# =========================================================
def elegir_opcion_por_defecto(candidatos, resultados_seleccionados):
    """
    Devuelve el candidato por defecto:
    la menor compliance estrictamente mayor a la última compliance elegida,
    pero sin exceder un salto relativo máximo.

    Si no existe una opción razonable, devuelve None.
    """
    seleccionadas_previas = [
        r for r in resultados_seleccionados
        if r["seleccionada"] and not np.isnan(r["Ci"])
    ]

    if len(seleccionadas_previas) == 0:
        return None

    ci_anterior = seleccionadas_previas[-1]["Ci"]
    tol_rel = CONFIG["tolerancia_default_rel"]
    ci_max_aceptable = ci_anterior * (1.0 + tol_rel)

    candidatos_validos = [
        c for c in candidatos
        if c["Ci"] > ci_anterior and c["Ci"] <= ci_max_aceptable
    ]

    if len(candidatos_validos) == 0:
        return None

    return min(candidatos_validos, key=lambda c: c["Ci"])

def pedir_opcion_usuario(candidatos, numero_descarga, resultados_seleccionados):
    if len(candidatos) == 0:
        print(f"\nDescarga {numero_descarga}: no hay opciones válidas con R² >= {CONFIG['r2_min']:.2f}.")
        return None

    opcion_defecto = elegir_opcion_por_defecto(candidatos, resultados_seleccionados)

    seleccionadas_previas = [
        r for r in resultados_seleccionados
        if r["seleccionada"] and not np.isnan(r["Ci"])
    ]

    if len(seleccionadas_previas) > 0:
        ci_anterior = seleccionadas_previas[-1]["Ci"]
        ci_max_default = ci_anterior * (1.0 + CONFIG["tolerancia_default_rel"])
    else:
        ci_anterior = np.nan
        ci_max_default = np.nan

    print("\n" + "=" * 100)
    print(f"DESCARGA {numero_descarga} - OPCIONES DE COMPLIANCE")
    print("=" * 100)

    for i, cand in enumerate(candidatos, start=1):
        marca = ""
        if opcion_defecto is not None and cand is opcion_defecto:
            marca = "  <-- default"

        print(
            f"[{i}] "
            f"Ci = {cand['Ci']:.6f} mm/kN | "
            f"R² = {cand['R2']:.4f} | "
            f"Ventana = {cand['ventana']} | "
            f"ΔCMOD = {cand['delta_cmod']:.5f} mm | "
            f"Fracción rango carga = {cand['fraccion_rango_carga']:.3f} | "
            f"CMOD medio = {cand['cmod_medio']:.5f} mm"
            f"{marca}"
        )

    print("[0] Saltar esta descarga")

    if not np.isnan(ci_anterior):
        print(
            f"Última compliance elegida = {ci_anterior:.6f} mm/kN | "
            f"Máximo aceptable para default = {ci_max_default:.6f} mm/kN"
        )

    if opcion_defecto is not None:
        print(
            f"Enter = tomar opción por defecto: "
            f"Ci = {opcion_defecto['Ci']:.6f} mm/kN"
        )
    else:
        print("Enter = no hay opción por defecto razonable, se salta la descarga")

    while True:
        entrada = input(f"Seleccione opción para la descarga {numero_descarga}: ").strip()

        if entrada == "":
            return opcion_defecto

        try:
            opcion = int(entrada)
        except ValueError:
            print("Entrada inválida. Ingrese un número o presione Enter.")
            continue

        if opcion == 0:
            return None

        if 1 <= opcion <= len(candidatos):
            return candidatos[opcion - 1]

        print("Opción fuera de rango.")

def preguntar_si_calcular_a0q():
    while True:
        entrada = input(
            "\n¿Desea pasar a la determinación del tamaño inicial de grieta a0q por compliance? [s/n]: "
        ).strip().lower()

        if entrada in ["s", "si", "sí"]:
            return True
        if entrada in ["n", "no"]:
            return False

        print("Respuesta inválida. Escriba 's' o 'n'.")

def pedir_datos_a0q():
    print("\nIngrese los datos necesarios para calcular a0q por compliance.")
    print("(Use unidades coherentes; por ejemplo, mm y MPa)")

    while True:
        try:
            W = float(input("W = ancho de la probeta [mm]: ").strip())
            B = float(input("B = espesor de la probeta [mm]: ").strip())

            txt_bn = input("BN = espesor neto [mm] (Enter si no hay side-grooves): ").strip()
            if txt_bn == "":
                BN = B
            else:
                BN = float(txt_bn)

            E = float(input("E = módulo elástico [MPa]: ").strip())
            nu = float(input("nu = coeficiente de Poisson [-]: ").strip())

            return {
                "W": W,
                "B": B,
                "BN": BN,
                "E": E,
                "nu": nu
            }
        except ValueError:
            print("Valor inválido. Ingrese números.")

def recalculo_interactivo_a0q(resultados_seleccionados, W, B, BN, E, nu):
    """
    Calcula a0q con todas las descargas activas corregidas.
    Si no cumple, pregunta si desea recalcular con las primeras N descargas
    o seguir adelante.
    """
    n_primeras = None

    while True:
        resultado = determinar_a0q_por_compliance_corregida(
            resultados_seleccionados=resultados_seleccionados,
            W=W,
            B=B,
            BN=BN,
            E=E,
            nu=nu,
            n_primeras=n_primeras
        )

        print("\n" + "=" * 100)
        print("DETERMINACIÓN DE a0q POR COMPLIANCE CORREGIDA")
        print("=" * 100)
        print(resultado["tabla"].to_string(index=False))

        if not np.isnan(resultado["a0q"]):
            print(f"\na0q promedio = {resultado['a0q']:.6f} mm")
        else:
            print("\na0q promedio = NaN")

        if "tolerancia_mm" in resultado:
            print(f"Tolerancia individual = ±{resultado['tolerancia_mm']:.6f} mm (0.002 W)")

        print(f"Resultado: {resultado['motivo']}")

        if resultado["cumple"]:
            print("El valor promedio a0q se acepta como tamaño inicial de grieta.")
            return resultado

        print("El valor promedio a0q NO se acepta como tamaño inicial de grieta.")

        while True:
            print("\nOpciones:")
            print("[1] Recalcular usando las primeras N descargas activas corregidas")
            print("[2] Seguir adelante de todos modos")

            entrada = input("Seleccione una opción: ").strip()

            if entrada == "1":
                while True:
                    entrada_n = input("¿Cuántas primeras descargas desea considerar? ").strip()
                    try:
                        n = int(entrada_n)
                        if n < 3:
                            print("Debe considerar al menos 3 descargas.")
                            continue
                        n_primeras = n
                        break
                    except ValueError:
                        print("Entrada inválida. Ingrese un entero.")
                break

            elif entrada == "2":
                print("Se continúa sin aceptar normativamente el a0q calculado.")
                return resultado

            else:
                print("Opción inválida.")

def preguntar_si_seguir(mensaje="\n¿Desea continuar con el análisis ASTM? [s/n]: "):
    while True:
        entrada = input(mensaje).strip().lower()
        if entrada in ["s", "si", "sí"]:
            return True
        if entrada in ["n", "no"]:
            return False
        print("Respuesta inválida. Escriba 's' o 'n'.")


def calcular_delta_a(resultados_seleccionados, a0q, W):
    """
    Calcula Δa ASTM respecto de a0q.

    Además:
    - ordena por número de descarga (no por x_descarga),
      para respetar la secuencia física;
    - mantiene una versión debug respecto del primer a_i activo;
    - NO fuerza monotonía de a_i.
    """
    activos = [
        r for r in resultados_seleccionados
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not np.isnan(r.get("a_i_mm", np.nan))
            and not np.isnan(r.get("x_descarga", np.nan))
        )
    ]

    if len(activos) == 0:
        for r in resultados_seleccionados:
            r["delta_a_mm"] = np.nan
            r["delta_a_debug_mm"] = np.nan
            r["delta_a_astm_mm"] = np.nan
            r["a_i_W"] = np.nan
        return

    # Orden físico correcto: por número de descarga
    activos = sorted(activos, key=lambda r: r["descarga"])

    ai_vals = np.array([r["a_i_mm"] for r in activos], dtype=float)

    print("\nChequeo de a_i y Δa:")

    a_ref = ai_vals[0]

    for r, ai_corr in zip(activos, ai_vals):
        r["a_i_mm"] = float(ai_corr)
        r["a_i_W"] = float(ai_corr / W)

        # Δa sólo para depuración visual, respecto del primer punto activo
        r["delta_a_debug_mm"] = float(ai_corr - a_ref)

        # Δa ASTM oficial del programa
        r["delta_a_astm_mm"] = float(ai_corr - a0q)
        r["delta_a_mm"] = float(r["delta_a_astm_mm"])

        print(
            f'Desc {r.get("numero", r.get("descarga"))}: '
            f'a_i = {r["a_i_mm"]:.6f} mm | '
            f'Δa_debug = {r["delta_a_debug_mm"]:.6f} mm | '
            f'Δa_astm = {r["delta_a_mm"]:.6f} mm'
        )
            
def factor_geometrico_ct(aW):
    """
    Factor geométrico f(a/W) para C(T).
    """
    if np.isnan(aW) or aW <= 0 or aW >= 1:
        return np.nan

    num = (2 + aW) * (
        0.886
        + 4.64 * aW
        - 13.32 * aW**2
        + 14.72 * aW**3
        - 5.6 * aW**4
    )
    den = (1 - aW) ** 1.5

    return num / den


def calcular_K_ct(P, B, BN, W, a):
    """
    Calcula K para probeta C(T).

    P  : carga [kN]
    B, BN, W, a : [mm]

    Devuelve K en MPa*sqrt(m), usando coherencia:
    - P en kN
    - dimensiones en mm
    """
    if np.isnan(P) or np.isnan(a) or W <= 0 or B <= 0 or BN <= 0:
        return np.nan

    aW = a / W
    f = factor_geometrico_ct(aW)
    if np.isnan(f):
        return np.nan

    # P [kN] -> N
    P_N = P * 1000.0

    # Fórmula en N y mm -> MPa*sqrt(mm), luego /sqrt(1000) -> MPa*sqrt(m)
    K_mpa_sqrt_mm = (P_N / np.sqrt(B * BN * W)) * f
    K_mpa_sqrt_m = K_mpa_sqrt_mm / np.sqrt(1000.0)

    return K_mpa_sqrt_m


def calcular_J_el(K, E, nu):
    """
    J_el = K^2 * (1 - nu^2) / E

    K en MPa*sqrt(m)
    E en MPa

    Devuelve J_el en kJ/m^2
    """
    if np.isnan(K):
        return np.nan

    return (K**2) * (1.0 - nu**2) / E * 1000.0


def calcular_eta_pl_ct(b, W):
    """
    eta_pl para C(T)
    """
    if np.isnan(b) or np.isnan(W) or W <= 0:
        return np.nan
    return 2.0 + 0.522 * (b / W)


def calcular_gamma_ct(b, W):
    """
    gamma para C(T)
    """
    if np.isnan(b) or np.isnan(W) or W <= 0:
        return np.nan
    return 1.0 + 0.76 * (b / W)


def integrar_area_hasta_punto(x, y, x_obj):
    """
    Integra el área bajo y(x) desde el inicio hasta x_obj usando trapecios.
    x debe ser creciente o al menos no decreciente en promedio.
    """
    if len(x) < 2 or len(y) < 2 or np.isnan(x_obj):
        return np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = x <= x_obj
    x_sub = x[mask]
    y_sub = y[mask]

    if len(x_sub) < 2:
        return np.nan

    return np.trapezoid(y_sub, x_sub)

def calcular_vll_desde_cmod_geometrico(cmod_i, a_i, W, d, H_ast):
    """
    Convierte CMOD medido a load-line displacement (V_LL) mediante
    una relación geométrica de rotación para probeta C(T).
    """
    if not np.isfinite(cmod_i) or not np.isfinite(a_i):
        return np.nan
    if W <= 0 or d <= 0 or H_ast <= 0:
        return np.nan
    if cmod_i < 0:
        return np.nan

    D = d / 2.0
    R_i = (W + a_i) / 2.0

    if not np.isfinite(R_i) or R_i <= 0:
        return np.nan

    denom = 2.0 * np.sqrt(D**2 + R_i**2)
    if denom <= 0:
        return np.nan

    arg = np.clip(cmod_i / denom, -1.0, 1.0)
    theta = np.arcsin(arg) - np.arctan(D / R_i)

    V_LL = 2.0 * np.sqrt(H_ast**2 + R_i**2) * np.sin(theta + np.arctan(H_ast / R_i))

    return V_LL if np.isfinite(V_LL) and V_LL >= 0 else np.nan


def calcular_v_elastico_desde_ll(carga_i, Ci_cmod, a_i, W, d, H_ast):
    """
    Calcula la componente elástica del desplazamiento, pero en base LLD.
    Parte de v_el_CMOD = P * C_CMOD y luego lo transforma a LLD.
    """
    if not np.isfinite(carga_i) or not np.isfinite(Ci_cmod):
        return np.nan
    if carga_i < 0 or Ci_cmod <= 0:
        return np.nan

    v_el_cmod = carga_i * Ci_cmod

    if not np.isfinite(v_el_cmod) or v_el_cmod < 0:
        return np.nan

    v_el_ll = calcular_vll_desde_cmod_geometrico(
        cmod_i=v_el_cmod,
        a_i=a_i,
        W=W,
        d=d,
        H_ast=H_ast
    )

    return v_el_ll


def calcular_J_integral_ct(
    resultados_seleccionados,
    cmod,
    carga,
    E,
    nu,
    W,
    B,
    BN,
    d,
    H_ast,
    ruta_guardado_jr_auto=None,
    mostrar=True,
    verbose=True
):
    """
    Calcula la integral J para probeta C(T), separando:
    - parte elástica: a partir de K
    - parte plástica: por incremento de área plástica

    IMPORTANTE:
    Esta función NO redefine el Δa ASTM oficial.
    Si existe r["delta_a_mm"] calculado previamente desde a0q,
    se preserva. Aquí sólo se calcula un Δa interno de depuración
    respecto del primer a_i activo.
    """
    resultados_validos = [
        r for r in resultados_seleccionados
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not np.isnan(r.get("a_i_mm", np.nan))
            and not np.isnan(r.get("Ci_final", np.nan))
            and not np.isnan(r.get("x_descarga", np.nan))
        )
    ]

    resultados_validos = sorted(resultados_validos, key=lambda r: r["descarga"])

    if len(resultados_validos) == 0:
        print("No hay puntos válidos para calcular J.")
        return [], []

    # Referencia interna SOLO para debug
    a_ref = resultados_validos[0]["a_i_mm"]

    J_vals = []
    delta_a_vals = []

    P_prev = None
    vpl_prev = None
    a_prev = None
    b_prev = None
    Jpl_prev = 0.0

    if verbose:
        print("\n" + "=" * 120)
        print("CÁLCULO DE J INTEGRAL PARA C(T)")
        print("=" * 120)
        print(
            "Desc |   CMOD   |   P_i    |  a_i   |  b_i   |  VLL   |  v_el  |  v_pl  |"
            "  dApl   |  J_el |  J_pl | J_total"
        )

    for r in resultados_validos:
        obs = []

        desc = r["descarga"]
        a_i = r.get("a_i_mm", np.nan)
        Ci_i = r.get("Ci_final", np.nan)

        # Tomar índices/valores guardados; si faltan, usar fallback
        idx_inicio = r.get("indice_inicio_descarga", None)
        cmod_i = r.get("x_descarga", np.nan)
        P_i = r.get("P_i_kN", np.nan)

        if idx_inicio is not None and 0 <= idx_inicio < len(cmod):
            cmod_i = float(cmod[idx_inicio])
        if idx_inicio is not None and 0 <= idx_inicio < len(carga):
            P_i = float(carga[idx_inicio])

        b_i = W - a_i

        # Limpiar campos por defecto
        r["K_i_MPa_sqrt_m"] = np.nan
        r["J_el_kJ_m2"] = np.nan
        r["J_pl_kJ_m2"] = np.nan
        r["J_total_kJ_m2"] = np.nan
        r["vLL_mm"] = np.nan
        r["vel_mm"] = np.nan
        r["vpl_mm"] = np.nan
        r["dApl_kN_mm"] = np.nan
        r["observacion_j"] = ""

        if (
            not np.isfinite(cmod_i)
            or not np.isfinite(P_i)
            or not np.isfinite(a_i)
            or not np.isfinite(Ci_i)
            or not np.isfinite(b_i)
            or b_i <= 0
        ):
            r["observacion_j"] = "dato faltante"
            continue

        # Δa interno sólo para debug local
        delta_a_debug_local = max(0.0, a_i - a_ref)

        # 1) Desplazamiento total en base load-line
        V_LL = calcular_vll_desde_cmod_geometrico(
            cmod_i=cmod_i,
            a_i=a_i,
            W=W,
            d=d,
            H_ast=H_ast
        )

        # 2) Parte elástica en la misma base load-line
        v_el_i = calcular_v_elastico_desde_ll(
            carga_i=P_i,
            Ci_cmod=Ci_i,
            a_i=a_i,
            W=W,
            d=d,
            H_ast=H_ast
        )

        # 3) Parte plástica
        if not np.isfinite(V_LL):
            obs.append("VLL inválido")
            v_pl_i = np.nan
        elif not np.isfinite(v_el_i):
            obs.append("vel inválido")
            v_pl_i = np.nan
        else:
            v_pl_i = V_LL - v_el_i
            if v_pl_i < 0:
                v_pl_i = 0.0
                obs.append("vpl<0 recortado")

        # 4) Parte elástica de J
        K_i = calcular_K_ct(P_i, B, BN, W, a_i)
        J_el = calcular_J_el(K_i, E, nu)

        if not np.isfinite(J_el) or J_el < 0:
            J_el = 0.0
            obs.append("Jel corregido")

        # 5) Parte plástica incremental
        if P_prev is None or vpl_prev is None or a_prev is None or b_prev is None:
            dApl = 0.0
            J_pl = 0.0
        else:
            if not np.isfinite(v_pl_i) or not np.isfinite(vpl_prev):
                dvpl = 0.0
                obs.append("dvpl inválido")
            else:
                dvpl = v_pl_i - vpl_prev
                if dvpl < 0:
                    dvpl = 0.0
                    obs.append("dvpl<0 recortado")

            dApl = 0.5 * (P_i + P_prev) * dvpl  # kN·mm

            da = a_i - a_prev
            if da < 0:
                da = 0.0
                obs.append("da<0 recortado")

            eta_prev = calcular_eta_pl_ct(b_prev, W)
            gamma_prev = calcular_gamma_ct(b_prev, W)

            if not np.isfinite(eta_prev):
                eta_prev = 0.0
                obs.append("eta inválido")

            if not np.isfinite(gamma_prev):
                gamma_prev = 0.0
                obs.append("gamma inválido")

            if not np.isfinite(b_prev) or b_prev <= 0:
                termino_correccion = 1.0
                incremento_j_pl = 0.0
                obs.append("b_prev inválido")
            else:
                termino_correccion = 1.0 - gamma_prev * (da / b_prev)
                termino_correccion = max(0.0, termino_correccion)

                incremento_j_pl = (eta_prev / (b_prev * BN)) * dApl * 1000.0
                J_pl = (Jpl_prev * termino_correccion) + incremento_j_pl

        if not np.isfinite(J_pl) or J_pl < 0:
            J_pl = 0.0
            obs.append("Jpl corregido")

        J_total = J_el + J_pl

        r["P_i_kN"] = P_i
        r["K_i_MPa_sqrt_m"] = K_i
        r["J_el_kJ_m2"] = J_el
        r["J_pl_kJ_m2"] = J_pl
        r["J_total_kJ_m2"] = J_total
        r["vLL_mm"] = V_LL
        r["vel_mm"] = v_el_i
        r["vpl_mm"] = v_pl_i
        r["dApl_kN_mm"] = dApl

        # Guardar sólo el debug local; NO pisar delta_a_mm ASTM
        r["delta_a_debug_local_mm"] = delta_a_debug_local
        r["observacion_j"] = "; ".join(obs)

        # Para la curva J-R usar el Δa ASTM si existe; si no, fallback debug
        delta_a_plot = r.get("delta_a_mm", np.nan)
        if not np.isfinite(delta_a_plot):
            delta_a_plot = delta_a_debug_local

        J_vals.append(J_total)
        delta_a_vals.append(delta_a_plot)

        if verbose:
            print(
                f"{desc:4d} | "
                f"{cmod_i:8.4f} | {P_i:8.3f} | {a_i:7.3f} | {b_i:7.3f} | "
                f"{V_LL:7.4f} | {v_el_i:7.4f} | {v_pl_i:7.4f} | {dApl:8.4f} | "
                f"{J_el:6.2f} | {J_pl:6.2f} | {J_total:7.2f}"
            )
            if obs:
                print(f"       obs: {r['observacion_j']}")

        P_prev = P_i
        vpl_prev = v_pl_i if np.isfinite(v_pl_i) else 0.0
        Jpl_prev = J_pl
        a_prev = a_i
        b_prev = b_i

    if len(J_vals) > 0 and len(delta_a_vals) > 0:
        graficar_J_vs_delta_a(
            resultados_seleccionados,
            "Curva J-R automática",
            ruta_guardado=ruta_guardado_jr_auto,
            mostrar=mostrar
        )

    return J_vals, delta_a_vals


def mostrar_tabla_J(resultados_seleccionados):
    filas = []

    for r in resultados_seleccionados:
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not np.isnan(r.get("J_total_kJ_m2", np.nan))
        ):
            filas.append({
                "descarga": r["descarga"],
                "x_descarga_mm": r["x_descarga"],
                "a_i_mm": r["a_i_mm"],
                "Δa_mm": r["delta_a_mm"],
                "P_i_kN": r.get("P_i_kN", np.nan),
                "K_i_MPa_sqrt_m": r.get("K_i_MPa_sqrt_m", np.nan),
                "J_el_kJ_m2": r.get("J_el_kJ_m2", np.nan),
                "J_pl_kJ_m2": r.get("J_pl_kJ_m2", np.nan),
                "J_total_kJ_m2": r.get("J_total_kJ_m2", np.nan),
            })

    if len(filas) == 0:
        print("No hay datos válidos para la tabla de J.")
        return

    tabla = pd.DataFrame(filas)

    print("\n" + "=" * 100)
    print("TABLA FINAL: CÁLCULO DE J")
    print("=" * 100)
    print(tabla.to_string(index=False))

def imprimir_diagnostico_ci_ai_j(resultados_seleccionados, a0_base):
    """
    Tabla de diagnóstico para rastrear inconsistencias entre:
    Ci original -> Ci corregida -> a_i -> Δa -> J

    Muestra también incrementos entre descargas consecutivas.
    """
    activos = [
        r for r in resultados_seleccionados
        if (
            r.get("seleccionada", False)
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
        )
    ]

    if len(activos) == 0:
        print("\nNo hay resultados activos para diagnóstico.")
        return

    activos = sorted(activos, key=lambda r: r["descarga"])

    print("\n" + "=" * 150)
    print("DIAGNÓSTICO: Ci -> Ci_final -> a_i -> Δa -> J")
    print("=" * 150)
    print(
        "Desc | x_desc [mm] | Ci_orig | Ci_final | dCi% | theta [deg] | "
        "a_i [mm] | da_prev [mm] | Δa=a_i-a0 [mm] | P [kN] | vpl [mm] | dApl [kN·mm] | "
        "Jel [kJ/m²] | Jpl [kJ/m²] | Jtot [kJ/m²]"
    )

    a_prev = np.nan

    for r in activos:
        ci_orig = r.get("Ci", np.nan)
        ci_fin = r.get("Ci_final", np.nan)

        if np.isfinite(ci_orig) and abs(ci_orig) > 1e-14 and np.isfinite(ci_fin):
            dci_pct = 100.0 * (ci_fin - ci_orig) / ci_orig
        else:
            dci_pct = np.nan

        a_i = r.get("a_i_mm", np.nan)

        if np.isfinite(a_i) and np.isfinite(a_prev):
            da_prev = a_i - a_prev
        else:
            da_prev = np.nan

        print(
            f'{r.get("descarga", np.nan):>4} | '
            f'{r.get("x_descarga", np.nan):>11.4f} | '
            f'{ci_orig:>7.6f} | '
            f'{ci_fin:>8.6f} | '
            f'{dci_pct:>6.1f} | '
            f'{r.get("theta_deg", np.nan):>11.3f} | '
            f'{a_i:>8.4f} | '
            f'{da_prev:>12.4f} | '
            f'{r.get("delta_a_mm", np.nan):>14.4f} | '
            f'{r.get("P_i_kN", np.nan):>6.3f} | '
            f'{r.get("vpl_mm", np.nan):>8.4f} | '
            f'{r.get("dApl_kN_mm", np.nan):>12.4f} | '
            f'{r.get("J_el_kJ_m2", np.nan):>11.3f} | '
            f'{r.get("J_pl_kJ_m2", np.nan):>11.3f} | '
            f'{r.get("J_total_kJ_m2", np.nan):>12.3f}'
        )

        if np.isfinite(a_i):
            a_prev = a_i

    print("-" * 150)
    print(f"a0 utilizado = {a0_base:.6f} mm")
    print("Referencias:")
    print(" - dCi%      = 100 * (Ci_final - Ci_orig) / Ci_orig")
    print(" - da_prev   = a_i(actual) - a_i(anterior)")
    print(" - Δa        = a_i - a0")
    print("Valores sospechosos:")
    print(" - dCi% muy grande en magnitud")
    print(" - da_prev < 0  (retroceso aparente de grieta)")
    print(" - Δa muy negativo con J ya alto")
    print("=" * 150)


def obtener_pares_jr_graficables(resultados_seleccionados):
    """
    Devuelve TODOS los puntos que se puedan dibujar en J-R,
    aunque no sean válidos ASTM.
    """
    pares = []

    for r in resultados_seleccionados:
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not r.get("eliminada_jr", False)
        ):
            delta_a = r.get("delta_a_astm_mm", np.nan)
            if not np.isfinite(delta_a):
                delta_a = r.get("delta_a_mm", np.nan)

            J = r.get("J_total_kJ_m2", np.nan)

            if np.isfinite(delta_a) and np.isfinite(J):
                pares.append({
                    "descarga": r["descarga"],
                    "delta_a_mm": float(delta_a),
                    "delta_a_astm_mm": float(delta_a),
                    "J_kJ_m2": float(J),
                    "valido_astm": False,   # se completa después
                })

    return pares


def obtener_pares_jr_validos(resultados_seleccionados):
    """
    Devuelve sólo los puntos utilizables para análisis ASTM.
    """
    pares = []

    for r in resultados_seleccionados:
        if (
            r["seleccionada"]
            and not r.get("eliminada", False)
            and not r.get("eliminada_corregida", False)
            and not r.get("eliminada_jr", False)
            and np.isfinite(r.get("delta_a_astm_mm", np.nan))
            and np.isfinite(r.get("J_total_kJ_m2", np.nan))
        ):
            pares.append({
                "descarga": r["descarga"],
                "delta_a_mm": float(r["delta_a_astm_mm"]),
                "delta_a_astm_mm": float(r["delta_a_astm_mm"]),
                "J_kJ_m2": float(r["J_total_kJ_m2"]),
                "valido_astm": False,
            })

    return pares


def enriquecer_puntos_astm(pares, sigma_y_MPa, b0_mm):
    """
    Enriquece cada punto J-R con criterios ASTM.

    Reglas aplicadas:
    - límite horizontal: J <= J_limit = b0 * sigma_y / 7.5
    - límite vertical:   Δa <= Δa_limit = 0.25 * b0
    - exclusión inferior: el punto debe estar a la derecha de 0.15 mm
      y por debajo de J = 2*sigma_y*(Δa - 0.15)
    - exclusión superior: si Δa > 1.50 mm, además debe estar por debajo
      de J = 2*sigma_y*(Δa - 1.50)

    Devuelve:
    - lista de dicts enriquecidos
    - J_limit
    - Δa_limit
    """
    j_limit = b0_mm * sigma_y_MPa / 7.5
    da_limit = 0.25 * b0_mm

    salida = []

    for p in pares:
        da = float(p["delta_a_mm"])
        J = float(p["J_kJ_m2"])

        J_bl = 2.0 * sigma_y_MPa * da
        J_excl_015 = 2.0 * sigma_y_MPa * (da - 0.15)
        J_excl_150 = 2.0 * sigma_y_MPa * (da - 1.50)

        # Exclusión inferior:
        # el punto debe estar a la derecha de 0.15 mm
        # y por debajo de la línea de exclusión de 0.15 mm
        cumple_da_min = (da >= 0.15)
        cumple_exclusion_inf = bool(cumple_da_min and (J <= J_excl_015))

        # Exclusión superior:
        # sólo empieza a tener efecto a la derecha de 1.50 mm
        if da <= 1.50:
            cumple_exclusion_sup = True
        else:
            cumple_exclusion_sup = bool(J <= J_excl_150)

        cumple_exclusion = bool(cumple_exclusion_inf and cumple_exclusion_sup)
        cumple_limites = bool((da <= da_limit) and (J <= j_limit))

        valido_astm = bool(cumple_exclusion and cumple_limites)

        q = p.copy()
        q["J_bl"] = J_bl
        q["J_excl_015"] = J_excl_015
        q["J_excl_150"] = J_excl_150

        q["cumple_da_min"] = bool(cumple_da_min)
        q["cumple_exclusion_inf"] = bool(cumple_exclusion_inf)
        q["cumple_exclusion_sup"] = bool(cumple_exclusion_sup)
        q["cumple_exclusion"] = bool(cumple_exclusion)
        q["cumple_limites"] = bool(cumple_limites)

        q["valido_astm"] = valido_astm
        q["valido_total"] = valido_astm

        salida.append(q)

    return salida, j_limit, da_limit

def ajustar_potencia_jr(delta_a, J, k_mm=1.0):
    delta_a = np.asarray(delta_a, dtype=float)
    J = np.asarray(J, dtype=float)

    mask = (
        np.isfinite(delta_a) &
        np.isfinite(J) &
        (delta_a > 0) &
        (J > 0)
    )

    x = delta_a[mask] / k_mm
    y = J[mask]

    print("\n[DEBUG ajustar_potencia_jr]")
    print("delta_a entrada:", delta_a)
    print("J entrada:", J)
    print("x usados:", x)
    print("y usados:", y)
    print("cantidad usados:", len(x))

    if len(x) < 2:
        print("ERROR: menos de 2 puntos válidos para ajustar.")
        return np.nan, np.nan, np.nan

    if np.allclose(x, x[0]):
        print("ERROR: todos los Δa son iguales.")
        return np.nan, np.nan, np.nan

    logx = np.log(x)
    logy = np.log(y)

    if not np.all(np.isfinite(logx)) or not np.all(np.isfinite(logy)):
        print("ERROR: logx o logy contienen NaN/inf.")
        return np.nan, np.nan, np.nan

    try:
        m, b = np.polyfit(logx, logy, 1)
    except Exception as e:
        print(f"ERROR en ajuste log-log: {e}")
        return np.nan, np.nan, np.nan

    c2 = float(m)
    c1 = float(np.exp(b))

    y_pred = c1 * (x ** c2)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot <= 0:
        r2 = np.nan
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    print("c1 =", c1)
    print("c2 =", c2)
    print("r2 =", r2)

    return c1, c2, r2


def evaluar_potencia_jr(delta_a, c1, c2, k_mm=1.0):
    delta_a = np.asarray(delta_a, dtype=float)

    if not np.isfinite(c1) or not np.isfinite(c2):
        return np.full_like(delta_a, np.nan, dtype=float)

    y = np.full_like(delta_a, np.nan, dtype=float)
    mask = np.isfinite(delta_a) & (delta_a > 0)
    y[mask] = c1 * (delta_a[mask] / k_mm) ** c2
    return y


def graficar_astm_etapa_1_exclusion(pares_astm, sigma_y_MPa, nombre_archivo,
                                    ruta_guardado=None, mostrar=True):
    fig, ax = plt.subplots(figsize=(9, 6))

    xv = [p["delta_a_astm_mm"] for p in pares_astm]
    yv = [p["J_kJ_m2"] for p in pares_astm]
    dv = [p["descarga"] for p in pares_astm]

    ax.scatter(xv, yv, s=45, label="Puntos experimentales")

    for x, y, d in zip(xv, yv, dv):
        ax.text(x, y, str(d), fontsize=9, ha="center", va="bottom")

    da_max = max(max(xv) if len(xv) else 1.5, 1.6) * 1.10
    da_grid = np.linspace(0.0, da_max, 400)

    bl = 4.0 * sigma_y_MPa * da_grid
    excl_015 = 2.0 * sigma_y_MPa * (da_grid - 0.15)
    excl_150 = 2.0 * sigma_y_MPa * (da_grid - 1.50)

    ax.plot(da_grid, bl, "--", linewidth=1.5, label="Blunting line")
    ax.plot(da_grid, excl_015, "--", linewidth=1.2, label="Exclusión 0.15 mm")
    ax.plot(da_grid, excl_150, "--", linewidth=1.2, label="Exclusión 1.50 mm")

    ax.set_xlabel("Δa [mm]")
    ax.set_ylabel("J [kJ/m²]")
    ax.set_title(f"ASTM - Etapa 1: líneas de construcción y exclusión\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    guardar_o_mostrar_figura(fig, ruta_guardado=ruta_guardado, mostrar=mostrar)


def graficar_astm_etapa_2_limites(pares_astm, sigma_y_MPa, j_limit, da_limit, nombre_archivo,
                                  ruta_guardado=None, mostrar=True):
    fig, ax = plt.subplots(figsize=(9, 6))

    xv = [p["delta_a_astm_mm"] for p in pares_astm]
    yv = [p["J_kJ_m2"] for p in pares_astm]
    dv = [p["descarga"] for p in pares_astm]

    ax.scatter(xv, yv, s=45, label="Puntos experimentales")

    for x, y, d in zip(xv, yv, dv):
        ax.text(x, y, str(d), fontsize=9, ha="center", va="bottom")

    da_max = max(max(xv) if len(xv) else 1.5, da_limit, 1.6) * 1.10
    da_grid = np.linspace(0.0, da_max, 400)

    bl = 2.0 * sigma_y_MPa * da_grid
    excl_015 = 2.0 * sigma_y_MPa * (da_grid - 0.15)
    excl_150 = 2.0 * sigma_y_MPa * (da_grid - 1.50)

    ax.plot(da_grid, bl, "--", linewidth=1.5, label="Blunting line")
    ax.plot(da_grid, excl_015, "--", linewidth=1.2, label="Exclusión 0.15 mm")
    ax.plot(da_grid, excl_150, "--", linewidth=1.2, label="Exclusión 1.50 mm")

    ax.axhline(j_limit, linestyle=":", linewidth=1.5, label=f"J_limit = {j_limit:.2f}")
    ax.axvline(da_limit, linestyle=":", linewidth=1.5, label=f"Δa_limit = {da_limit:.2f}")

    ax.set_xlabel("Δa [mm]")
    ax.set_ylabel("J [kJ/m²]")
    ax.set_title(f"ASTM - Etapa 2: exclusión + límites máximos\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()

    guardar_o_mostrar_figura(fig, ruta_guardado=ruta_guardado, mostrar=mostrar)


def graficar_astm_etapa_3_ajuste(pares_astm, sigma_y_MPa, j_limit, da_limit,
                                 c1, c2, r2, nombre_archivo,
                                 ruta_guardado=None, mostrar=True):
    fig, ax = plt.subplots(figsize=(9, 6))

    usados_ajuste = [p for p in pares_astm if p.get("usado_en_ajuste", False)]
    validos_no_usados = [
        p for p in pares_astm
        if p.get("valido_total", False) and not p.get("usado_en_ajuste", False)
    ]
    excluidos = [p for p in pares_astm if not p.get("valido_total", False)]

    def extraer_xy(lista):
        x = np.array([p.get("delta_a_astm_mm", np.nan) for p in lista], dtype=float)
        y = np.array([p.get("J_kJ_m2", np.nan) for p in lista], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    x_usados, y_usados = extraer_xy(usados_ajuste)
    x_validos_no_usados, y_validos_no_usados = extraer_xy(validos_no_usados)
    x_excluidos, y_excluidos = extraer_xy(excluidos)

    if len(x_usados) > 0:
        ax.scatter(
            x_usados,
            y_usados,
            s=55,
            label="Puntos usados en ajuste",
            zorder=4
        )

    if len(x_validos_no_usados) > 0:
        ax.scatter(
            x_validos_no_usados,
            y_validos_no_usados,
            s=50,
            marker="o",
            label="Válidos ASTM no usados",
            zorder=3
        )

    if len(x_excluidos) > 0:
        ax.scatter(
            x_excluidos,
            y_excluidos,
            s=45,
            marker="x",
            label="Puntos excluidos",
            zorder=2
        )

    for p in pares_astm:
        x_txt = p.get("delta_a_astm_mm", np.nan)
        y_txt = p.get("J_kJ_m2", np.nan)
        if np.isfinite(x_txt) and np.isfinite(y_txt):
            ax.text(
                x_txt, y_txt, str(p["descarga"]),
                fontsize=9, ha="center", va="bottom"
            )

    todos_x = np.array([p.get("delta_a_astm_mm", np.nan) for p in pares_astm], dtype=float)
    todos_x = todos_x[np.isfinite(todos_x)]

    refs_x = []
    if len(todos_x) > 0:
        refs_x.append(np.max(todos_x))
    if np.isfinite(da_limit):
        refs_x.append(da_limit)
    da_max = max(refs_x) * 1.10 if len(refs_x) > 0 else 1.6
    da_max = max(da_max, 1.6)

    da_grid = np.linspace(1e-6, da_max, 500)

    bl = 2.0 * sigma_y_MPa * da_grid
    excl_015 = 2.0 * sigma_y_MPa * (da_grid - 0.15)
    excl_150 = 2.0 * sigma_y_MPa * (da_grid - 1.50)

    ax.plot(da_grid, bl, "--", linewidth=1.2, label="Blunting line")
    ax.plot(da_grid, excl_015, "--", linewidth=1.0, label="Exclusión 0.15 mm")
    ax.plot(da_grid, excl_150, "--", linewidth=1.0, label="Exclusión 1.50 mm")

    if np.isfinite(j_limit):
        ax.axhline(j_limit, linestyle=":", linewidth=1.2, label="J_limit")
    if np.isfinite(da_limit):
        ax.axvline(da_limit, linestyle=":", linewidth=1.2, label="Δa_limit")

    if np.isfinite(c1) and np.isfinite(c2):
        if len(x_usados) >= 2:
            xmin = np.min(x_usados[x_usados > 0]) if np.any(x_usados > 0) else np.nan
            xmax = np.max(x_usados) if len(x_usados) > 0 else np.nan

            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                da_fit = np.linspace(xmin, xmax, 300)
                jr_fit = evaluar_potencia_jr(da_fit, c1, c2, k_mm=1.0)
                ax.plot(
                    da_fit, jr_fit,
                    linewidth=2.0,
                    color="red",
                    label=f"Ajuste potencia: J = {c1:.3f}(Δa)^{c2:.3f} | R²={r2:.4f}"
                )

    todos_y = np.array([p.get("J_kJ_m2", np.nan) for p in pares_astm], dtype=float)
    todos_y = todos_y[np.isfinite(todos_y)]

    y_refs = []
    if len(todos_y) > 0:
        y_refs.append(np.max(todos_y))
    if np.isfinite(j_limit):
        y_refs.append(j_limit)

    if len(y_refs) > 0:
        y_max = max(y_refs) * 1.15
        y_min = min(0.0, np.min(todos_y) * 1.10) if len(todos_y) > 0 else 0.0
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Δa [mm]")
    ax.set_ylabel("J [kJ/m²]")
    ax.set_title(f"ASTM - Etapa 3: ajuste por ley de potencia\n{nombre_archivo}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    guardar_o_mostrar_figura(fig, ruta_guardado=ruta_guardado, mostrar=mostrar)

def analisis_astm_post_jr_interactivo(resultados_seleccionados, nombre_archivo, W, a0, sigma_y_MPa,
                                      ruta_guardado_base=None, mostrar=True):
    b0 = W - a0
    if b0 <= 0:
        print("Error: b0 <= 0. Revise W y a0.")
        return None

    pares = obtener_pares_jr_validos(resultados_seleccionados)
    if len(pares) < 2:
        print("No hay suficientes puntos J-R para análisis ASTM.")
        return None

    pares_astm, j_limit, da_limit = enriquecer_puntos_astm(pares, sigma_y_MPa, b0)

    ruta1 = None
    ruta2 = None
    ruta3 = None
    if ruta_guardado_base is not None:
        base, ext = os.path.splitext(ruta_guardado_base)
        if ext == "":
            ext = ".png"
        ruta1 = f"{base}_etapa1_exclusion{ext}"
        ruta2 = f"{base}_etapa2_limites{ext}"
        ruta3 = f"{base}_etapa3_ajuste{ext}"

    print("\n" + "=" * 100)
    print("ANÁLISIS ASTM POSTERIOR A LA CURVA J-R")
    print("=" * 100)
    print(f"b0 = W - a0 = {b0:.4f} mm")
    print(f"σy = {sigma_y_MPa:.4f} MPa")
    print(f"J_limit = {j_limit:.4f} kJ/m²")
    print(f"Δa_limit = {da_limit:.4f} mm")

    # ETAPA 1
    graficar_astm_etapa_1_exclusion(
        pares_astm=pares_astm,
        sigma_y_MPa=sigma_y_MPa,
        nombre_archivo=nombre_archivo,
        ruta_guardado=ruta1,
        mostrar=mostrar
    )

    if not preguntar_si_seguir("\n¿Desea seguir al gráfico con límites máximos? [s/n]: "):
        print("Análisis ASTM interrumpido por el usuario.")
        return {
            "etapa": 1,
            "pares_astm": pares_astm,
            "J_limit": j_limit,
            "da_limit": da_limit
        }

    # ETAPA 2
    graficar_astm_etapa_2_limites(
        pares_astm=pares_astm,
        sigma_y_MPa=sigma_y_MPa,
        j_limit=j_limit,
        da_limit=da_limit,
        nombre_archivo=nombre_archivo,
        ruta_guardado=ruta2,
        mostrar=mostrar
    )

    if not preguntar_si_seguir("\n¿Desea seguir al ajuste por ley de potencia? [s/n]: "):
        print("Análisis ASTM interrumpido por el usuario.")
        return {
            "etapa": 2,
            "pares_astm": pares_astm,
            "J_limit": j_limit,
            "da_limit": da_limit
        }

  # ETAPA 3
    validos = [p for p in pares_astm if p["valido_total"]]
    
    print("\nResumen ASTM:")
    for p in pares_astm:
        print(
            f'Desc {p["descarga"]} | '
            f'Δa={p["delta_a_astm_mm"]:.4f} | '
            f'J={p["J_kJ_m2"]:.2f} | '
            f'excl={p["cumple_exclusion"]} | '
            f'lim={p["cumple_limites"]} | '
            f'total={p["valido_total"]}'
        )
    
    print("Descargas válidas:", [p["descarga"] for p in pares_astm if p["valido_total"]])
    print("Cantidad válidas:", len(validos))
    
    if len(validos) < 2:
        print("No hay suficientes puntos válidos para ajustar ley de potencia.")
        return {
            "etapa": 3,
            "pares_astm": pares_astm,
            "J_limit": j_limit,
            "da_limit": da_limit,
            "c1": np.nan,
            "c2": np.nan,
            "r2": np.nan
        }
    
    # Marcar todo como no usado en ajuste
    for p in pares_astm:
        p["usado_en_ajuste"] = False
    
    # ------------------------------------------------------------------
    # VISTA PREVIA ANTES DEL AJUSTE: permite eliminar puntos a mano
    # ------------------------------------------------------------------
    graficar_astm_etapa_3_ajuste(
        pares_astm=pares_astm,
        sigma_y_MPa=sigma_y_MPa,
        j_limit=j_limit,
        da_limit=da_limit,
        c1=np.nan,
        c2=np.nan,
        r2=np.nan,
        nombre_archivo=nombre_archivo,
        ruta_guardado=None,
        mostrar=True
    )
    
    resp = input("¿Querés eliminar algún punto manualmente antes del ajuste? [s/n]: ").strip().lower()
    
    if resp in ["s", "si", "sí", "y", "yes"]:
        print("\nPuntos disponibles:")
        for p in pares_astm:
            print(
                f'Descarga {p["descarga"]}: '
                f'Δa={p["delta_a_astm_mm"]:.4f} mm | '
                f'J={p["J_kJ_m2"]:.2f} kJ/m² | '
                f'válido={p["valido_total"]}'
            )
    
        txt = input("\nIngresá los números de descarga a eliminar separados por coma: ").strip()
    
        a_eliminar = set()
        if txt:
            for parte in txt.split(","):
                parte = parte.strip()
                if parte.isdigit():
                    a_eliminar.add(int(parte))
    
        for p in pares_astm:
            if p["descarga"] in a_eliminar:
                p["valido_total"] = False
                p["motivo"] = "eliminado manualmente"
    
    validos = [p for p in pares_astm if p["valido_total"]]
    
    if len(validos) < 2:
        print("No hay suficientes puntos válidos para ajustar ley de potencia después de la eliminación manual.")
        return {
            "etapa": 3,
            "pares_astm": pares_astm,
            "J_limit": j_limit,
            "da_limit": da_limit,
            "c1": np.nan,
            "c2": np.nan,
            "r2": np.nan
        }
    
    validos_ajuste = [
        p for p in validos
        if np.isfinite(p["delta_a_astm_mm"])
        and np.isfinite(p["J_kJ_m2"])
        and p["delta_a_astm_mm"] > 0
        and p["J_kJ_m2"] > 0
    ]
    
    # Marcar cuáles sí entran al ajuste
    for p in validos_ajuste:
        p["usado_en_ajuste"] = True
    
    if len(validos_ajuste) < 2:
        print("No hay suficientes puntos positivos para ajustar ley de potencia.")
        return {
            "etapa": 3,
            "pares_astm": pares_astm,
            "J_limit": j_limit,
            "da_limit": da_limit,
            "c1": np.nan,
            "c2": np.nan,
            "r2": np.nan
        }
    
    delta_a_valid = np.array([p["delta_a_astm_mm"] for p in validos_ajuste], dtype=float)
    J_valid = np.array([p["J_kJ_m2"] for p in validos_ajuste], dtype=float)
    
    print("\n--- PUNTOS USADOS PARA AJUSTE ---")
    for p in validos_ajuste:
        print(
            f'Desc {p["descarga"]} | '
            f'Δa={p["delta_a_astm_mm"]:.6f} | '
            f'J={p["J_kJ_m2"]:.6f}'
        )
    
    print("delta_a_valid =", delta_a_valid)
    print("J_valid =", J_valid)
    
    c1, c2, r2 = ajustar_potencia_jr(delta_a_valid, J_valid, k_mm=1.0)
    
    if not np.isfinite(c1) or not np.isfinite(c2):
        print("El ajuste por ley de potencia no pudo realizarse correctamente.")
        return {
            "etapa": 3,
            "pares_astm": pares_astm,
            "J_limit": j_limit,
            "da_limit": da_limit,
            "c1": np.nan,
            "c2": np.nan,
            "r2": np.nan
        }
    
    graficar_astm_etapa_3_ajuste(
        pares_astm=pares_astm,
        sigma_y_MPa=sigma_y_MPa,
        j_limit=j_limit,
        da_limit=da_limit,
        c1=c1,
        c2=c2,
        r2=r2,
        nombre_archivo=nombre_archivo,
        ruta_guardado=ruta3,
        mostrar=mostrar
    )
    
    print("\n" + "=" * 100)
    print("RESULTADOS DEL AJUSTE ASTM")
    print("=" * 100)
    print(f"Puntos totales J-R: {len(pares_astm)}")
    print(f"Puntos válidos ASTM: {len(validos)}")
    print(f"Puntos usados en ajuste: {len(validos_ajuste)}")
    print(f"Ajuste: J = {c1:.6f} * (Δa)^({c2:.6f})")
    print(f"R² = {r2:.6f}")
    
    return {
        "etapa": 3,
        "pares_astm": pares_astm,
        "J_limit": j_limit,
        "da_limit": da_limit,
        "c1": c1,
        "c2": c2,
        "r2": r2
    }


def loop_eliminacion_interactiva_jr(resultados_seleccionados, nombre_archivo):
    """
    Revisión manual de la curva J-R.
    Muestra todos los puntos graficables, aunque luego no sean válidos ASTM.
    """
    while True:
        activos = []

        for r in resultados_seleccionados:
            if (
                r["seleccionada"]
                and not r.get("eliminada", False)
                and not r.get("eliminada_corregida", False)
                and not r.get("eliminada_jr", False)
            ):
                delta_a = r.get("delta_a_mm", np.nan)
                if not np.isfinite(delta_a):
                    delta_a = r.get("delta_a_debug_mm", np.nan)

                J = r.get("J_total_kJ_m2", np.nan)

                if np.isfinite(delta_a) and np.isfinite(J):
                    activos.append({
                        "descarga": r["descarga"],
                        "delta_a": float(delta_a),
                        "J": float(J),
                    })

        if len(activos) == 0:
            print("\nNo hay puntos graficables para la curva J-R.")
            return

        activos = sorted(activos, key=lambda p: p["delta_a"])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            [p["delta_a"] for p in activos],
            [p["J"] for p in activos],
            s=45
        )

        for p in activos:
            ax.text(p["delta_a"], p["J"], str(p["descarga"]), fontsize=9, ha="center", va="bottom")

        ax.set_xlabel("Δa [mm]")
        ax.set_ylabel("J [kJ/m²]")
        ax.set_title(f"Curva J-R para revisión manual\n{nombre_archivo}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

        print("\nPuntos actualmente mostrados en la curva J-R:")
        print(", ".join(str(p["descarga"]) for p in activos))
        print("[0] Terminar eliminación y mostrar gráfico final")

        entrada = input("Ingrese el número de descarga que desea eliminar de la curva J-R: ").strip()

        try:
            opcion = int(entrada)
        except ValueError:
            print("Entrada inválida. Ingrese un número.")
            continue

        if opcion == 0:
            graficar_J_vs_delta_a(
                resultados_seleccionados,
                nombre_archivo,
                mostrar=True
            )
            return

        encontrados = [
            r for r in resultados_seleccionados
            if (
                r["descarga"] == opcion
                and r["seleccionada"]
                and not r.get("eliminada", False)
                and not r.get("eliminada_corregida", False)
                and not r.get("eliminada_jr", False)
            )
        ]

        if len(encontrados) == 0:
            print(f"No existe un punto activo con número de descarga {opcion} en la curva J-R.")
            continue

        encontrados[0]["eliminada_jr"] = True
        print(f"Se eliminó de la curva J-R el punto correspondiente a la descarga {opcion}.")

def resolver_a0_si_no_cumple(resultado_a0q):
    """
    Si a0q no cumple criterio normativo, permite:
    [1] ingresar un a0 manual
    [2] continuar con el a0q promedio no normativo
    [0] cancelar cálculo de Δa y J

    Devuelve:
        {"continuar": True/False, "a0": valor_o_nan, "fuente": "..."}
    """
    while True:
        print("\n" + "=" * 100)
        print("a0q NO CUMPLE CRITERIO NORMATIVO")
        print("=" * 100)

        if not np.isnan(resultado_a0q.get("a0q", np.nan)):
            print(f"a0q promedio calculado = {resultado_a0q['a0q']:.6f} mm")
        else:
            print("a0q promedio calculado = NaN")

        print("\nOpciones:")
        print("[1] Ingresar un a0 manual")
        print("[2] Continuar con el a0q promedio NO normativo")
        print("[0] Cancelar cálculo de Δa y J")

        entrada = input("Seleccione una opción: ").strip()

        if entrada == "0":
            return {
                "continuar": False,
                "a0": np.nan,
                "fuente": "cancelado"
            }

        elif entrada == "1":
            while True:
                txt = input("Ingrese a0 manual [mm]: ").strip()
                try:
                    a0_manual = float(txt)
                    if a0_manual <= 0:
                        print("a0 debe ser mayor que cero.")
                        continue

                    return {
                        "continuar": True,
                        "a0": a0_manual,
                        "fuente": "manual"
                    }

                except ValueError:
                    print("Entrada inválida. Ingrese un número.")

        elif entrada == "2":
            a0q = resultado_a0q.get("a0q", np.nan)

            if np.isnan(a0q):
                print("No se puede continuar porque a0q es NaN.")
                continue

            return {
                "continuar": True,
                "a0": a0q,
                "fuente": "a0q_no_normativo"
            }

        else:
            print("Opción inválida.")

# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================
def main():
    print("=" * 72)
    print("CETERO-S:  Herramienta semiautomática para análisis de compliance por descarga parcial y construcción de curvas J-R en probetas C(T) / mini-C(T).")
    print("=" * 72)
    print("Modo semi-automático supervisado:")
    print(" - Detecta descargas")
    print(" - Muestra cada descarga por separado")
    print(" - Genera hasta 8 opciones de compliance")
    print(" - Muestra una tabla superpuesta con compliances ya elegidas")
    print(" - El usuario elige una opción para cada descarga")
    print(" - Grafica Compliance vs CMOD")
    print(" - Permite eliminar puntos manualmente")
    print(" - Luego permite corrección por rotación")
    print(" - Luego permite calcular a0q, Δa y J")
    print(" - Permite revisar manualmente la curva J-R")
    print(" - Análisis posterior basado en ASTM E1820")
    print(" - Generación de una carpeta de reporte con gráficos principales\n")

    ruta = seleccionar_archivo()

    if not ruta:
        print("No se seleccionó ningún archivo.")
        return

    print(f"Archivo seleccionado: {ruta}")

    carpeta_figuras = crear_carpeta_reporte(ruta)
    figuras_reporte = {}

    try:
        df = leer_csv_maquina(ruta)
        df = convertir_a_numerico(df)

        print("\nColumnas detectadas:")
        for c in df.columns:
            print(f" - {c}")

        col_cabezal, col_carga, col_cmod = elegir_columnas(df)

        # -------------------------------------------------
        # Curva carga vs desplazamiento del cabezal
        # -------------------------------------------------
        cabezal, carga_cabezal = preparar_datos(df, col_cabezal, col_carga)
        
        figuras_reporte["carga_vs_cabezal"] = os.path.join(carpeta_figuras, "01_carga_vs_cabezal.png")

        graficar_curva(
            cabezal,
            carga_cabezal,
            nombre_archivo=os.path.basename(ruta),
            etiqueta_x="Desplazamiento del cabezal [mm]",
            etiqueta_y="Carga [kN]",
            titulo="Curva carga vs desplazamiento del cabezal",
            ruta_guardado=figuras_reporte["carga_vs_cabezal"],
            mostrar=True
        )

        # -------------------------------------------------
        # Curva carga vs CMOD
        # -------------------------------------------------
        cmod, carga = preparar_datos(df, col_cmod, col_carga)
        
        figuras_reporte["carga_vs_cmod"] = os.path.join(carpeta_figuras, "02_carga_vs_cmod.png")

        graficar_curva(
            cmod,
            carga,
            nombre_archivo=os.path.basename(ruta),
            etiqueta_x="CMOD [mm]",
            etiqueta_y="Carga [kN]",
            titulo="Curva carga vs CMOD",
            ruta_guardado=figuras_reporte["carga_vs_cmod"],
            mostrar=True
        )

        # -------------------------------------------------
        # Detección de descargas
        # -------------------------------------------------
        segmentos = detectar_descargas(
            carga,
            min_puntos=CONFIG["min_puntos_descarga"],
            umbral_derivada=CONFIG["umbral_derivada_descarga"],
            caida_minima=CONFIG["caida_minima_carga"],
            corr_min=CONFIG["corr_min_descarga"],
            ventana_suavizado=CONFIG["ventana_suavizado"]
        )

        print(f"\nDescargas detectadas: {len(segmentos)}")

        if len(segmentos) == 0:
            print("No se detectaron ciclos de descarga con los criterios actuales.")
            return

        resultados_seleccionados = []

        # =========================================================
        # Selección descarga por descarga
        # =========================================================
        resultados_seleccionados = []
        filas_tabla = []
        opciones_por_descarga = {}
        datos_base_descargas = {}
        
        # -------------------------------------------------
        # 1) Generar tabla de opciones para TODAS las descargas
        # -------------------------------------------------
        for i, (ini, fin) in enumerate(segmentos, start=1):
            carga_seg = carga[ini:fin]
            cmod_seg = cmod[ini:fin]
        
            indice_inicio_descarga = int(ini)
            v_inicio_descarga = float(cmod[ini]) if 0 <= ini < len(cmod) else np.nan
            P_i_kN = float(carga[ini]) if 0 <= ini < len(carga) else np.nan
        
            candidatos_todos = generar_candidatos_compliance(
                carga_seg,
                cmod_seg,
                numero_descarga=i,
                n_submuestras=400,
                n_puntos_sub=18,
                seed=1234 + i
            )
        
            candidatos_6 = seleccionar_6_extremos(candidatos_todos, n_extremos=3)
        
            fila = resumir_candidatos_descarga(i, candidatos_6, n_cols=6)
            filas_tabla.append(fila)
        
            opciones_por_descarga[i] = mapear_opciones_descarga(candidatos_6, n_cols=6)
        
            datos_base_descargas[i] = {
                "indice_inicio_descarga": indice_inicio_descarga,
                "v_inicio_descarga": v_inicio_descarga,
                "P_i_kN": P_i_kN,
                "carga_seg": carga_seg,
                "cmod_seg": cmod_seg,
            }
        
        
        # -------------------------------------------------
        # 2) Mostrar tabla resumen
        # -------------------------------------------------
        tabla_opciones = pd.DataFrame(filas_tabla)
        
        print("\n" + "=" * 100)
        print("TABLA DE OPCIONES DE COMPLIANCE POR DESCARGA")
        print("=" * 100)
        print(tabla_opciones.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        
        print("\nLeyenda:")
        print("C1-C3 -> compliance más bajas")
        print("C4-C6 -> compliance más altas")
        
        # -------------------------------------------------
        # 3) Elegir una opción por descarga
        # -------------------------------------------------
        for i in range(1, len(segmentos) + 1):
            opciones = opciones_por_descarga[i]
            base = datos_base_descargas[i]
        
            print("\n" + "-" * 90)
            print(f"DESCARGA {i}")
            print("-" * 90)
        
            # lista de selecciones previas válidas
            seleccionadas_previas = [
                r for r in resultados_seleccionados
                if r.get("seleccionada", False)
            ]
        
            # >>> MOSTRAR EL GRÁFICO ANTES DE PEDIR LA ELECCIÓN <<<
            graficar_descarga_con_opciones(
                carga_seg=base["carga_seg"],
                cmod_seg=base["cmod_seg"],
                opciones_dict=opciones,
                numero_descarga=i,
                nombre_archivo=os.path.basename(ruta),
                seleccionadas_previas=seleccionadas_previas
            )
        
            for k in range(1, 7):
                clave = f"C{k}"
                cand = opciones[clave]
        
                if cand is None:
                    print(f"{clave}: ---")
                else:
                    print(
                        f"{clave}: "
                        f"Ci={cand['Ci']:.6f} mm/kN | "
                        f"R²={cand['R2']:.4f} | "
                        f"Ventana={cand['ventana']} | "
                        f"Tipo={cand['tipo']} | "
                        f"n={cand['n_puntos']}"
                    )
        
            eleccion_txt = input("Elegí una opción (C1-C6) o ENTER para omitir: ").strip().upper()
        
            seleccion = opciones.get(eleccion_txt, None)
        
            if seleccion is None:
                resultados_seleccionados.append({
                    "descarga": i,
                    "cmod_medio": np.nan,
                    "x_descarga": np.nan,
                    "v_inicio_descarga": np.nan,
                    "Ci": np.nan,
                    "Ci_final": np.nan,
                    "a_i_mm": np.nan,
                    "a_i_W": np.nan,
                    "R_i_mm": np.nan,
                    "theta_deg": np.nan,
                    "u": np.nan,
                    "R2": np.nan,
                    "ventana": None,
                    "intercepto": np.nan,
                    "seleccionada": False,
                    "eliminada": False,
                    "eliminada_corregida": False,
                    "eliminada_jr": False,
                    "motivo": "descarga omitida por usuario/default",
                    "opcion_elegida": "",
                    "indice_inicio_descarga": None,
                    "P_i_kN": np.nan,
                })
            else:
                resultados_seleccionados.append({
                    "descarga": i,
                    "cmod_medio": seleccion["cmod_medio"],
                    "x_descarga": base["v_inicio_descarga"],
                    "v_inicio_descarga": base["v_inicio_descarga"],
                    "indice_inicio_descarga": base["indice_inicio_descarga"],
                    "P_i_kN": base["P_i_kN"],
                    "Ci": seleccion["Ci"],
                    "Ci_final": np.nan,
                    "a_i_mm": np.nan,
                    "a_i_W": np.nan,
                    "R_i_mm": np.nan,
                    "theta_deg": np.nan,
                    "u": np.nan,
                    "R2": seleccion["R2"],
                    "ventana": seleccion["ventana"],
                    "intercepto": seleccion["intercepto"],
                    "seleccionada": True,
                    "eliminada": False,
                    "eliminada_corregida": False,
                    "eliminada_jr": False,
                    "motivo": f"seleccionada {eleccion_txt}",
                    "opcion_elegida": eleccion_txt,
                })
                

                
        # -------------------------------------------------
        # 4) Mostrar resumen de selección
        # -------------------------------------------------
        print("\n" + "=" * 100)
        print("RESUMEN DE SELECCIONES")
        print("=" * 100)
        for r in resultados_seleccionados:
            if r["seleccionada"]:
                print(
                    f"Descarga {r['descarga']:>2} | "
                    f"Ci = {r['Ci']:.6f} mm/kN | "
                    f"R² = {r['R2']:.4f} | "
                    f"Ventana = {r['ventana']}"
                )
            else:
                print(f"Descarga {r['descarga']:>2} | OMITIDA")


        # -------------------------------------------------
        # Tabla resumen inicial
        # -------------------------------------------------
        tabla = pd.DataFrame(resultados_seleccionados)

        print("\n" + "=" * 100)
        print("TABLA RESUMEN FINAL (ANTES DE CORRECCIÓN POR ROTACIÓN)")
        print("=" * 100)
        print(
            tabla[
                [
                    "descarga",
                    "x_descarga",
                    "v_inicio_descarga",
                    "Ci",
                    "R2",
                    "seleccionada",
                    "eliminada",
                    "motivo",
                ]
            ].to_string(index=False)
        )

        # -------------------------------------------------
        # Revisión manual de la curva original
        # -------------------------------------------------
        print("\nSe abrirá la curva Compliance vs CMOD para revisión manual.")
        print("Cada punto tendrá arriba el número de descarga correspondiente.")
        print("Ingrese el número que quiera eliminar de la gráfica.")
        print("Cuando termine, ingrese 0 para mostrar el gráfico final.\n")

        loop_eliminacion_interactiva(
            resultados_seleccionados,
            os.path.basename(ruta)
        )
        
        activos_finales = [
            r for r in resultados_seleccionados
            if r["seleccionada"] and not r.get("eliminada", False)
            and not np.isnan(r["Ci"]) and not np.isnan(r["cmod_medio"])
        ]

        figuras_reporte["compliance_vs_cmod_final"] = os.path.join(carpeta_figuras, "03_compliance_vs_cmod_final.png")
        
        if len(activos_finales) > 0:
            
            graficar_compliance_vs_cmod(
                [r["x_descarga"] for r in activos_finales],
                [r["Ci"] for r in activos_finales],
                [r["descarga"] for r in activos_finales],
                os.path.basename(ruta),
                mostrar_numeros=False,
                ruta_guardado=figuras_reporte["compliance_vs_cmod_final"],
                mostrar=False
            )

        # -------------------------------------------------
        # Corrección por rotación
        # -------------------------------------------------
        rot_cfg = preguntar_correccion_rotacion()

        if rot_cfg["usar"]:
            aplicar_correccion_rotacion_a_resultados(resultados_seleccionados, rot_cfg)

            tabla_corr = pd.DataFrame(resultados_seleccionados)

            print("\n" + "=" * 100)
            print("TABLA RESUMEN CON CORRECCIÓN POR ROTACIÓN")
            print("=" * 100)
            print(
                tabla_corr[
                    [
                        "descarga",
                        "cmod_medio",
                        "v_inicio_descarga",
                        "Ci",
                        "Ci_final",
                        "a_i_mm",
                        "a_i_W",
                        "R_i_mm",
                        "theta_deg",
                        "u",
                        "seleccionada",
                        "eliminada",
                        "eliminada_corregida",
                    ]
                ].to_string(index=False)
            )

            print("\nAhora se abrirá la curva Compliance corregida por rotación vs CMOD.")
            print("Se usarán solamente los puntos que sobrevivieron a la curva original.")
            print("Cada punto tendrá arriba el número de descarga correspondiente.")
            print("Ingrese el número que quiera eliminar de la gráfica corregida.")
            print("Cuando termine, ingrese 0 para mostrar el gráfico final corregido.\n")

            loop_eliminacion_interactiva_corregida(
                resultados_seleccionados,
                os.path.basename(ruta)
            )
            
            activos_corr_finales = [
                r for r in resultados_seleccionados
                if r["seleccionada"]
                and not r.get("eliminada", False)
                and not r.get("eliminada_corregida", False)
                and not np.isnan(r["Ci_final"])
                and not np.isnan(r["cmod_medio"])
            ]
            
            figuras_reporte["compliance_corregida_vs_cmod_final"] = os.path.join(carpeta_figuras, "04_compliance_corregida_vs_cmod_final.png")
            
            if len(activos_corr_finales) > 0:
                graficar_compliance_vs_cmod(
                    [r["x_descarga"] for r in activos_corr_finales],
                    [r["Ci_final"] for r in activos_corr_finales],
                    [r["descarga"] for r in activos_corr_finales],
                    f"{os.path.basename(ruta)} - corregida por rotación",
                    mostrar_numeros=False,
                    ruta_guardado=figuras_reporte["compliance_corregida_vs_cmod_final"],
                    mostrar=False
                )
            

            # -------------------------------------------------
            # Determinación opcional de a0q usando Ci_final
            # -------------------------------------------------
            resultado_a0q = None

            if preguntar_si_calcular_a0q():
                resultado_a0q = recalculo_interactivo_a0q(
                    resultados_seleccionados=resultados_seleccionados,
                    W=rot_cfg["W"],
                    B=rot_cfg["B"],
                    BN=rot_cfg["BN"],
                    E=rot_cfg["E"],
                    nu=rot_cfg["nu"]
                )
            else:
                print("\nNo se calcularán Δa ni J porque el usuario decidió no calcular a0q.")
                return

            if resultado_a0q is None:
                print("\nNo se pudo determinar a0q. Se cancela el cálculo de Δa y J.")
                return

            if resultado_a0q["cumple"]:
                a0_base = resultado_a0q["a0q"]
                fuente_a0 = "a0q_normativo"
            else:
                decision_a0 = resolver_a0_si_no_cumple(resultado_a0q)

                if decision_a0 is None:
                    print("\nNo se pudo resolver el valor de a0. Se cancela el cálculo de Δa y J.")
                    return

                if not decision_a0["continuar"]:
                    print("\nNo se calcularán Δa ni J.")
                    return

                a0_base = decision_a0["a0"]
                fuente_a0 = decision_a0["fuente"]
            
            print(f"\nSe calculará Δa_i = a_i - a0")
            print(f"a0 utilizado = {a0_base:.6f} mm | fuente = {fuente_a0}")
            
            calcular_delta_a(
                resultados_seleccionados,
                a0_base,
                rot_cfg["W"],
            )
            
            mostrar_tabla_delta_a(resultados_seleccionados, a0_base)
            graficar_delta_a_vs_cmod(
                resultados_seleccionados,
                os.path.basename(ruta)
            )
            
            figuras_reporte["delta_a_vs_cmod"] = os.path.join(carpeta_figuras, "05_delta_a_vs_cmod.png")

            graficar_delta_a_vs_cmod(
                resultados_seleccionados,
                os.path.basename(ruta),
                ruta_guardado=figuras_reporte["delta_a_vs_cmod"],
                mostrar=True
            )
            
            print("\nAhora se calculará J para cada descarga corregida.")
            
            figuras_reporte["jr_auto"] = os.path.join(carpeta_figuras, "06_jr_auto.png")
            

            J_vals_B, da_B = calcular_J_integral_ct(
                resultados_seleccionados=resultados_seleccionados,
                cmod=cmod,
                carga=carga,
                E=rot_cfg["E"],
                nu=rot_cfg["nu"],
                W=rot_cfg["W"],
                B=rot_cfg["B"],
                BN=rot_cfg["BN"],
                d=rot_cfg["d"],
                H_ast=rot_cfg["H_ast"],
                ruta_guardado_jr_auto=figuras_reporte["jr_auto"],
                mostrar=True,
                verbose=True
            )
            
            imprimir_diagnostico_ci_ai_j(resultados_seleccionados, a0_base)
            

            print(
                f'Desc {r["descarga"]:>2} | '
                f'Ci={r.get("Ci_final", np.nan):.6f} | '
                f'a={r.get("a_i_mm", np.nan):.4f} | '
                f'Δa={r.get("delta_a_mm", np.nan):.4f} | '
                f'P={r.get("P_i_kN", np.nan):.3f} | '
                f'vpl={r.get("vpl_mm", np.nan):.4f} | '
                f'dApl={r.get("dApl_kN_mm", np.nan):.4f} | '
                f'Jpl={r.get("J_pl_kJ_m2", np.nan):.2f} | '
                f'J={r.get("J_total_kJ_m2", np.nan):.2f}'
            )
            
            print("\nSe abrirá la curva J-R para revisión manual.")
            print("Cada punto tendrá arriba el número de descarga correspondiente.")
            print("Ingrese el número que quiera eliminar de la curva J-R.")
            print("Cuando termine, ingrese 0 para mostrar el gráfico final.\n")
            
            loop_eliminacion_interactiva_jr(
                resultados_seleccionados,
                os.path.basename(ruta)
            )
            
            figuras_reporte["jr_final"] = os.path.join(carpeta_figuras, "07_jr_final.png")

            graficar_J_vs_delta_a(
                resultados_seleccionados,
                os.path.basename(ruta),
                ruta_guardado=figuras_reporte["jr_final"],
                mostrar=False
            )
            
            mostrar_tabla_J(resultados_seleccionados)
            
            print("\nAhora se hará el análisis ASTM posterior sobre la curva J-R.")
            
            while True:
                try:
                    sigma_y = float(input("Ingrese sigma_y [MPa]: ").strip())
                    if sigma_y <= 0:
                        print("sigma_y debe ser mayor que cero.")
                        continue
                    break
                except ValueError:
                    print("Entrada inválida. Ingrese un número.")
            
            figuras_reporte["analisis_astm_jr"] = os.path.join(
                carpeta_figuras,
                "08_analisis_astm_jr.png"
            )

            analisis_astm_post_jr(
                resultados_seleccionados=resultados_seleccionados,
                nombre_archivo=os.path.basename(ruta),
                W=rot_cfg["W"],
                a0=a0_base,
                B=rot_cfg["B"],
                sigma_y_MPa=sigma_y,
                ruta_guardado=figuras_reporte["analisis_astm_jr"],
                mostrar=True
            )
                        
            for r in resultados_seleccionados:
                if (
                    r["seleccionada"]
                    and not r.get("eliminada", False)
                    and not r.get("eliminada_corregida", False)
                    and not r.get("eliminada_jr", False)
                    and not np.isnan(r.get("a_i_mm", np.nan))
                    and not np.isnan(r.get("delta_a_mm", np.nan))
                    and not np.isnan(r.get("J_total_kJ_m2", np.nan))
                ):
                    print(
                        f"Desc {r['descarga']:02d} | "
                        f"a = {r['a_i_mm']:.4f} mm | "
                        f"Δa = {r['delta_a_mm']:.4f} mm | "
                        f"J = {r['J_total_kJ_m2']:.4f} kJ/m²"
                    )

        else:
            print("\nNo se aplicó corrección por rotación.")
            print("No se calcularán a0q, Δa ni J en esta corrida.")

    except Exception as e:
        print("\nOcurrió un error al procesar el archivo:")
        print(str(e))

        print("\n" + "=" * 100)
        print("FIGURAS GUARDADAS PARA EL REPORTE")
        print("=" * 100)
        for nombre_fig, ruta_fig in figuras_reporte.items():
            print(f"{nombre_fig}: {ruta_fig}")


if __name__ == "__main__":
    main()










