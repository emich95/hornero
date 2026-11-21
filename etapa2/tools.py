"""
tools.py (etapa 2): las tools de RRHH y de Supply chain, y la fecha de hoy.

Las tools de Marketing están en tools_marketing.py, que es una copia sin cambios de etapa1/tools.py.

Mínimo privilegio: este módulo carga los datos de las dos áreas, pero cada agente recibe solo las
tools de la suya (ver agentes.py). El agente de políticas solo puede buscar en el manual; el de people
analytics solo ve empleados.db y el modelo de attrition; el de supply chain solo ve ventas.db, el
modelo de ventas y el pronóstico.

Recordatorios de la clase 4 que se siguen acá:
- El docstring de cada tool es lo que el LLM lee para decidir cuándo y cómo usarla.
- Todas reciben al menos un parámetro y devuelven texto corto: resúmenes, no tablas enteras.
- El texto libre se normaliza (mayúsculas, tildes) y un valor inválido devuelve un mensaje con los
  valores válidos, para que el LLM corrija la llamada en vez de inventar.
- Modelos, índices y bases se cargan una sola vez, al importar el módulo.
"""

import asyncio
import json
import re
import sqlite3
import sys
import threading
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from sqlalchemy import create_engine

from preparacion import CATEGORIAS, REGIONES, VARIABLES_VENTAS
from rag import IndicePDF

# Carpeta con los datos del TP (datos/etapa1 y datos/etapa2)
RUTA_DATOS = Path(__file__).parent / "datos"
RUTA_ETAPA2 = RUTA_DATOS / "etapa2"
RUTA_MODELOS = Path(__file__).parent / "modelos"
RUTA_BASES = Path(__file__).parent / "bases"

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def _plano(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar texto libre."""
    texto = unicodedata.normalize("NFKD", str(texto).lower()).encode("ascii", "ignore").decode()
    return " ".join(texto.replace("_", " ").split())     # "satisfaccion_clima" = "satisfaccion clima"


def elegir(valor: str, opciones: list[str]) -> str | None:
    """Lleva un valor escrito libremente a una de las opciones ('aires' -> 'Aires acondicionados',
    'cd cordoba' -> 'CD Córdoba'). None si no coincide con ninguna o si coincide con más de una."""
    plano = _plano(valor)
    if not plano:
        return None
    exactas = [o for o in opciones if _plano(o) == plano]
    if exactas:
        return exactas[0]
    parciales = [o for o in opciones if _plano(o).startswith(plano) or plano in _plano(o)]
    return parciales[0] if len(parciales) == 1 else None


def _numero(x: float, decimales: int = 0) -> str:
    """Número con formato argentino: 1.234,5"""
    x = float(x) + 0.0 if round(float(x), decimales) != 0 else 0.0      # evita "-0,0"
    return f"{x:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ==========================================================================
# Fecha
# ==========================================================================

@tool
def fecha_de_hoy(formato: Literal["numerico", "texto"]) -> str:
    """Devuelve la fecha de hoy. formato='numerico' devuelve AAAA-MM-DD;
    formato='texto' devuelve, por ejemplo, 'sábado 7 de noviembre de 2026'."""
    hoy = date.today()
    if formato == "texto":
        return f"{DIAS[hoy.weekday()]} {hoy.day} de {MESES[hoy.month - 1]} de {hoy.year}"
    return hoy.isoformat()


def semana_que_viene(hoy: date | None = None) -> tuple[date, date]:
    """Lunes y domingo de la semana siguiente a la de hoy."""
    hoy = hoy or date.today()
    lunes = hoy + timedelta(days=7 - hoy.weekday())
    return lunes, lunes + timedelta(days=6)


# ==========================================================================
# RRHH: políticas (RAG sobre el manual)
# ==========================================================================

indice_manual = IndicePDF(RUTA_ETAPA2 / "manual_politicas_rrhh.pdf", "manual_rrhh", k=3)


@tool
def buscar_politica(consulta: str) -> str:
    """Busca en el manual de políticas de RRHH de Hornero las 3 secciones más relacionadas con la
    consulta (trabajo híbrido, reintegros, capacitación, licencias, beneficios, código de conducta,
    contacto con RRHH); si dos son del mismo tema, suma el resto de ese tema. Devuelve el texto de
    cada sección con su título. Si ninguna responde la consulta, el manual no lo dice."""
    if not consulta.strip():
        return "La consulta está vacía: decime qué política buscás."
    resultados = indice_manual.buscar(consulta)
    # Si dos resultados son del mismo tema, la consulta es sobre el tema entero y se suma el resto (en la consulta
    # 10, «capacitación» trajo 3.1, 3.2 y 3.4, y quedó afuera 3.3 Posgrados)
    temas = [r["seccion"].split(" > ")[0] for r in resultados]
    repetidos = {t for t in temas if temas.count(t) >= 2}
    vistos = {r["texto"] for r in resultados}
    resto = [c for c in indice_manual.chunks if c["seccion"].split(" > ")[0] in repetidos and c["texto"] not in vistos]
    return "\n\n".join([f"(similitud {r['puntaje']:.2f})\n{r['texto']}" for r in resultados]
                       + [f"(mismo tema)\n{c['texto']}" for c in resto])


# ==========================================================================
# Bases SQL de solo lectura (una por área)
# ==========================================================================
# Patrón de la clase 4: la tabla se escribe una vez con un engine de escritura, que después se descarta.
# Los agentes SQL abren la base con mode=ro: cualquier INSERT, UPDATE o DELETE falla con
# "attempt to write a readonly database". Cada área tiene su propio archivo, así el agente de
# people analytics no puede leer ventas ni el de supply chain puede leer empleados.

def preparar_base(ruta_csv: Path, tabla: str) -> Path:
    """Crea bases/<tabla>.db a partir del CSV, solo si no existe o si el CSV es más nuevo."""
    ruta_db = RUTA_BASES / f"{tabla}.db"
    if not ruta_db.exists() or ruta_db.stat().st_mtime < ruta_csv.stat().st_mtime:
        RUTA_BASES.mkdir(exist_ok=True)
        ruta_db.unlink(missing_ok=True)
        escritura = create_engine(f"sqlite:///{ruta_db}")
        pd.read_csv(ruta_csv).to_sql(tabla, escritura, index=False)
        escritura.dispose()
    return ruta_db


class BaseRecortada(SQLDatabase):
    """SQLDatabase que recorta resultados largos. Un SELECT * sobre ventas devuelve ~450.000 caracteres:
    sin el recorte, una sola consulta mal planteada se come el cupo de tokens de Groq."""
    MAX_CARACTERES = 2500

    def run(self, command, fetch="all", **kwargs):
        resultado = super().run(command, fetch, **kwargs)
        if isinstance(resultado, str) and len(resultado) > self.MAX_CARACTERES:
            return (resultado[:self.MAX_CARACTERES] + f"\n[Resultado recortado: tenía {len(resultado):,} caracteres. "
                    "Reescribí la consulta con GROUP BY, funciones de agregación o LIMIT.]")
        return resultado


def base_solo_lectura(ruta_db: Path) -> SQLDatabase:
    """SQLDatabase de LangChain sobre la base abierta en modo solo lectura. La URI se arma con
    as_uri() para que funcione aunque la ruta tenga espacios."""
    uri = ruta_db.resolve().as_uri() + "?mode=ro"
    motor = create_engine("sqlite://", creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False))
    return BaseRecortada(motor, sample_rows_in_table_info=2)


base_empleados = base_solo_lectura(preparar_base(RUTA_ETAPA2 / "empleados.csv", "empleados"))
base_ventas = base_solo_lectura(preparar_base(RUTA_ETAPA2 / "ventas.csv", "ventas"))


# ==========================================================================
# RRHH: people analytics (modelo de attrition)
# ==========================================================================

ATTRITION = joblib.load(RUTA_MODELOS / "attrition.joblib")
CAT_ATT, NUM_ATT = ATTRITION["features_cat"], ATTRITION["features_num"]
VALORES_CAT = ATTRITION["valores_cat"]          # valores válidos de area, centro, nivel y modalidad
NOMBRES_VARIABLES = {
    "antiguedad_anios": "antigüedad (años)", "sueldo_vs_mercado_pct": "sueldo frente al mercado (puntos)",
    "meses_sin_aumento": "meses sin aumento", "horas_extra_mes": "horas extra por mes",
    "distancia_km": "distancia al trabajo (km)", "ultima_evaluacion": "última evaluación (1 a 5)",
    "capacitaciones_ultimo_anio": "capacitaciones en el último año", "satisfaccion_clima": "satisfacción con el clima (1 a 5)",
    "area": "área", "centro": "centro", "nivel": "nivel", "modalidad": "modalidad",
}
# Rangos posibles al simular: las escalas de 1 a 5 no se pasan de sus bordes y las cantidades no son negativas
LIMITES_NATURALES = {"ultima_evaluacion": (1, 5), "satisfaccion_clima": (1, 5), "sueldo_vs_mercado_pct": (-100, None)}

EMPLEADOS = pd.read_csv(RUTA_ETAPA2 / "empleados.csv")
# Decisión D7: el riesgo se calcula sobre los activos (los que renunciaron ya no están en riesgo).
# Las probabilidades se calculan una sola vez, al importar (patrón de la clase 2).
ACTIVOS = EMPLEADOS[EMPLEADOS["renuncio"] == "No"].copy()
ACTIVOS["prob"] = ATTRITION["pipeline"].predict_proba(ACTIVOS[CAT_ATT + NUM_ATT])[:, 1]


def _filtrar_activos(area: str, centro: str, nivel: str, modalidad: str) -> tuple[pd.DataFrame, str, str]:
    """Activos que cumplen los filtros. Devuelve (datos, descripción del filtro, error)."""
    datos, partes = ACTIVOS, []
    for columna, valor in [("area", area), ("centro", centro), ("nivel", nivel), ("modalidad", modalidad)]:
        if not (valor or "").strip():
            continue
        elegido = elegir(valor, VALORES_CAT[columna])
        if elegido is None:
            return datos.iloc[0:0], "", (f"«{valor}» no es un valor válido de {columna}. "
                                         f"Valores posibles: {', '.join(VALORES_CAT[columna])}.")
        datos = datos[datos[columna] == elegido]
        partes.append(f"{columna} = {elegido}")
    return datos, (", ".join(partes) or "todos los activos"), ""


def _umbral_valido(umbral: float) -> str:
    return "" if 0 < umbral < 1 else f"El umbral tiene que estar entre 0 y 1 (por ejemplo, 0.5); recibí {umbral}."


@tool
def predecir_riesgo_attrition(umbral: float = 0.5, area: str = "", centro: str = "", nivel: str = "",
                              modalidad: str = "", agrupar_por: str = "", top_n: int = 0) -> str:
    """Predice con el modelo de attrition cuántos empleados ACTIVOS están en riesgo de renunciar.
    En riesgo = probabilidad de renuncia >= umbral (0.5 por defecto; decilo siempre en la respuesta).
    Filtros opcionales: area, centro, nivel, modalidad (vacío = sin filtro).
    agrupar_por: '', 'area', 'centro', 'nivel' o 'modalidad', para ver el riesgo por grupo.
    top_n: cuántos empleados con mayor probabilidad listar (solo su id), 0 para ninguno."""
    error = _umbral_valido(umbral)
    datos, filtro, error_filtro = _filtrar_activos(area, centro, nivel, modalidad)
    if error or error_filtro:
        return error or error_filtro
    if datos.empty:
        return f"No hay empleados activos con {filtro}."
    en_riesgo = datos["prob"] >= umbral
    lineas = [f"Empleados activos considerados ({filtro}): {_numero(len(datos))}.",
              f"En riesgo (probabilidad de renuncia >= {_numero(umbral, 2)}): {_numero(en_riesgo.sum())} "
              f"({_numero(en_riesgo.mean() * 100, 1)} %).",
              f"Probabilidad media de renuncia: {_numero(datos['prob'].mean() * 100, 1)} %."]
    if agrupar_por:
        columna = elegir(agrupar_por, CAT_ATT)
        if columna is None:
            return f"agrupar_por tiene que ser area, centro, nivel o modalidad; recibí «{agrupar_por}»."
        grupos = (datos.assign(riesgo=en_riesgo).groupby(columna)
                  .agg(activos=("prob", "size"), en_riesgo=("riesgo", "sum"), prob_media=("prob", "mean"))
                  .assign(pct=lambda g: g["en_riesgo"] / g["activos"] * 100).sort_values("pct", ascending=False))
        lineas.append(f"Por {columna} (de mayor a menor % en riesgo):")
        lineas += [f"- {nombre}: {_numero(g['en_riesgo'])} de {_numero(g['activos'])} en riesgo "
                   f"({_numero(g['pct'], 1)} %), probabilidad media {_numero(g['prob_media'] * 100, 1)} %"
                   for nombre, g in grupos.iterrows()]
    if top_n > 0:
        top = datos.nlargest(min(top_n, 20), "prob")
        lineas.append("Mayor probabilidad: " + ", ".join(f"id {int(f.id_empleado)} ({_numero(f.prob * 100, 0)} %)"
                                                        for f in top.itertuples()))
    lineas.append("Ojo: son predicciones sobre empleados que el modelo vio al entrenar.")
    return "\n".join(lineas)


@tool
def explicar_factores_attrition(top_n: int = 6) -> str:
    """Explica qué variables pesan más en el modelo de attrition y en qué sentido actúan (si suben o
    bajan las chances de renunciar). top_n: cuántas variables mostrar, de la más a la menos importante."""
    importancia = sorted(ATTRITION["importancia_permutacion_auc"].items(), key=lambda kv: -kv[1])
    lineas = ["Variables del modelo de attrition (regresión logística), de la que más pesa a la que menos. "
              "Importancia = cuánto cae el AUC en test si se desordena la variable."]
    for variable, caida in importancia[:max(1, top_n)]:
        if variable in ATTRITION["efectos_numericos"]:
            odds = ATTRITION["efectos_numericos"][variable]["odds_ratio_por_unidad"]
            # «cada unidad más baja las chances» se leía también como «cada unidad más baja (menor)»: en la
            # consulta 4 el LLM dio vuelta el sentido de capacitaciones
            sentido = (f"si sube 1 unidad, las chances de renunciar {'aumentan' if odds > 1 else 'disminuyen'} un "
                       f"{_numero(abs(odds - 1) * 100, 1)} %")
        else:
            efectos = ATTRITION["efectos_categoricos"][variable]
            mayor, menor = max(efectos, key=efectos.get), min(efectos, key=efectos.get)
            sentido = f"mayor riesgo en {mayor}, menor en {menor} (a igualdad del resto)"
        lineas.append(f"- {NOMBRES_VARIABLES.get(variable, variable)}: caída de AUC {_numero(caida, 3)}; {sentido}.")
    lineas.append("Son asociaciones que encontró el modelo, no relaciones causales.")
    return "\n".join(lineas)


@tool
def simular_escenario_attrition(variable: str, cambio: float = 0.0,
                                tipo_cambio: Literal["absoluto", "porcentual"] = "absoluto",
                                limite: float | None = None, nuevo_valor: str = "", umbral: float = 0.5,
                                area: str = "", centro: str = "", nivel: str = "", modalidad: str = "") -> str:
    """Simula un escenario con el modelo de attrition: cambia una variable a los empleados ACTIVOS
    (todos o los del filtro area/centro/nivel/modalidad) y compara el riesgo antes y después.
    variable numérica (sueldo_vs_mercado_pct, meses_sin_aumento, horas_extra_mes, satisfaccion_clima,
    capacitaciones_ultimo_anio, antiguedad_anios, distancia_km, ultima_evaluacion): se le suma 'cambio'
    (tipo_cambio='absoluto', en las unidades de la variable) o un porcentaje (tipo_cambio='porcentual').
    limite (opcional): valor que la variable no puede pasar con el cambio; a quien ya está del otro lado
    no se le cambia. Ejemplo: acercar los sueldos 10 puntos al mercado = variable='sueldo_vs_mercado_pct',
    cambio=10, limite=0 (quien está debajo del mercado sube hasta 10 puntos sin pasar de 0).
    variable categórica (area, centro, nivel, modalidad): se pasa a 'nuevo_valor' para todos los del filtro.
    umbral: probabilidad desde la que alguien está en riesgo (0.5 por defecto)."""
    error = _umbral_valido(umbral)
    datos, filtro, error_filtro = _filtrar_activos(area, centro, nivel, modalidad)
    if error or error_filtro:
        return error or error_filtro
    if datos.empty:
        return f"No hay empleados activos con {filtro}."
    columna = elegir(variable, NUM_ATT + CAT_ATT)
    if columna is None:
        return f"«{variable}» no es una variable del modelo. Variables: {', '.join(NUM_ATT + CAT_ATT)}."
    escenario = datos[CAT_ATT + NUM_ATT].copy()
    if columna in CAT_ATT:
        valor = elegir(nuevo_valor, VALORES_CAT[columna])
        if valor is None:
            return f"Para {columna} indicá nuevo_valor entre: {', '.join(VALORES_CAT[columna])}."
        escenario[columna] = valor
        descripcion = f"{columna} = {valor}"
    else:
        if cambio == 0:
            return "Indicá un cambio distinto de 0."
        actual = escenario[columna].astype(float)
        nuevo = actual + cambio if tipo_cambio == "absoluto" else actual * (1 + cambio / 100)
        if limite is not None:   # no se cruza el límite, y a quien ya lo cruzó no se lo toca
            if cambio > 0:
                nuevo = np.where(actual >= limite, actual, np.minimum(nuevo, limite))
            else:
                nuevo = np.where(actual <= limite, actual, np.maximum(nuevo, limite))
        piso, techo = LIMITES_NATURALES.get(columna, (0, None))
        escenario[columna] = np.clip(nuevo, piso, techo)
        descripcion = (f"{NOMBRES_VARIABLES[columna]} {'+' if cambio > 0 else ''}{_numero(cambio, 1)}"
                       f"{' %' if tipo_cambio == 'porcentual' else ''}"
                       + (f", sin pasar de {_numero(limite, 1)}" if limite is not None else ""))
    cambiados = int((escenario[columna] != datos[columna]).sum())
    antes = datos["prob"].to_numpy()
    despues = ATTRITION["pipeline"].predict_proba(escenario)[:, 1]
    riesgo_antes, riesgo_despues = antes >= umbral, despues >= umbral
    return "\n".join([
        f"Escenario: {descripcion}. Población: {filtro} ({_numero(len(datos))} activos; el cambio alcanza a "
        f"{_numero(cambiados)}). Umbral de riesgo: {_numero(umbral, 2)}.",
        f"En riesgo antes: {_numero(riesgo_antes.sum())}; después: {_numero(riesgo_despues.sum())}.",
        f"Dejan de estar en riesgo: {_numero((riesgo_antes & ~riesgo_despues).sum())}; "
        f"pasan a estar en riesgo: {_numero((~riesgo_antes & riesgo_despues).sum())}.",
        f"Probabilidad media de renuncia: {_numero(antes.mean() * 100, 1)} % -> {_numero(despues.mean() * 100, 1)} % "
        f"(renuncias esperadas: {_numero(antes.sum())} -> {_numero(despues.sum())}).",
        "Es una asociación del modelo, no un efecto causal garantizado.",
    ])


# ==========================================================================
# Supply chain: modelo de ventas
# ==========================================================================

VENTAS_MODELO = joblib.load(RUTA_MODELOS / "ventas.joblib")
DUAN = VENTAS_MODELO["factor_correccion_log"]     # corrige la vuelta de log a unidades (ver notas del informe)
VENTAS = pd.read_csv(RUTA_ETAPA2 / "ventas.csv", parse_dates=["semana"])
SEMANAS = np.sort(VENTAS["semana"].unique())
NOMBRES_VENTAS = {"temperatura_media_c": "temperatura media (°C)", "lluvia_mm": "lluvia semanal (mm)",
                  "precio_cobre_usd_ton": "precio del cobre (USD/t)"}


def _predecir_unidades(datos: pd.DataFrame) -> np.ndarray:
    return VENTAS_MODELO["pipeline"].predict(datos[["categoria", "region"] + VARIABLES_VENTAS]) * DUAN


def _lista(valores: list[str] | str | None, opciones: list[str], nombre: str) -> tuple[list[str], str]:
    """Normaliza una lista de categorías o regiones. Vacía o 'todas' = todas."""
    if isinstance(valores, str):
        valores = [v for v in re.split(r",| y ", valores) if v.strip()]
    if not valores or any(_plano(v) in ("todas", "todos") for v in valores):
        return list(opciones), ""
    elegidos = []
    for v in valores:
        e = elegir(v, opciones)
        if e is None:
            return [], f"«{v}» no es una {nombre} válida. Opciones: {', '.join(opciones)}."
        elegidos.append(e)
    return list(dict.fromkeys(elegidos)), ""


@tool
def predecir_ventas(categoria: str, region: str, temperatura_media_c: float | None = None,
                    lluvia_mm: float | None = None, precio_cobre_usd_ton: float | None = None) -> str:
    """Predice con el modelo de ventas las unidades que se venderían en UNA semana de una categoría en
    una región, dado el clima de la semana (temperatura media en °C y lluvia total de la semana en mm)
    y el precio del cobre (USD por tonelada). Si falta un valor, usa el promedio de las últimas 8
    semanas de la región (clima) o el último precio del cobre, y lo aclara.
    Categorías: Aires acondicionados, Calefacción, Cocina, Electro, Jardín, Muebles.
    Regiones: AMBA, Centro, Cuyo, Litoral, NOA, Patagonia."""
    cat, reg = elegir(categoria, CATEGORIAS), elegir(region, REGIONES)
    if cat is None or reg is None:
        return (f"Categoría o región inválida. Categorías: {', '.join(CATEGORIAS)}. "
                f"Regiones: {', '.join(REGIONES)}.")
    clima = VENTAS_MODELO["clima_ultimas_8_semanas"][reg]
    valores, supuestos = {}, []
    for variable, dado, defecto, origen in [
            ("temperatura_media_c", temperatura_media_c, clima["temperatura_media_c"], "promedio de las últimas 8 semanas"),
            ("lluvia_mm", lluvia_mm, clima["lluvia_mm"], "promedio de las últimas 8 semanas"),
            ("precio_cobre_usd_ton", precio_cobre_usd_ton, VENTAS_MODELO["cobre_ultimo"], "último precio registrado")]:
        valores[variable] = float(defecto if dado is None else dado)
        if dado is None:
            supuestos.append(f"{NOMBRES_VENTAS[variable]} = {_numero(valores[variable], 1)} ({origen})")
    fila = pd.DataFrame([{"categoria": cat, "region": reg, **valores}])
    unidades = float(_predecir_unidades(fila)[0])
    sigma = VENTAS_MODELO["desvio_residuo_log"]
    bajo, alto = unidades * np.exp(-1.28 * sigma), unidades * np.exp(1.28 * sigma)
    lineas = [f"{cat} en {reg}: {_numero(unidades)} unidades en la semana (rango probable del 80 %: "
              f"{_numero(bajo)} a {_numero(alto)}).",
              f"Entradas: temperatura media {_numero(valores['temperatura_media_c'], 1)} °C, lluvia "
              f"{_numero(valores['lluvia_mm'], 1)} mm, cobre {_numero(valores['precio_cobre_usd_ton'])} USD/t."]
    if supuestos:
        lineas.append("Supuestos: " + "; ".join(supuestos) + ".")
    t_min, t_max = VENTAS_MODELO["rango_temperatura_por_region"][reg]
    c_min, c_max = VENTAS_MODELO["rangos"]["precio_cobre_usd_ton"]
    if not t_min <= valores["temperatura_media_c"] <= t_max or not c_min <= valores["precio_cobre_usd_ton"] <= c_max:
        lineas.append("Ojo: alguna entrada está fuera del rango histórico; la predicción extrapola.")
    return "\n".join(lineas)


@tool
def explicar_factores_ventas(categoria: str = "todas") -> str:
    """Explica qué mueve las ventas según el modelo: el efecto de la temperatura, la lluvia y el precio
    del cobre sobre las unidades de cada categoría (en %) y qué regiones venden más.
    categoria: una categoría o 'todas'."""
    categorias, error = _lista(categoria, CATEGORIAS, "categoría")
    if error:
        return error
    lineas = ["Efecto sobre las unidades vendidas (modelo lineal sobre log de las unidades; a igualdad del resto):"]
    for cat in categorias:
        e = VENTAS_MODELO["efectos_legibles"][cat]
        lineas.append(f"- {cat}: +1 °C -> {_numero(e['pct_por_1_grado'], 1)} %; +10 mm de lluvia -> "
                      f"{_numero(e['pct_por_10_mm_lluvia'], 1)} %; +1.000 USD/t de cobre -> "
                      f"{_numero(e['pct_por_1000_usd_cobre'], 1)} %.")
    por_region = (VENTAS[VENTAS["categoria"].isin(categorias)].groupby(["region", "semana"])["unidades_vendidas"]
                  .sum().groupby("region").mean().sort_values(ascending=False))
    lineas.append("Unidades por semana en promedio por región (" + ("todas las categorías" if len(categorias) > 1
                  else categorias[0]) + "): " + ", ".join(f"{r} {_numero(u)}" for r, u in por_region.items()) + ".")
    lineas.append("Efectos cercanos a 0 % significan que esa variable no mueve a esa categoría.")
    return "\n".join(lineas)


@tool
def simular_escenario_ventas(variable: str, cambio: float,
                             tipo_cambio: Literal["absoluto", "porcentual"] = "porcentual",
                             categorias: list[str] | str | None = None, regiones: list[str] | str | None = None,
                             semanas_base: int = 8) -> str:
    """Simula un escenario con el modelo de ventas: toma como base las últimas 'semanas_base' semanas
    (8 por defecto), le cambia una variable y compara las unidades por semana antes y después, por
    categoría. variable: precio_cobre_usd_ton, temperatura_media_c o lluvia_mm. tipo_cambio='porcentual'
    (cambio en %, p. ej. 20 = +20 %) o 'absoluto' (en las unidades de la variable, p. ej. +2 °C).
    categorias y regiones: listas de nombres (o texto separado por comas); vacías = todas."""
    columna = elegir(variable, VARIABLES_VENTAS) or {"cobre": "precio_cobre_usd_ton", "temperatura": "temperatura_media_c",
                                                     "lluvia": "lluvia_mm"}.get(_plano(variable))
    if columna is None:
        return f"«{variable}» no es una variable del modelo. Variables: {', '.join(VARIABLES_VENTAS)}."
    cats, error_c = _lista(categorias, CATEGORIAS, "categoría")
    regs, error_r = _lista(regiones, REGIONES, "región")
    if error_c or error_r:
        return error_c or error_r
    semanas_base = int(min(max(semanas_base, 1), len(SEMANAS)))
    base = VENTAS[(VENTAS["semana"] >= SEMANAS[-semanas_base]) & VENTAS["categoria"].isin(cats) & VENTAS["region"].isin(regs)].copy()
    escenario = base.copy()
    escenario[columna] = base[columna] + cambio if tipo_cambio == "absoluto" else base[columna] * (1 + cambio / 100)
    if columna == "lluvia_mm":
        escenario[columna] = escenario[columna].clip(lower=0)
    base["antes"], base["despues"] = _predecir_unidades(base), _predecir_unidades(escenario)
    resumen = base.groupby("categoria")[["antes", "despues"]].sum() / semanas_base
    desde = SEMANAS[-semanas_base].astype("datetime64[D]").item()
    lineas = [f"Escenario: {NOMBRES_VENTAS[columna]} {'+' if cambio > 0 else ''}{_numero(cambio, 1)}"
              f"{' %' if tipo_cambio == 'porcentual' else ''}. Base: las últimas {semanas_base} semanas (desde el "
              f"{desde:%d/%m/%Y}), regiones: {', '.join(regs) if len(regs) < len(REGIONES) else 'todas'}.",
              "Unidades por semana según el modelo, antes -> después:"]
    # La diferencia va escrita: en la consulta 6 el LLM tomó el «después» (304) como lo que caían las ventas
    for cat, f in resumen.iterrows():
        diferencia = round(f["despues"]) - round(f["antes"])
        lineas.append(f"- {cat}: {_numero(f['antes'])} -> {_numero(f['despues'])} por semana; "
                      f"{'suben' if diferencia >= 0 else 'caen'} {_numero(abs(diferencia))} unidades por semana "
                      f"({'+' if f['despues'] >= f['antes'] else ''}{_numero((f['despues'] / f['antes'] - 1) * 100, 1)} %)")
    v_min, v_max = VENTAS_MODELO["rangos"][columna]
    if escenario[columna].max() > v_max or escenario[columna].min() < v_min:
        lineas.append(f"Ojo: el escenario lleva {NOMBRES_VENTAS[columna]} fuera del rango histórico "
                      f"({_numero(v_min)} a {_numero(v_max)}); el resultado extrapola.")
    return "\n".join(lineas)


# ==========================================================================
# Supply chain: pronóstico del clima (MCP, con respaldo en el CSV)
# ==========================================================================
# El server mcp_weather_server corre como subproceso (stdio) y consulta Open-Meteo. Su respuesta trae
# el clima hora por hora (~100.000 caracteres para una semana) más instrucciones para el LLM: la tool
# la resume en código a los valores semanales que usa el modelo y no le pasa nada más al agente.
# Mínimo privilegio: de las 8 tools del server, el host usa solo get_weather_byDateTimeRange.

CIUDADES = {"AMBA": "Buenos Aires", "Centro": "Córdoba", "Cuyo": "Mendoza", "Litoral": "Rosario",
            "NOA": "San Miguel de Tucumán", "Patagonia": "Neuquén"}
SERVER_CLIMA = {"clima": {"command": sys.executable, "args": ["-m", "mcp_weather_server"], "transport": "stdio"}}
_tool_clima = None


def _correr(corrutina):
    """asyncio.run, o en un thread aparte si ya hay un event loop corriendo (por ejemplo, en Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(corrutina)
    resultado = {}
    hilo = threading.Thread(target=lambda: resultado.setdefault("valor", asyncio.run(corrutina)))
    hilo.start()
    hilo.join()
    return resultado["valor"]


async def _clima_mcp(ciudad: str, desde: date, hasta: date) -> dict:
    """Pide el pronóstico al server MCP y lo resume. Lanza una excepción si algo falla."""
    respuesta = await _tool_clima.ainvoke({"city": ciudad, "start_date": desde.isoformat(), "end_date": hasta.isoformat()})
    texto = respuesta[0]["text"] if isinstance(respuesta, list) else str(respuesta)
    if texto.strip().startswith("Error"):
        raise RuntimeError(texto.strip()[:200])
    inicio, fin = texto.index("{"), texto.rindex("}") + 1     # el JSON está entre el encabezado y las instrucciones
    datos = json.loads(texto[inicio:fin])
    if datos.get("latitude", 0) > 0:                          # geocodificó otra ciudad con el mismo nombre
        raise RuntimeError(f"el server ubicó «{ciudad}» en el hemisferio norte")
    horas = pd.DataFrame(datos["weather_data"])
    return {"temperatura_media_c": horas["temperature_c"].mean(), "temperatura_min_c": horas["temperature_c"].min(),
            "temperatura_max_c": horas["temperature_c"].max(), "lluvia_mm": horas["precipitation_mm"].sum()}


async def _climas_mcp(ciudades: list[str], desde: date, hasta: date) -> list:
    """Pronósticos de varias ciudades en paralelo (cada llamada levanta su propio proceso del server).
    Devuelve, por ciudad, el resumen o la excepción."""
    global _tool_clima
    from langchain_mcp_adapters.client import MultiServerMCPClient
    if _tool_clima is None:
        tools = await MultiServerMCPClient(SERVER_CLIMA).get_tools()
        _tool_clima = next(t for t in tools if t.name == "get_weather_byDateTimeRange")
    return await asyncio.gather(*[_clima_mcp(c, desde, hasta) for c in ciudades], return_exceptions=True)


def _clima_csv(region: str) -> tuple[dict, date, date]:
    datos = pd.read_csv(RUTA_ETAPA2 / "pronostico_clima.csv", parse_dates=["fecha"])
    datos = datos[datos["region"] == region]
    return ({"temperatura_media_c": datos["temperatura_media_c"].mean(), "temperatura_min_c": datos["temperatura_min_c"].min(),
             "temperatura_max_c": datos["temperatura_max_c"].max(), "lluvia_mm": datos["lluvia_mm"].sum()},
            datos["fecha"].min().date(), datos["fecha"].max().date())


def pronosticos(regiones: list[str]) -> dict:
    """Pronóstico de la semana que viene por región: {region: {"clima", "desde", "hasta", "fuente"}}.
    Usa el MCP (dos intentos: Open-Meteo a veces tarda más de lo que espera el server) y, si falla, el CSV."""
    desde, hasta = semana_que_viene()
    salida, pendientes, errores = {}, list(regiones), {}
    for _ in range(2):
        if not pendientes:
            break
        try:
            resultados = _correr(_climas_mcp([CIUDADES[r] for r in pendientes], desde, hasta))
        except Exception as e:                                # no levantó el server
            resultados = [e] * len(pendientes)
        for region, resultado in zip(list(pendientes), resultados):
            if isinstance(resultado, Exception):
                errores[region] = f"{type(resultado).__name__}: {str(resultado)[:150]}"
            else:
                salida[region] = {"clima": resultado, "desde": desde, "hasta": hasta,
                                  "fuente": "server MCP mcp_weather_server (Open-Meteo)"}
                pendientes.remove(region)
    for region in pendientes:
        clima, d, h = _clima_csv(region)
        salida[region] = {"clima": clima, "desde": d, "hasta": h,
                          "fuente": (f"pronostico_clima.csv, porque el server MCP falló ({errores[region]}). Ojo: el "
                                     f"CSV cubre del {d:%d/%m} al {h:%d/%m/%Y}, no necesariamente la semana que viene")}
    return salida


@tool
def pronostico_semana(region: str) -> str:
    """Devuelve el pronóstico del clima de la semana que viene (lunes a domingo) para una región, resumido
    como lo usa el modelo de ventas: temperatura media de la semana (°C) y lluvia total de la semana (mm),
    más la mínima y la máxima. Usa el server MCP del clima (Open-Meteo); si falla, usa el pronóstico
    guardado en pronostico_clima.csv y lo aclara. Regiones: AMBA, Centro, Cuyo, Litoral, NOA, Patagonia."""
    reg = elegir(region, REGIONES)
    if reg is None:
        return f"«{region}» no es una región válida. Regiones: {', '.join(REGIONES)}."
    p = pronosticos([reg])[reg]
    clima = p["clima"]
    return (f"Pronóstico para {CIUDADES[reg]} ({reg}), del {p['desde']:%d/%m} al {p['hasta']:%d/%m/%Y}: temperatura "
            f"media {_numero(clima['temperatura_media_c'], 1)} °C (mínima {_numero(clima['temperatura_min_c'], 1)} °C, "
            f"máxima {_numero(clima['temperatura_max_c'], 1)} °C), lluvia total {_numero(clima['lluvia_mm'], 1)} mm.\n"
            f"Fuente: {p['fuente']}.\nPara predecir las ventas de esa semana: temperatura_media_c = "
            f"{round(clima['temperatura_media_c'], 1)}, lluvia_mm = {round(clima['lluvia_mm'], 1)}.")


def clima_normal(region: str, desde: date) -> dict:
    """Temperatura media y lluvia semanal promedio de las mismas semanas del año (±10 días) en los años
    anteriores, según ventas.csv. Es la referencia de "clima normal para la época"."""
    semanas = VENTAS[(VENTAS["region"] == region) & (VENTAS["categoria"] == CATEGORIAS[0])]
    cerca = pd.Series(False, index=semanas.index)
    for anio in sorted(semanas["semana"].dt.year.unique()):
        if anio < desde.year:
            centro = pd.Timestamp(desde.replace(year=int(anio)))
            cerca |= (semanas["semana"] - centro).abs() <= pd.Timedelta(days=10)
    elegidas = semanas[cerca]
    return {"temperatura_media_c": float(elegidas["temperatura_media_c"].mean()),
            "lluvia_mm": float(elegidas["lluvia_mm"].mean()),
            "anios": sorted(elegidas["semana"].dt.year.unique().tolist())}


# Temperaturas semanales que registró cada región en ventas.csv: fuera de ese rango, el modelo extrapola
RANGO_TEMPERATURA = {reg: (g.min(), g.max()) for reg, g in VENTAS.groupby("region")["temperatura_media_c"]}


@tool
def prever_ventas_semana(categoria: str, regiones: list[str] | str | None = None) -> str:
    """Estima las unidades de una categoría para la semana que viene en una o varias regiones: pide el
    pronóstico del clima (server MCP; si falla, el CSV) y se lo pasa al modelo de ventas, con el último
    precio del cobre. Para cada región compara con lo que el modelo espera con un clima normal para la época
    (las mismas semanas de los años anteriores): así se ve dónde el pronóstico sube o baja la demanda más de
    lo habitual. Sirve para "cuánto vamos a vender la semana que viene" y para "en qué región conviene
    empujar una categoría". regiones: lista o texto; vacío = las 6."""
    cat = elegir(categoria, CATEGORIAS)
    if cat is None:
        return f"«{categoria}» no es una categoría válida. Categorías: {', '.join(CATEGORIAS)}."
    regs, error = _lista(regiones, REGIONES, "región")
    if error:
        return error
    cobre = VENTAS_MODELO["cobre_ultimo"]
    sigma = VENTAS_MODELO["desvio_residuo_log"]
    filas = []
    for reg, p in pronosticos(regs).items():
        normal = clima_normal(reg, p["desde"])
        datos = pd.DataFrame([{"categoria": cat, "region": reg, "precio_cobre_usd_ton": cobre,
                               "temperatura_media_c": clima["temperatura_media_c"], "lluvia_mm": clima["lluvia_mm"]}
                              for clima in (p["clima"], normal)])
        semana, base = _predecir_unidades(datos)
        filas.append((reg, semana, base, p, normal))
    anios = filas[0][4]["anios"]
    lineas = [f"{cat}, semana del {filas[0][3]['desde']:%d/%m} al {filas[0][3]['hasta']:%d/%m/%Y} (cobre: último precio, "
              f"{_numero(cobre)} USD/t). Clima normal = promedio de las mismas semanas de {', '.join(map(str, anios))}. "
              "De mayor a menor venta esperada:"]
    for reg, semana, base, p, normal in sorted(filas, key=lambda f: -f[1]):
        c = p["clima"]
        lineas.append(f"- {reg} ({CIUDADES[reg]}): {_numero(semana)} unidades (rango probable del 80 %: "
                      f"{_numero(semana * np.exp(-1.28 * sigma))} a {_numero(semana * np.exp(1.28 * sigma))}); "
                      f"{'+' if semana >= base else ''}{_numero((semana / base - 1) * 100, 0)} % frente a un clima normal "
                      # Con «22,5 °C y 10,0 mm pronosticados, contra 22,2 °C y 12,3 mm», el LLM leyó 22,2 como una mínima
                      f"para la época. Pronóstico de la semana: temperatura media {_numero(c['temperatura_media_c'], 1)} °C, "
                      f"lluvia total {_numero(c['lluvia_mm'], 1)} mm. Clima normal para la época: temperatura media "
                      f"{_numero(normal['temperatura_media_c'], 1)} °C, lluvia total {_numero(normal['lluvia_mm'], 1)} mm")
        # En la consulta 9, Patagonia daba +431 % con 24,5 °C pronosticados, cuando en los datos nunca pasó de 21,9 °C
        t_min, t_max = RANGO_TEMPERATURA[reg]
        if not t_min <= c["temperatura_media_c"] <= t_max:
            lineas.append(f"  Ojo: {_numero(c['temperatura_media_c'], 1)} °C está fuera de lo que registró {reg} en los datos "
                          f"({_numero(t_min, 1)} a {_numero(t_max, 1)} °C): esta estimación extrapola y no es confiable: no la uses para elegir región.")
    fuentes = sorted({p["fuente"] for _, _, _, p, _ in filas})
    lineas.append("Fuente del pronóstico: " + " | ".join(fuentes) + ".")
    return "\n".join(lineas)
