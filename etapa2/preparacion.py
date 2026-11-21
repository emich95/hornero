"""
preparacion.py: preparación de datos compartida entre el entrenamiento (entrenar_modelos.py)
y las tools (tools.py).

Tener una sola versión garantiza que, al predecir, el modelo reciba los datos armados exactamente
igual que cuando se entrenó. Si el preprocesamiento difiere, las predicciones salen mal sin dar error.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Attrition (empleados.csv)
# --------------------------------------------------------------------------
# Se excluyen id_empleado (no es una característica) y sueldo_bruto: depende del nivel, no mejora
# el AUC y la posición del sueldo frente al mercado ya está en sueldo_vs_mercado_pct.
CAT_ATTRITION = ["area", "centro", "nivel", "modalidad"]
NUM_ATTRITION = ["antiguedad_anios", "sueldo_vs_mercado_pct", "meses_sin_aumento", "horas_extra_mes",
                 "distancia_km", "ultima_evaluacion", "capacitaciones_ultimo_anio", "satisfaccion_clima"]


def objetivo_attrition(empleados: pd.DataFrame) -> pd.Series:
    """1 si el empleado renunció en el último año, 0 si no."""
    return (empleados["renuncio"] == "Si").astype(int)


# --------------------------------------------------------------------------
# Ventas (ventas.csv)
# --------------------------------------------------------------------------
CATEGORIAS = ["Aires acondicionados", "Calefacción", "Cocina", "Electro", "Jardín", "Muebles"]
REGIONES = ["AMBA", "Centro", "Cuyo", "Litoral", "NOA", "Patagonia"]
VARIABLES_VENTAS = ["temperatura_media_c", "lluvia_mm", "precio_cobre_usd_ton"]
COLUMNAS_VENTAS = ["categoria", "region"] + VARIABLES_VENTAS


def diseno_ventas(datos: pd.DataFrame) -> np.ndarray:
    """Matriz del modelo de ventas. Sobre log(unidades), cada combinación categoría-región tiene su
    nivel propio, y cada categoría tiene su propio efecto de la temperatura, la lluvia y el cobre
    (al calor los aires suben y la calefacción baja; el cobre encarece a unas categorías y no a otras)."""
    columnas = []
    for categoria in CATEGORIAS:
        es_categoria = (datos["categoria"] == categoria).to_numpy(dtype=float)
        for region in REGIONES:
            columnas.append(es_categoria * (datos["region"] == region).to_numpy(dtype=float))
        for variable in VARIABLES_VENTAS:
            columnas.append(es_categoria * datos[variable].to_numpy(dtype=float))
    return np.column_stack(columnas)


def nombres_diseno_ventas() -> list[str]:
    """Nombres de las columnas de diseno_ventas, en el mismo orden."""
    nombres = []
    for categoria in CATEGORIAS:
        nombres += [f"{categoria} | {region}" for region in REGIONES]
        nombres += [f"{categoria} x {variable}" for variable in VARIABLES_VENTAS]
    return nombres
