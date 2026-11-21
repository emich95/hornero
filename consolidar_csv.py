"""Arma el contenidos_aprobados.csv de la entrega: las 8 filas originales más las piezas aprobadas en las
dos etapas, en orden de fecha. Cada etapa guarda sus piezas en su propia carpeta (datos/etapa1/), porque
cada una corre por separado; este script las junta. Correr desde esta carpeta:
    python consolidar_csv.py
"""

import csv
from pathlib import Path

AQUI = Path(__file__).parent
ORIGINALES = 8
SALIDA = AQUI / "contenidos_aprobados.csv"


def leer(etapa: str) -> tuple[list[str], list[list[str]]]:
    with open(AQUI / etapa / "datos" / "etapa1" / "contenidos_aprobados.csv", newline="", encoding="utf-8") as f:
        filas = list(csv.reader(f))
    return filas[0], filas[1:]


encabezado, etapa1 = leer("etapa1")
encabezado_2, etapa2 = leer("etapa2")
assert encabezado == encabezado_2 == ["fecha", "canal", "segmento", "titulo", "texto"], "Cambió el encabezado"
assert etapa1[:ORIGINALES] == etapa2[:ORIGINALES], "Las 8 filas originales no coinciden entre las dos etapas"

piezas, vistas = [], set()
for fila in etapa1[ORIGINALES:] + etapa2[ORIGINALES:]:
    if tuple(fila) not in vistas:
        vistas.add(tuple(fila))
        piezas.append(fila)
piezas.sort(key=lambda fila: fila[0])

with open(SALIDA, "w", newline="", encoding="utf-8") as f:   # mismo formato que el original: fin de fila \r\n
    csv.writer(f).writerows([encabezado] + etapa1[:ORIGINALES] + piezas)

print(f"{SALIDA.name}: {ORIGINALES} originales + {len(etapa1) - ORIGINALES} piezas de la etapa 1 "
      f"+ {len(etapa2) - ORIGINALES} de la etapa 2 = {ORIGINALES + len(piezas)} filas")
