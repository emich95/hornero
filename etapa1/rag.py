"""
rag.py: índice vectorial (FAISS) sobre un PDF, guardado en disco.

La primera vez se arma el índice y se guarda en indices/: los vectores en <nombre>.faiss y los
chunks en <nombre>.json. Las veces siguientes se levanta de disco. Si el PDF cambia (se compara su
hash MD5) o cambia el modelo de embeddings, el índice se vuelve a armar solo.

Respecto de la clase 1 cambian dos cosas, porque los PDFs del TP son cortos y están ordenados en
secciones numeradas:
- Las tablas se extraen como tablas y se escriben fila por fila ("Canal: Email | Hashtags: no | ...").
  Con el texto plano de pdfplumber las columnas salen mezcladas y el LLM lee mal los límites.
- Los chunks se cortan por sección ("4. Reglas por canal", "1.2 Días de trabajo remoto") en vez de
  cada 500 caracteres, así una regla no queda partida en dos. Si una sección es muy larga, se parte
  con solapamiento.

No es una tool: lo usan las tools de tools.py.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
RUTA_INDICES = Path(__file__).parent / "indices"
MAX_CARACTERES = 1200
SOLAPAMIENTO = 200

TITULO_SECCION = re.compile(r"^\d+\.\s+[A-ZÁÉÍÓÚÑ]")            # "4. Reglas por canal"
TITULO_SUBSECCION = re.compile(r"^\d+\.\d+\s+[A-ZÁÉÍÓÚÑ]")      # "1.2 Días de trabajo remoto"
PIE_DE_PAGINA = re.compile(r"Página \d+\s*$")
EMPIEZA_ITEM = re.compile(r"^(•|-|\d+\s|\[ \])")

_modelo = None


def modelo_embeddings() -> SentenceTransformer:
    """Carga el modelo de embeddings una sola vez y lo comparten todos los índices."""
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    return _modelo


def _tabla_a_texto(filas: list[list]) -> str:
    filas = [[(celda or "").replace("\n", " ").strip() for celda in fila] for fila in filas]
    encabezado, resto = filas[0], filas[1:]
    return "\n".join("- " + " | ".join(f"{col}: {valor}" for col, valor in zip(encabezado, fila) if valor)
                     for fila in resto)


def extraer_lineas(ruta_pdf: Path) -> list[str]:
    """Texto del PDF línea por línea, con cada tabla reescrita fila por fila y sin pies de página."""
    bloques = []
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            desde = 0
            for tabla in sorted(pagina.find_tables(), key=lambda t: t.bbox[1]):
                _, arriba, _, abajo = tabla.bbox
                bloques.append(pagina.crop((0, desde, pagina.width, arriba)).extract_text() or "")
                bloques.append(_tabla_a_texto(tabla.extract()))
                desde = abajo
            bloques.append(pagina.crop((0, desde, pagina.width, pagina.height)).extract_text() or "")
    return [linea.strip() for bloque in bloques for linea in bloque.splitlines()
            if linea.strip() and not PIE_DE_PAGINA.search(linea)]


def _unir_renglones(lineas: list[str]) -> str:
    """Vuelve a unir los renglones cortados por el ancho de la página; respeta viñetas, párrafos y
    filas de tabla (las que empiezan con "- ")."""
    parrafos = []
    for linea in lineas:
        if (parrafos and not EMPIEZA_ITEM.match(linea) and not parrafos[-1].startswith("- ")
                and not re.search(r"[.:;!?)]$", parrafos[-1])):
            parrafos[-1] += " " + linea
        else:
            parrafos.append(linea)
    return "\n".join(parrafos)


def _partir(texto: str) -> list[str]:
    if len(texto) <= MAX_CARACTERES:
        return [texto]
    partes, actual = [], []
    for linea in texto.split("\n"):
        if actual and len("\n".join(actual + [linea])) > MAX_CARACTERES:
            partes.append("\n".join(actual))
            solapado = []
            while actual and len("\n".join(solapado)) < SOLAPAMIENTO:
                solapado.insert(0, actual.pop())
            actual = solapado
        actual.append(linea)
    partes.append("\n".join(actual))
    return partes


def partir_en_chunks(lineas: list[str]) -> list[dict]:
    """Un chunk por sección o subsección. Cada chunk lleva el título adelante, para que el embedding
    sepa de qué trata y el LLM pueda citar la sección."""
    secciones, seccion, titulo, cuerpo = [], "", "Introducción", []
    for linea in lineas:
        if TITULO_SECCION.match(linea) or TITULO_SUBSECCION.match(linea):
            if cuerpo:
                secciones.append((titulo, cuerpo))
            if TITULO_SECCION.match(linea):
                seccion = titulo = linea
            else:
                titulo = f"{seccion} > {linea}"
            cuerpo = []
        else:
            cuerpo.append(linea)
    if cuerpo:
        secciones.append((titulo, cuerpo))
    return [{"seccion": titulo, "texto": f"[{titulo}]\n{parte}"}
            for titulo, cuerpo in secciones for parte in _partir(_unir_renglones(cuerpo))]


class IndicePDF:
    """Índice FAISS sobre un PDF. Se levanta de disco si existe y el PDF no cambió; si no, se arma."""

    def __init__(self, ruta_pdf: Path, nombre: str, k: int = 3):
        self.ruta_pdf, self.nombre, self.k = Path(ruta_pdf), nombre, k
        self.ruta_faiss = RUTA_INDICES / f"{nombre}.faiss"
        self.ruta_chunks = RUTA_INDICES / f"{nombre}.json"
        self.chunks, self.indice, self.origen = self._levantar_o_armar()

    def _levantar_o_armar(self):
        md5 = hashlib.md5(self.ruta_pdf.read_bytes()).hexdigest()
        if self.ruta_faiss.exists() and self.ruta_chunks.exists():
            guardado = json.loads(self.ruta_chunks.read_text(encoding="utf-8"))
            if guardado.get("md5_pdf") == md5 and guardado.get("modelo") == MODELO_EMBEDDINGS:
                return guardado["chunks"], faiss.read_index(str(self.ruta_faiss)), "levantado de disco"
        chunks = partir_en_chunks(extraer_lineas(self.ruta_pdf))
        vectores = modelo_embeddings().encode([c["texto"] for c in chunks], normalize_embeddings=True)
        indice = faiss.IndexFlatIP(vectores.shape[1])          # producto interno = coseno (vectores normalizados)
        indice.add(np.asarray(vectores, dtype="float32"))
        RUTA_INDICES.mkdir(exist_ok=True)
        faiss.write_index(indice, str(self.ruta_faiss))
        self.ruta_chunks.write_text(json.dumps(
            {"pdf": self.ruta_pdf.name, "md5_pdf": md5, "modelo": MODELO_EMBEDDINGS,
             "armado": datetime.now().isoformat(timespec="seconds"), "chunks": chunks},
            ensure_ascii=False, indent=1), encoding="utf-8")
        return chunks, indice, "armado ahora"

    def buscar(self, consulta: str, k: int | None = None) -> list[dict]:
        """Los k chunks más parecidos a la consulta, del más al menos parecido."""
        vector = modelo_embeddings().encode([consulta], normalize_embeddings=True)
        puntajes, posiciones = self.indice.search(np.asarray(vector, dtype="float32"), k or self.k)
        return [{**self.chunks[i], "puntaje": round(float(p), 3)}
                for p, i in zip(puntajes[0], posiciones[0]) if i >= 0]
