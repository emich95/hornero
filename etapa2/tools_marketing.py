"""
tools.py: las tools de la etapa 1 (Marketing).

Tools que usan los agentes (el LLM decide cuándo llamarlas):
- fecha_de_hoy: la fecha de hoy (sirve para ubicar la pieza en la temporada).
- buscar_guia: RAG sobre guia_de_marca.pdf (tono, reglas de cada canal, legales, segmentos).
- leer_historial: lo último que se le comunicó a un segmento (lectura de contenidos_aprobados.csv).

Tool que llama el código al final del flujo, solo si la pieza quedó aprobada:
- guardar_pieza: agrega la pieza como fila nueva de contenidos_aprobados.csv (escritura).
  No se la damos a ningún agente: así un LLM no puede guardar una pieza que el revisor no aprobó.

Funciones comunes, sin LLM, que usan los nodos del grafo (agentes.py):
- fragmentos_guia y ejemplos_del_canal: arman lo que recibe el redactor.
- armar_texto_final: arma el texto publicable con el formato del historial.
- validar_pieza: los chequeos de la guía que se pueden hacer con código (largos, emojis, hashtags,
  palabras prohibidas, plazos de entrega, legales, datos que no estaban en el pedido).
"""

import csv
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain_core.tools import tool

from rag import IndicePDF

# Carpeta con los datos del TP (datos/etapa1 y datos/etapa2)
RUTA_DATOS = Path(__file__).parent / "datos"
RUTA_GUIA = RUTA_DATOS / "etapa1" / "guia_de_marca.pdf"
RUTA_HISTORIAL = RUTA_DATOS / "etapa1" / "contenidos_aprobados.csv"

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

CANALES = ["Instagram", "Email", "WhatsApp", "Push", "LinkedIn"]
SEGMENTOS = ["Nuevos", "Recurrentes", "Premium", "Todos"]
SINONIMOS = {"ig": "Instagram", "insta": "Instagram", "mail": "Email", "e-mail": "Email",
             "correo": "Email", "wsp": "WhatsApp", "wpp": "WhatsApp", "whats app": "WhatsApp",
             "notificacion push": "Push", "notificacion": "Push", "nuevo": "Nuevos",
             "recurrente": "Recurrentes", "todo": "Todos", "general": "Todos"}

# Índice de la guía: se levanta de disco si ya existe (indices/guia_de_marca.*); si no, se arma.
indice_guia = IndicePDF(RUTA_GUIA, "guia_de_marca", k=3)


def _plano(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar texto libre."""
    return unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode().strip()


def normalizar(valor: str, opciones: list[str]) -> str | None:
    """Lleva un canal o segmento escrito libremente a su nombre oficial ('mail' -> 'Email'). None si no existe."""
    plano = _plano(valor or "")
    plano = _plano(SINONIMOS.get(plano, plano))
    return next((o for o in opciones if _plano(o) == plano or _plano(o).rstrip("s") == plano), None)


def _historial() -> pd.DataFrame:
    return pd.read_csv(RUTA_HISTORIAL).sort_values("fecha", ascending=False)


# ==========================================================================
# Tools de los agentes
# ==========================================================================

@tool
def fecha_de_hoy(formato: Literal["numerico", "texto"]) -> str:
    """Devuelve la fecha de hoy. formato='numerico' devuelve AAAA-MM-DD;
    formato='texto' devuelve, por ejemplo, 'sábado 7 de noviembre de 2026'."""
    hoy = date.today()
    if formato == "texto":
        return f"{DIAS[hoy.weekday()]} {hoy.day} de {MESES[hoy.month - 1]} de {hoy.year}"
    return hoy.isoformat()


@tool
def buscar_guia(consulta: str) -> str:
    """Busca en la guía de marca de Hornero las secciones que responden a la consulta: voz y tono,
    palabras prohibidas, reglas de cada canal (largo, hashtags, emojis), plazos de entrega, legales
    obligatorios de las promociones, cómo hablarle a cada segmento y recuperación de detractores.
    Devuelve las 3 secciones más parecidas, cada una con su título entre corchetes."""
    if not consulta or not consulta.strip():
        return "La consulta está vacía. Escribí qué regla de la guía necesitás."
    return "\n\n".join(r["texto"] for r in indice_guia.buscar(consulta))


@tool
def leer_historial(segmento: str, cantidad: int = 5) -> str:
    """Lee las últimas piezas aprobadas que recibió un segmento de clientes, para no repetirle el
    mismo mensaje. segmento: Nuevos, Recurrentes, Premium o Todos (también trae las piezas que se
    mandaron a 'Todos', porque esas le llegaron a cada segmento). cantidad: cuántas piezas devolver,
    de la más nueva a la más vieja (máximo 10)."""
    nombre = normalizar(segmento, SEGMENTOS)
    if nombre is None:
        return f"El segmento '{segmento}' no existe. Usá uno de estos: {', '.join(SEGMENTOS)}."
    piezas = _historial()
    if nombre != "Todos":
        piezas = piezas[piezas["segmento"].isin([nombre, "Todos"])]
    piezas = piezas.head(max(1, min(int(cantidad), 10)))
    if piezas.empty:
        return f"No hay piezas aprobadas para el segmento {nombre}."
    lineas = [f"Últimas {len(piezas)} piezas que recibió el segmento {nombre}:"]
    for _, p in piezas.iterrows():
        resumen = " ".join(str(p["texto"]).split())
        resumen = resumen if len(resumen) <= 220 else resumen[:217] + "..."
        lineas.append(f"- {p['fecha']} | {p['canal']} | {p['segmento']} | {p['titulo']}\n  {resumen}")
    return "\n".join(lineas)


# ==========================================================================
# Tool de escritura (la llama el código, no un agente)
# ==========================================================================

@tool
def guardar_pieza(canal: str, segmento: str, titulo: str, texto: str) -> str:
    """Agrega una pieza aprobada por el revisor como fila nueva de contenidos_aprobados.csv, con la
    fecha y hora de ahora. texto: el texto publicable completo, con el formato del historial."""
    canal_ok, segmento_ok = normalizar(canal, CANALES), normalizar(segmento, SEGMENTOS)
    if canal_ok is None or segmento_ok is None:
        return f"No se guardó: canal '{canal}' o segmento '{segmento}' inválido."
    if not titulo.strip() or not texto.strip():
        return "No se guardó: el título y el texto no pueden estar vacíos."
    previas = _historial()
    if ((previas["canal"] == canal_ok) & (previas["texto"].str.strip() == texto.strip())).any():
        return "No se guardó: esa misma pieza ya estaba en el historial."
    # El archivo original usa fin de línea \r\n: lo respetamos para que todas las filas queden iguales
    if not RUTA_HISTORIAL.read_bytes().endswith(b"\n"):
        with open(RUTA_HISTORIAL, "ab") as f:
            f.write(b"\r\n")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(RUTA_HISTORIAL, "a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\r\n").writerow([fecha, canal_ok, segmento_ok, titulo.strip(), texto.strip()])
    return f"Pieza guardada en contenidos_aprobados.csv: {fecha} | {canal_ok} | {segmento_ok} | {titulo.strip()}"


# ==========================================================================
# Funciones comunes que usan los nodos del grafo
# ==========================================================================

def fragmentos_guia(consultas: list[str]) -> list[dict]:
    """La sección de la guía más parecida a cada consulta, sin repetir y en el orden de la guía."""
    elegidas = {}
    for consulta in consultas:
        mejor = indice_guia.buscar(consulta, k=1)[0]
        elegidas[mejor["seccion"]] = mejor
    orden = [c["seccion"] for c in indice_guia.chunks]
    return sorted(elegidas.values(), key=lambda c: orden.index(c["seccion"]))


def ejemplos_del_canal(canal: str, segmento: str, cantidad: int = 2) -> list[dict]:
    """Piezas aprobadas del mismo canal, para que el redactor copie el formato. Solo de otros segmentos:
    con una pieza del mismo segmento como ejemplo, el modelo tiende a copiarla. Si no hay, una sola
    del mismo segmento."""
    piezas = _historial()
    piezas = piezas[piezas["canal"] == canal]
    otros = piezas[piezas["segmento"] != segmento]
    return (otros.head(cantidad) if not otros.empty else piezas.head(1)).to_dict("records")


REGLAS_CANAL = {   # guía de marca, sección 4
    "Instagram": {"max_caracteres": 2200, "ideal": (150, 300), "hashtags": (3, 5), "max_emojis": 3},
    "Email": {"max_encabezado": 50, "max_palabras": 150, "hashtags": (0, 0), "max_emojis": 1},
    "WhatsApp": {"max_caracteres": 300, "hashtags": (0, 0), "max_emojis": 2},
    "Push": {"max_encabezado": 40, "max_caracteres": 90, "hashtags": (0, 0), "max_emojis": 1},
    "LinkedIn": {"max_caracteres": 1300, "hashtags": (2, 3), "max_emojis": 0},
}
LEGALES_WHATSAPP = "Bases y condiciones: hornero.com.ar/legales"     # el link que ya usa el historial

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿⌀-⏿]")
CTA_INSTAGRAM = re.compile(r"link en la bio|encontral[oa]s? en la app")
SALUDO_EMAIL = re.compile(r"^\s*hola,?\s*\{nombre\}\s*[:,.!]?\s*", re.IGNORECASE)

PROHIBIDAS = [   # guía, sección 3 y 8 (se buscan sobre el texto en minúsculas y sin tildes)
    (r"mas barat[oa]s? del mercado", "«el más barato del mercado»"),
    (r"precios? imbatibles?", "«precios imbatibles»"),
    (r"mejor precio garantizado", "«el mejor precio garantizado»"),
    (r"\bbarat[oa]s?\b", "«barato» (la guía prefiere «a precio justo» o «accesible»)"),
    (r"\bgratis\b", "«gratis» (usá «sin cargo» con legales)"),
    (r"100\s?% garantizado|\bgarantizad[oa]s?\b", "«garantizado»"),
    (r"\bsin riesgos?\b", "«sin riesgo»"),
    (r"\b(cositas?|precitos?|ofertitas?|descuentitos?)\b", "diminutivo condescendiente"),
    (r"lamentamos las molestias", "«lamentamos las molestias» (suena a plantilla)"),
    (r"cambios? faciles?", "«cambios fáciles» (la postventa no lo garantiza)"),
]
PLAZOS = [r"\b\d+\s*(h|hs|horas|dias)\b", r"\bllega(n|s)? (manana|hoy)\b", r"\b(envio|entrega)s? inmediat[oa]s?\b",
          r"\ben el dia\b", r"\bmismo dia\b", r"\bexpress\b", r"\ben (pocas|unas) horas\b"]
VELOCIDAD = r"\brapid|\bvelocidad|\bvolando\b"
COMPETIDORES = ["mercado libre", "mercadolibre", "fravega", "garbarino", "musimundo", "coto", "carrefour",
                "easy", "sodimac", "falabella", "amazon", "cetrogar", "megatone", "naldo", "on city"]
# Formas de tuteo y de "usted". Se buscan sobre el texto CON tildes: "disfrutá" (voseo) y "disfruta" (tuteo)
# solo se distinguen por la tilde. El LLM revisor confundía la primera persona ("te regalamos") con falta de
# voseo y dejaba pasar "vuelve": por eso este chequeo lo hace el código.
# El redactor no corregía el tuteo aunque la corrección le nombrara la palabra (devolvía el mismo texto):
# el código lo pasa a voseo con esta tabla. «usted» no tiene reemplazo y sigue siendo un error
VOSEO = {"tienes": "tenés", "puedes": "podés", "quieres": "querés", "sabes": "sabés", "eres": "sos", "vienes": "venís",
         "necesitas": "necesitás", "descubre": "descubrí", "aprovecha": "aprovechá", "elige": "elegí",
         "disfruta": "disfrutá", "ingresa": "ingresá", "mantén": "mantené", "obtén": "obtené", "haz": "hacé",
         "vuelve": "volvé", "mira": "mirá", "prepárate": "preparate", "anímate": "animate", "regístrate": "registrate",
         "suscríbete": "suscribite", "ahorra": "ahorrá", "renueva": "renová", "equipa": "equipá", "abre": "abrí",
         "conoce": "conocé", "visita": "visitá", "transforma": "transformá", "llévate": "llevate", "pide": "pedí",
         "crea": "creá", "explora": "explorá", "encuentra": "encontrá", "accede": "accedé"}
# Estas coinciden con la tercera persona del presente («Visita la app» / «el aire ahorra energía»): solo cuentan
# como tuteo cuando abren una oración, que es donde va el imperativo. En la corrida final del 19/11 el código
# convirtió «Aire acondicionado que ahorra energía» en «que ahorrá energía»
AMBIGUAS = {"descubre", "aprovecha", "elige", "disfruta", "ingresa", "vuelve", "mira", "ahorra", "renueva", "equipa",
            "abre", "conoce", "visita", "transforma", "pide", "crea", "explora", "encuentra", "accede"}
USTED = r"(?<![\wáéíóúñ])usted(?![\wáéíóúñ])"
SIGLAS_PERMITIDAS = {"CFTEA", "AMBA", "CABA"}
DATOS_A_CONFIRMAR = {r"\d+(?:[.,]\d+)?\s?%": "[X]%", r"\$\s?\d[\d.,]*": "$X",      # patrón -> marcador
                     r"\b\d{1,2}/\d{1,2}\b": "DD/MM", r"\b\d+\s+cuotas\b": "[X] cuotas",
                     # Códigos de cupón (mayúsculas con letras y números): en la corrida oficial apareció «NUEVO15»
                     r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b": "[CÓDIGO]"}
CONTACTOS = [r"[\w.+-]+@[\w-]+\.[\w.]+",                                      # mails
             r"(?<![\w.])(?:www\.)?(?!hornero\.com\.ar)[\w-]+\.(?:com|net|org|ar)(?:\.ar)?(?:/\S*)?",   # otros dominios
             r"(?<!\d)(?:\+?\d[\s-]?){8,}(?!\d)"]                                # teléfonos
MAX_PARECIDO = 0.30   # parte de la pieza que puede repetir de una pieza ya enviada (secuencias de 4 palabras)


def _confirmado(dato: str, pedido: str) -> bool:
    return dato.replace(" ", "") == "0%" or re.sub(r"\s", "", _plano(dato)) in re.sub(r"\s", "", _plano(pedido))


def quitar_datos_no_confirmados(texto: str, pedido: str) -> tuple[str, list[str]]:
    """Reemplaza por su marcador cada porcentaje, monto, fecha o cantidad de cuotas que no esté en el
    pedido del usuario. Devuelve el texto limpio y la lista de reemplazos hechos."""
    reemplazos = []
    for patron, marcador in DATOS_A_CONFIRMAR.items():
        def cambiar(m):
            if _confirmado(m.group(0), pedido):
                return m.group(0)
            reemplazos.append(f"{m.group(0)} → {marcador}")
            return marcador
        texto = re.sub(patron, cambiar, texto)
    return texto, reemplazos


def _palabras(texto: str) -> list[str]:
    return re.findall(r"[\wáéíóúüñÁÉÍÓÚÜÑ%]+", re.sub(r"\{nombre\}|^\s*hola,?", " ", texto, flags=re.IGNORECASE))


def _tejas(palabras: list[str], n: int = 4) -> set:
    planas = [_plano(p) for p in palabras]
    return {tuple(planas[i:i + n]) for i in range(len(planas) - n + 1)}


def pieza_mas_parecida(texto: str, n: int = 4) -> tuple[float, str, list[str]]:
    """La pieza del historial que más se parece al texto: qué parte de sus secuencias de 4 palabras ya
    aparece en ella, su título y las frases repetidas (para decirle al redactor qué cambiar)."""
    palabras = _palabras(texto)
    nuevas = _tejas(palabras, n)
    if not nuevas:
        return 0.0, "", []
    parecido, titulo, viejas = max(((len(nuevas & _tejas(_palabras(str(p["texto"])), n)) / len(nuevas),
                                     f"{p['titulo']} ({p['fecha'][:10]})", _tejas(_palabras(str(p["texto"])), n))
                                    for _, p in _historial().iterrows()), key=lambda x: x[0])
    repetida = [False] * len(palabras)
    for i in range(len(palabras) - n + 1):
        if tuple(_plano(p) for p in palabras[i:i + n]) in viejas:
            repetida[i:i + n] = [True] * n
    frases, actual = [], []
    for palabra, marca in zip(palabras + [""], repetida + [False]):
        if marca:
            actual.append(palabra)
        elif actual:
            frases.append(" ".join(actual))
            actual = []
    return parecido, titulo, sorted(frases, key=len, reverse=True)


def cuerpo_de_la_pieza(canal: str, pieza: dict) -> str:
    """El copy limpio: sin renglones vacíos (como el historial) y, en el email, sin el saludo, el asunto
    ni el botón, que el código pone en su lugar fijo aunque el redactor los haya metido en el cuerpo."""
    texto = re.sub(r"\n\s*\n+", "\n", (pieza.get("texto") or "").strip())
    if canal == "Email":
        texto = "\n".join(l for l in texto.split("\n") if not re.match(r"\s*(\[bot[oó]n|asunto:)", l, re.IGNORECASE))
        texto = SALUDO_EMAIL.sub("", texto.strip()).strip()
    return texto


def armar_texto_final(canal: str, pieza: dict) -> str:
    """Arma el texto publicable con el mismo formato que las piezas de contenidos_aprobados.csv.
    El saludo del email y el link de legales de WhatsApp los pone el código: son fijos por la guía."""
    texto = cuerpo_de_la_pieza(canal, pieza)
    legales = (pieza.get("legales") or "").strip()
    hashtags = " ".join(h if h.startswith("#") else f"#{h}" for h in (pieza.get("hashtags") or []))
    if canal == "Email":
        texto = "Hola, {nombre}:\n" + texto
        partes = [f"Asunto: {pieza.get('encabezado', '').strip()}", texto,
                  f"[Botón: {pieza.get('cta', '').strip()}]", legales]
    elif canal == "Push":
        partes = [f"Título: {pieza.get('encabezado', '').strip()}", f"Texto: {texto}"]
    elif canal == "WhatsApp":
        # Los legales van una sola vez, como link al final; el link de la categoría tiene que estar en el texto
        texto = re.sub(r"\s*·?\s*bases y condiciones:?\s*(hornero\.com\.ar/legales)?\.?", "", texto, flags=re.IGNORECASE).strip()
        link = re.search(r"hornero\.com\.ar/\w+", pieza.get("cta") or "")
        if link and not re.search(r"hornero\.com\.ar/(?!legales)\w+", texto):
            texto = f"{texto} {link.group(0)}"
        return f"{texto} · {LEGALES_WHATSAPP}" if legales else texto
    else:   # Instagram y LinkedIn: legales al final del texto y después los hashtags
        cta = (pieza.get("cta") or "").strip()
        if canal == "Instagram" and CTA_INSTAGRAM.search(_plano(cta)) and not CTA_INSTAGRAM.search(_plano(texto)):
            # En la ronda 2 quedó solo en el campo cta. Si la última oración ya mandaba a la app o a la bio con otras
            # palabras, se reemplaza: si no, quedaban dos llamados («Descubrí la colección en la app. Encontralo en la app.»)
            ultima = (_frases(texto) or [""])[-1]
            if re.search(r"\b(app|bio)\b", _plano(ultima)):
                texto = texto[:texto.rfind(ultima)].rstrip()
            texto = f"{texto} {cta[0].upper()}{cta[1:]}{'' if cta[-1] in '.!' else '.'}"
        partes = [texto, legales, hashtags]
    return "\n".join(p for p in partes if p)


def _como_acortar(largo: int, maximo: int) -> str:
    # Los LLM cuentan mal los caracteres: se les da también una meta en palabras (unos 6,5 caracteres por palabra)
    return f"sacale al menos {largo - maximo} caracteres, apuntá a {max(1, int(maximo / 6.5))} palabras o menos"


def pasar_a_voseo(pieza: dict) -> tuple[dict, list[str]]:
    """Reemplaza las formas de tuteo de la tabla VOSEO por su voseo en el encabezado, el texto y el CTA.
    Devuelve la pieza corregida y la lista de cambios («Visita → Visitá»)."""
    cambios = []
    def corregir(texto):
        for m in reversed(formas_de_tuteo(texto)):
            tuteo = m.group(0)
            voseo = VOSEO[tuteo.lower()]
            voseo = voseo[0].upper() + voseo[1:] if tuteo[0].isupper() else voseo
            cambios.insert(0, f"{tuteo} → {voseo}")
            texto = texto[:m.start()] + voseo + texto[m.end():]
        return texto
    corregida = {**pieza, **{campo: corregir(pieza.get(campo) or "") for campo in ("encabezado", "texto", "cta")}}
    return corregida, list(dict.fromkeys(cambios))


def formas_de_tuteo(texto: str) -> list[re.Match]:
    """Formas de la tabla VOSEO que hay en el texto. Las AMBIGUAS cuentan solo al principio de una oración."""
    patron = r"(?<![\wáéíóúñ])(" + "|".join(VOSEO) + r")(?![\wáéíóúñ])"
    return [m for m in re.finditer(patron, texto, flags=re.IGNORECASE)
            if m.group(0).lower() not in AMBIGUAS or re.search(r"(^|[.!?¡¿:;\n])\W*$", texto[:m.start()])]


def recortar_push(pieza: dict) -> str:
    """Si el texto del push se pasa de largo, devuelve sus primeras oraciones completas que entran en el máximo,
    salteando las que repiten el título ("" si ya entra o si ni la primera oración entra).
    El redactor no lograba acortarlo: devolvía el mismo texto."""
    texto, maximo = cuerpo_de_la_pieza("Push", pieza), REGLAS_CANAL["Push"]["max_caracteres"]
    if len(texto) <= maximo:
        return ""
    frases = _frases(texto)
    # En la consulta 4 quedó sola la oración que repetía el título y el push decía dos veces lo mismo
    sin_repetir = [f for f in frases if not _repite_el_titulo(f, pieza.get("encabezado") or "")]
    for opciones in (sin_repetir, frases):
        recortes = [" ".join(opciones[:i]) for i in range(1, len(opciones) + 1) if len(" ".join(opciones[:i])) <= maximo]
        if recortes:
            return recortes[-1]
    return ""


def _repite_el_titulo(frase: str, titulo: str) -> bool:
    """Al menos el 60 % de las palabras de 4 letras o más de la oración ya están en el título."""
    palabras = lambda s: set(re.findall(r"\w{4,}", _plano(s)))
    de_la_frase = palabras(frase)
    return bool(de_la_frase) and len(de_la_frase & palabras(titulo)) / len(de_la_frase) >= 0.6


def _frases(texto: str) -> list[str]:
    return [f for f in re.split(r"(?<=[.!?…])\s+|\n", texto) if f.strip()]


def validar_pieza(canal: str, pieza: dict, pedido: str = "", region: str = "") -> dict:
    """Chequeos de la guía que no necesitan un LLM. Devuelve {"errores", "avisos", "texto_final"}.
    Un error obliga a reescribir la pieza; un aviso se le pasa al revisor para que lo evalúe."""
    reglas = REGLAS_CANAL[canal]
    errores, avisos = [], []
    final = armar_texto_final(canal, pieza)
    encabezado = (pieza.get("encabezado") or "").strip()
    texto = cuerpo_de_la_pieza(canal, pieza)
    legales = (pieza.get("legales") or "").strip()
    hashtags = [h for h in (pieza.get("hashtags") or []) if h.strip()] + re.findall(r"#\w+", texto)
    # Todo menos los hashtags. En el push el llamado a la acción es tocar la notificación y no se publica:
    # la guía lo escribe «Abre la app» y el redactor lo copiaba, así que no se chequea
    escrito = "\n".join([encabezado, texto, legales] + ([] if canal == "Push" else [pieza.get("cta") or ""]))
    plano = _plano(escrito)

    # Reglas del canal (sección 4). El saludo con {nombre} es del email y del WhatsApp: un post es público
    if canal in ("Instagram", "LinkedIn") and "{nombre}" in escrito:
        errores.append(f"Un post de {canal} es público: sin saludo ni {{nombre}}, que son del email y del WhatsApp.")
    minimo, maximo = reglas["hashtags"]
    if not minimo <= len(hashtags) <= maximo:
        errores.append(f"{canal} lleva {'ningún hashtag' if maximo == 0 else f'de {minimo} a {maximo} hashtags'} "
                       f"y la pieza tiene {len(hashtags)}.")
    if canal == "Instagram" and "#horneroencasa" not in [h.lower() for h in hashtags]:
        errores.append("En Instagram los hashtags siempre incluyen #HorneroEnCasa.")
    emojis = len(EMOJI.findall(escrito))
    if emojis > min(reglas["max_emojis"], 3):
        errores.append(f"{canal} admite hasta {reglas['max_emojis']} emojis y la pieza tiene {emojis}.")
    if "max_caracteres" in reglas:
        medido = texto if canal == "Push" else final
        if len(medido) > reglas["max_caracteres"]:
            errores.append(f"El {'texto del push' if canal == 'Push' else 'texto'} tiene {len(medido)} caracteres; "
                           f"el máximo de {canal} es {reglas['max_caracteres']}: {_como_acortar(len(medido), reglas['max_caracteres'])}.")
    if "max_encabezado" in reglas:
        nombre = "asunto" if canal == "Email" else "título del push"
        if not encabezado:
            errores.append(f"Falta el {nombre}: va en el campo encabezado, de {7 if canal == 'Email' else 5} palabras o menos.")
        elif len(encabezado) > reglas["max_encabezado"]:
            errores.append(f"El {nombre} tiene {len(encabezado)} caracteres; el máximo es {reglas['max_encabezado']}: "
                           f"{_como_acortar(len(encabezado), reglas['max_encabezado'])}.")
    if canal == "Email":
        palabras = len(re.findall(r"\S+", texto)) + 2          # + "Hola, {nombre}:"
        if palabras > reglas["max_palabras"]:
            errores.append(f"El cuerpo del email tiene {palabras} palabras; el máximo es 150.")
        if EMOJI.search(texto + legales):
            errores.append("En el email el único emoji permitido va en el asunto.")
        if not (pieza.get("cta") or "").strip():
            errores.append("El email necesita el texto de su único botón (cta).")
    if canal == "Instagram":
        if not CTA_INSTAGRAM.search(_plano(final)):
            errores.append("En Instagram el llamado a la acción es «Link en la bio» o «Encontralo en la app».")
        largo, (ideal_min, ideal_max) = len(texto), reglas["ideal"]
        if not ideal_min <= largo <= ideal_max:
            avisos.append(f"El copy tiene {largo} caracteres; lo ideal en Instagram es de {ideal_min} a {ideal_max}.")
    if canal == "WhatsApp" and not re.search(r"hornero\.com\.ar/(?!legales)\w+", final):
        errores.append("En WhatsApp el llamado a la acción es un link corto (por ejemplo, hornero.com.ar/cocina).")

    # Palabras prohibidas, plazos de entrega y competidores (secciones 3, 5 y 8)
    errores += [f"Usa una expresión prohibida: {nombre}." for patron, nombre in PROHIBIDAS if re.search(patron, plano)]
    if any(re.search(p, plano) for p in PLAZOS):
        errores.append("Promete un plazo de entrega concreto (la guía lo prohíbe en toda pieza).")
    if region in ("Cuyo", "NOA", "Centro") and re.search(VELOCIDAD, plano):
        errores.append(f"La pieza va a clientes de {region}: no se menciona la velocidad de entrega.")
    errores += [f"Menciona a un competidor por su nombre ({c})." for c in COMPETIDORES if re.search(rf"\b{c}\b", plano)]

    # Voz y tono (sección 2)
    sin_links = re.sub(r"#\w+|\S+\.com\S*|\{nombre\}", "", escrito)
    gritos = [p for p in re.findall(r"\b[A-ZÁÉÍÓÚÑ]{4,}\b", sin_links) if p not in SIGLAS_PERMITIDAS]
    if gritos:
        errores.append(f"Tiene mayúsculas sostenidas: {', '.join(gritos)}.")
    if re.search(r"!!|¡¡", escrito):
        errores.append("Tiene signos de exclamación repetidos (la guía pide entusiasmo sin gritar).")
    for frase in _frases(texto + "\n" + legales):
        n = len(re.findall(r"\S+", frase))
        if n >= 25:
            errores.append(f"Frase de {n} palabras (la guía pide menos de 25): «{frase[:60]}...».")
    tuteo = sorted({m.group(0).lower() for m in formas_de_tuteo(escrito)} | set(re.findall(USTED, escrito.lower())))
    if tuteo:
        errores.append(f"Tiene tuteo o trato de usted ({', '.join(tuteo)}): la guía pide voseo (tenés, elegí, volvé, mirá).")

    # Legales obligatorios (sección 6): van una vez, en su lugar, no dentro del cuerpo
    if re.search(r"no acumulable|hasta agotar stock|cftea|consulta zonas", _plano(texto)):
        errores.append("Los legales están metidos en el cuerpo: van solo en el campo legales, una vez.")
    promo = _plano(encabezado + " " + texto)
    tipos = {"descuento": re.search(r"\d+\s?%|\[x\]\s?%|descuento|cupon", promo),
             "cuotas": re.search(r"\bcuotas\b", promo), "envío sin cargo": re.search(r"sin cargo|envio gratis", promo)}
    tipos = [t for t, hay in tipos.items() if hay]
    if tipos and canal == "WhatsApp" and not legales:
        errores.append(f"Hay {', '.join(tipos)}: en WhatsApp los legales van como link a «Bases y condiciones».")
    elif tipos and canal == "Push":
        avisos.append(f"El push menciona {', '.join(tipos)}: no entran legales, tienen que estar en la app.")
    elif tipos and canal != "WhatsApp":
        exigido = {"descuento": "no acumulable", "cuotas": "cftea", "envío sin cargo": "consulta zonas"}
        faltan = [t for t in tipos if exigido[t] not in _plano(legales)]
        if faltan:
            errores.append(f"Hay {', '.join(faltan)} y faltan los legales de la guía (sección 6) para eso.")

    # Datos que no estaban en el pedido: van como marcador ([X]%, DD/MM, $X), nunca inventados
    for patron, marcador in DATOS_A_CONFIRMAR.items():
        for dato in re.findall(patron, encabezado + " " + texto + " " + legales):
            if not _confirmado(dato, pedido):
                errores.append(f"«{dato}» no está en el pedido: usá el marcador {marcador} o sacalo.")
    # Si el pedido da el beneficio y no habla de cupón, decir que se usa con cupón es inventar el mecanismo:
    # en la corrida final del 19/11 el WhatsApp pedía «Usá el cupón al pagar» (en la primera había inventado «NUEVO15»)
    if (re.search(r"\bcupon|\bcodigo", plano) and re.search(r"\d\s?%|cuotas|sin cargo|envio gratis", _plano(pedido))
            and not re.search(r"cupon|codigo", _plano(pedido))):
        errores.append("El pedido da el beneficio pero no dice que se use con cupón ni con código: sacá esa mención, "
                       "no inventes cómo se aplica.")

    # Mails, teléfonos o sitios que nadie dio: un canal de contacto inventado es un dato inventado
    for patron in CONTACTOS:
        for dato in re.findall(patron, escrito):
            if _plano(dato) not in _plano(pedido):
                errores.append(f"«{dato.strip()}» es un dato de contacto que no está en el pedido: para el contacto "
                               "directo usá «respondé este email» o la app, sin inventar direcciones ni teléfonos.")

    # No repetir lo que ya se mandó (el historial): mismo mensaje con otras palabras sí, mismas frases no
    parecido, titulo, frases = pieza_mas_parecida(texto)
    if parecido > MAX_PARECIDO:
        errores.append(f"Repite frases de una pieza ya enviada, «{titulo}» ({parecido:.0%} de coincidencia). "
                       f"Reescribí con otras palabras y otro enfoque estas frases: "
                       + "; ".join(f"«{f}»" for f in frases[:5]) + ".")
    return {"errores": list(dict.fromkeys(errores)), "avisos": avisos, "texto_final": final}
