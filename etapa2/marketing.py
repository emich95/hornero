"""
agentes.py: el pipeline de Marketing de la etapa 1.

Tres agentes en secuencia, conectados con un StateGraph de LangGraph. Entre agente y agente hay
nodos de código (sin LLM) que deciden qué recibe cada uno:

    planificador ──► preparar_contexto ──► redactor ──► validar ──► revisor ──► guardar ──► FIN
    (agente+tools)       (código)          (agente)     (código)    (agente)    (código)
                                              ▲             │           │
                                              └─────────────┴───────────┘
                         corregir: hasta 4 versiones en total y 2 rechazos del revisor; si sigue mal, FIN sin guardar

- planificador: lee el historial del segmento (tool leer_historial) y, si lo necesita, la guía
  (tool buscar_guia). Devuelve el brief: canal, segmento, situación, mensaje, un ángulo distinto de
  lo último que recibió el segmento y los datos que faltan. Es el único agente que ve el historial.
- preparar_contexto (código): elige con el RAG las secciones de la guía que aplican al brief y dos
  piezas aprobadas del mismo canal como ejemplo de formato.
- redactor: escribe la pieza con el brief, esas secciones y los ejemplos. No tiene tools.
- validar (código): los chequeos de la guía que no necesitan un LLM. Si algo falla, la pieza vuelve
  al redactor con la lista de errores, sin gastar una llamada al revisor.
- revisor: recibe la pieza, el brief sin el ángulo (que nombra piezas del historial) y el checklist de
  la guía. Aprueba o devuelve observaciones. No tiene tools.
- guardar (código): llama a la tool guardar_pieza solo si la pieza quedó aprobada.

main.py solo usa la función responder().
"""

import json
import operator
import os
import re
from functools import wraps
from time import perf_counter
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, StructuredOutputError, ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tools_marketing import (CANALES, LEGALES_WHATSAPP, SEGMENTOS, buscar_guia, cuerpo_de_la_pieza, ejemplos_del_canal,
                   fecha_de_hoy, fragmentos_guia, guardar_pieza, leer_historial, normalizar,
                   pasar_a_voseo, quitar_datos_no_confirmados, recortar_push, validar_pieza)

load_dotenv()

MODELO = "openai/gpt-oss-20b"
# reasoning_effort="low": con el valor por defecto, gpt-oss-20b a veces razona hasta agotar los
# tokens de salida y devuelve una respuesta vacía. Además gasta muchos menos tokens.
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=MODELO, temperature=0.0, max_retries=5,
               reasoning_effort="low")
# El revisor razona un poco más ("medium"). En las pruebas, con "low" pedía cambios que rompían la pieza
# ("Sabés" en vez de "Sabemos" por una supuesta falta de voseo) y sugería agregar un mail de contacto
# inventado; con "medium" marcó solo lo que faltaba. El redactor también probó "medium": no corregía mejor y
# generaba hasta 7.000 tokens de salida por consulta, que con el cupo de 8.000 por minuto frenaban todo.
llm_cuidadoso = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=MODELO, temperature=0.0, max_retries=5,
                         reasoning_effort="medium")

MAX_VERSIONES = 4         # versiones del redactor en total (las correcciones del validador son baratas)
MAX_RECHAZOS = 2          # si el revisor rechaza dos veces, la pieza no se guarda (decisión D4)
REGIONES = ["AMBA", "Centro", "Cuyo", "Litoral", "NOA", "Patagonia"]
SITUACIONES = ["promocion", "lanzamiento", "recuperacion", "contenido"]
PROMO_EN_EL_PEDIDO = re.compile(r"\d\s?%|descuento|cuotas|sin cargo|env[ií]o gratis|cup[oó]n|oferta|promo", re.IGNORECASE)
EJEMPLOS = [              # las consultas de prueba de la consigna (main.py las muestra como botones)
    "Armá un post de Instagram para clientes Premium anunciando la nueva línea de muebles de exterior.",
    "Necesito un email para clientes que tuvieron problemas con una entrega, para recuperarlos.",
    "Escribí un mensaje de WhatsApp para clientes nuevos con un 15% de descuento en Cocina y guardalo.",
    "Proponé un push para clientes recurrentes sobre aires acondicionados que no repita lo último que les mandamos.",
]


# --------------------------------------------------------------------------
# Contador de llamadas y tokens (el de la clase 4)
# --------------------------------------------------------------------------

class ContadorUso(BaseCallbackHandler):
    # LangChain llama a on_llm_end cada vez que termina una llamada al LLM
    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.llamadas, self.rechazadas, self.entrada, self.salida = 0, 0, 0, 0

    def on_llm_error(self, error, **kwargs):
        # Groq rechazó la llamada (por ejemplo, una tool mal llamada): cuenta, pero no informa tokens
        self.llamadas += 1
        self.rechazadas += 1

    def on_llm_end(self, response, **kwargs):
        self.llamadas += 1
        for generaciones in response.generations:
            for g in generaciones:
                uso = getattr(g.message, "usage_metadata", None) or {}   # tokens que informa el proveedor
                self.entrada += uso.get("input_tokens", 0)
                self.salida += uso.get("output_tokens", 0)

    def resumen(self) -> str:
        miles = lambda n: f"{n:,}".replace(",", ".")
        n = self.rechazadas
        rechazadas = f" ({n} rechazada{'s' if n > 1 else ''} por Groq, sin tokens informados)" if n else ""
        return (f"{self.llamadas} llamada{'' if self.llamadas == 1 else 's'} al LLM{rechazadas} · "
                f"{miles(self.entrada)} tokens de entrada y {miles(self.salida)} de salida")


contador = ContadorUso()
llm.callbacks = [contador]             # planificador y redactor
llm_cuidadoso.callbacks = [contador]


# --------------------------------------------------------------------------
# Lo que devuelve cada agente (salida estructurada: el código no tiene que adivinar)
# --------------------------------------------------------------------------

class Brief(BaseModel):
    """Brief de la pieza para el redactor."""
    model_config = ConfigDict(extra="forbid")   # el modo estricto de Groq lo exige
    fuera_de_alcance: bool = Field(description="true si el pedido no es una pieza de marketing de texto de Hornero")
    motivo: str = Field(description="Si está fuera de alcance, por qué, en una oración. Si no, vacío")
    canal: str = Field(description="Instagram, Email, WhatsApp, Push o LinkedIn")
    segmento: str = Field(description="Nuevos, Recurrentes, Premium o Todos")
    situacion: str = Field(description="promocion (descuento, cuotas o envío sin cargo), lanzamiento (un producto, línea o colección nueva), recuperacion (clientes que tuvieron un problema) o contenido (tips y guías)")
    objetivo: str = Field(description="Qué tiene que lograr la pieza, en una oración")
    mensaje_principal: str = Field(description="La idea del único mensaje principal en pocas palabras, sin redactarla como copy (por ejemplo: 'descuento de bienvenida en Cocina'). Sin características del producto que el pedido no dice")
    angulo: str = Field(description="Qué la diferencia de las últimas piezas que recibió el segmento (nombrá cuáles)")
    llamado_a_la_accion: str = Field(description="La acción que se le pide al cliente, en pocas palabras (por ejemplo: 'entrar a la categoría Cocina')")
    datos_del_pedido: list[str] = Field(description="Datos concretos que da el pedido (producto, categoría, descuento, región). Solo lo que dice el pedido")
    datos_faltantes: list[str] = Field(description="Datos que la pieza necesita y el pedido no da (porcentaje, fechas, monto mínimo). Van como marcador: [X]%, DD/MM, $X")
    region: str = Field(description="Región a la que va dirigida si el pedido la dice (AMBA, Centro, Cuyo, Litoral, NOA o Patagonia); si no, vacío")


class Pieza(BaseModel):
    """Una pieza de marketing de texto lista para revisar."""
    model_config = ConfigDict(extra="forbid")   # el modo estricto de Groq lo exige
    idea: str = Field(description="La idea de la pieza en una oración")
    titulo: str = Field(description="Título interno corto para el historial, con mayúscula solo al principio, como 'Cuotas sin interés en Electro'")
    encabezado: str = Field(description="Email: el asunto, de 7 palabras o menos (máximo 50 caracteres). Push: el título, de 5 palabras o menos (máximo 40). Otros canales: vacío")
    texto: str = Field(description="El copy, sin hashtags ni legales. Email: el cuerpo, empezando con 'Hola, {nombre}:'. Instagram y LinkedIn: sin saludo, son posts públicos. WhatsApp: de 35 palabras o menos, con el link corto. Push: de 13 palabras o menos (máximo 90 caracteres)")
    cta: str = Field(description="El llamado a la acción. Email: el texto del único botón. Otros canales: la frase o el link de cierre que ya está en el texto")
    legales: str = Field(description="Los legales de la guía si hay descuento, cuotas o envío sin cargo, con marcadores si falta un dato. Vacío si no hay promoción")
    hashtags: list[str] = Field(description="Instagram: de 3 a 5, con #HorneroEnCasa. LinkedIn: 2 o 3. Otros canales: lista vacía")
    imagen_sugerida: str = Field(description="Una línea que describe la imagen que acompaña la pieza")


class Revision(BaseModel):
    """Revisión de la pieza contra el checklist de la guía, punto por punto. Aprueba el código: solo si todo es true."""
    model_config = ConfigDict(extra="forbid")   # el modo estricto de Groq lo exige
    un_mensaje_y_un_llamado: bool = Field(description="true si hay un solo mensaje principal y un solo llamado a la acción (en el email, el botón es ese llamado)")
    tono_de_la_situacion: bool = Field(description="true si el tono es el de la situación y el segmento. En recuperación de detractores, si cumple los 4 pasos: reconoce el problema sin excusas, cuenta qué se mejora sin prometer plazos, ofrece un beneficio con legales y cierra con un canal de contacto")
    sin_promesas_ni_datos_inventados: bool = Field(description="false si promete algo que no depende de Hornero ('cada envío será perfecto') o afirma algo que el brief no dice: características del producto (materiales, calidad, tecnología, garantía), un código de cupón o un paso de compra. Los marcadores [X]% y DD/MM son correctos")
    cumple_el_brief: bool = Field(description="true si respeta el canal, el segmento y la idea del mensaje principal del brief. No hace falta que use las mismas palabras del brief")
    observaciones: list[str] = Field(description="Por cada punto en false, qué cambiar y dónde. Lista vacía si todos son true")


PUNTOS_DEL_CHECKLIST = ["un_mensaje_y_un_llamado", "tono_de_la_situacion", "sin_promesas_ni_datos_inventados",
                        "cumple_el_brief"]


# --------------------------------------------------------------------------
# Los tres agentes
# --------------------------------------------------------------------------

PROMPT_PLANIFICADOR = """ROL: Sos el planificador de contenidos del equipo de Marketing de Hornero, un e-commerce \
argentino de artículos para el hogar y electro. Convertís el pedido del usuario en un brief para el redactor.

ALCANCE: solo piezas de marketing de texto para Instagram, Email, WhatsApp, Push o LinkedIn, dirigidas a los \
segmentos Nuevos, Recurrentes, Premium o Todos. Si el pedido es otra cosa (una consulta de RRHH o de ventas, una \
imagen o un video, algo ajeno a Hornero), marcá fuera_de_alcance y explicá el motivo.

CÓMO TRABAJÁS:
1. Llamá a leer_historial con el segmento del pedido. El brief tiene que proponer un ángulo distinto de esas \
piezas: otro enfoque, otro título y otras frases. Si ya hubo una pieza de la misma situación para el segmento, \
decí en el ángulo qué cambia (el foco, el beneficio o el cierre) y no uses sus frases en mensaje_principal.
2. Si te falta una regla (tono de la situación, cómo hablarle al segmento, legales), usá buscar_guia.
3. Si el pedido no dice el segmento, elegí el que mejor encaje. Si no dice el canal, usá el preferido del \
segmento según la guía.

REGLAS:
- Nunca inventes datos: precios, porcentajes, cuotas, plazos de entrega, fechas, códigos de cupón ni links. En \
datos_del_pedido va solo lo que dice el pedido; lo que la pieza necesita y no está va en datos_faltantes.
- No agregues características del producto (tecnología, materiales, calidad, garantía) ni beneficios que el \
pedido no menciona.
- El brief describe la pieza, no la escribe: el copy lo redacta otro agente.
- Un solo mensaje principal y un solo llamado a la acción.
- Respondé siempre con la tool brief, también cuando el pedido está fuera de alcance.

TONO: castellano rioplatense con voseo, concreto."""

PROMPT_REDACTOR = """ROL: Sos el redactor de Hornero. Escribís una sola pieza de texto a partir del brief, con la \
voz de la marca: cálida, clara y práctica, en voseo rioplatense.

ALCANCE: solo texto (idea, copy, hashtags, asunto o título, legales y una línea de imagen sugerida). Usá solo lo \
que está en el brief, en las secciones de la guía y en los ejemplos que te pasamos.

REGLAS:
- Nunca inventes precios, porcentajes, cuotas, plazos de entrega, fechas ni códigos. Si falta un dato, dejá un \
marcador: [X]%, DD/MM, $X.
- No afirmes características del producto que no están en el brief (materiales, medidas, garantía, origen): \
contá para qué sirve y qué beneficio de Hornero tiene.
- Cumplí las reglas del canal (largo, hashtags, emojis, llamado a la acción). Si hay descuento, cuotas o envío \
sin cargo, copiá tal cual los legales de la guía.
- Los links siguen el formato del historial: hornero.com.ar/<categoría>. Para un canal de contacto directo usá \
"respondé este email" o la app: nunca inventes mails, teléfonos ni sitios.
- Un solo mensaje principal y un solo llamado a la acción. Frases de menos de 25 palabras.
- Nada de mayúsculas sostenidas, "!!", palabras prohibidas ni plazos de entrega.
- Los ejemplos sirven para copiar el formato, no el contenido: no repitas sus frases.
- Si te pasan correcciones de una versión anterior, aplicalas todas sin romper lo que estaba bien.

TONO: el que la guía indica para la situación y el segmento del brief."""

PROMPT_REVISOR = """ROL: Sos el revisor de marca de Hornero. Controlás una pieza contra la guía de marca antes de \
que se publique.

ALCANCE: solo revisás, no reescribís. Recibís la pieza separada en partes, el brief y las secciones de la guía que \
aplican. El código ya controló voseo, largos, emojis, hashtags, palabras prohibidas, plazos de entrega, legales y \
datos inventados: no los vuelvas a contar.

CÓMO REVISÁS: respondé el checklist de la guía punto por punto, true si cumple y false si no. Marcá false solo \
cuando puedas señalar la frase concreta que no cumple, y escribí en observaciones qué cambiar y dónde. Los \
marcadores como [X]% o DD/MM son correctos: los completa el equipo antes de publicar. Los legales y el botón vienen \
en su propia línea. Nunca pidas agregar datos que no están en el brief (mails, teléfonos, links, porcentajes, \
fechas): si falta un canal de contacto, pedí "respondé este email" o la app. No marques false por gusto personal \
ni por cosas que la guía no pide. El canal y el segmento los eligió quien hizo el pedido: no los cuestiones.

TONO: directo y breve."""

# Groq no deja combinar el modo JSON con tools en la misma llamada: el planificador, que tiene tools,
# devuelve el brief como una llamada a una tool más (ToolStrategy). Redactor y revisor no tienen tools
# y usan la salida estructurada nativa en modo estricto: Groq restringe la generación al esquema (sin
# strict, gpt-oss-20b a veces devolvía el esquema JSON en vez de la pieza).
# ToolStrategy llama a la tool como la clase. gpt-oss-20b a veces la llamaba "brief" (así aparece en el prompt)
# y Groq rechazaba la llamada por no existir "brief": se nombra en minúsculas.
Brief.__name__ = "brief"
planificador = create_agent(model=llm, tools=[leer_historial, buscar_guia], system_prompt=PROMPT_PLANIFICADOR,
                            response_format=ToolStrategy(Brief))
redactor = create_agent(model=llm, tools=[], system_prompt=PROMPT_REDACTOR,
                        response_format=ProviderStrategy(Pieza, strict=True))
revisor = create_agent(model=llm_cuidadoso, tools=[], system_prompt=PROMPT_REVISOR,
                       response_format=ProviderStrategy(Revision, strict=True))


# --------------------------------------------------------------------------
# Estado y nodos del grafo
# --------------------------------------------------------------------------

class EstadoMarketing(TypedDict, total=False):
    pedido: str
    brief: dict | None
    contexto_guia: str
    ejemplos: str
    pieza: dict
    intentos: int
    rechazos: int
    validacion: dict
    revision: dict
    correcciones: list[str]
    guardado: str
    pasos: Annotated[list[dict], operator.add]


def con_tiempo(nodo):
    # Agrega al último paso de cada nodo cuántos segundos tardó
    @wraps(nodo)
    def envuelto(estado):
        inicio = perf_counter()
        salida = nodo(estado)
        if salida.get("pasos"):
            salida["pasos"][-1]["titulo"] += f" ({perf_counter() - inicio:.1f} s)"
        return salida
    return envuelto


def aviso_sin_cupo(error: Exception) -> str:
    """El plan gratis de Groq tiene cupo por minuto y por día (200.000 tokens, que se van recargando). El error
    dice cuál se agotó y cuánto falta; eso va al mensaje, porque «unos minutos» puede ser más de una hora."""
    texto = str(error)
    tope = " diario" if "per day" in texto else " por minuto" if "per minute" in texto else ""
    espera = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", texto)
    if espera and any(espera.groups()):
        minutos = int(espera.group(1) or 0) * 60 + int(espera.group(2) or 0) + (1 if float(espera.group(3) or 0) else 0)
        cuando = f"{minutos // 60} h {minutos % 60} min" if minutos >= 60 else f"{max(minutos, 1)} min"
        return f"se agotó el cupo{tope} del plan gratis de Groq. Se puede volver a probar en {cuando}."
    return f"se agotó el cupo{tope} del plan gratis de Groq. Probá de nuevo en unos minutos."


def es_falla_de_generacion(error: Exception) -> bool:
    # Salidas que no respetan el esquema o que Groq rechaza por mal generadas (tool inexistente, JSON vacío).
    # Volver a preguntar suele andar; un error de cupo o de red ya lo reintenta ChatGroq (max_retries).
    return isinstance(error, StructuredOutputError) or any(
        codigo in str(error) for codigo in ("tool_use_failed", "json_validate_failed"))


def rescatar_generacion(error: Exception, esquema: type[BaseModel]) -> dict | None:
    # Cuando Groq rechaza una llamada mal generada, devuelve en failed_generation lo que el modelo quiso
    # responder. En las pruebas, ante un pedido fuera de alcance, gpt-oss-20b escribió el brief correcto como
    # texto en vez de llamar a la tool, y con temperatura 0 los reintentos daban lo mismo. Si ese texto es un
    # JSON que valida contra el esquema, se usa; si no, se descarta.
    cuerpo = getattr(error, "body", None)
    if isinstance(cuerpo, dict):
        cuerpo = cuerpo.get("error", cuerpo)
    generado = cuerpo.get("failed_generation") if isinstance(cuerpo, dict) else None
    if not generado:
        return None
    try:
        datos = json.loads(generado)
        datos = datos.get("arguments", datos) if isinstance(datos, dict) else datos   # {"name": ..., "arguments": {...}}
        if isinstance(datos, dict) and datos.get("fuera_de_alcance") is True:
            # Ante un pedido fuera de alcance el modelo manda solo fuera_de_alcance y motivo, y Groq rechaza la
            # llamada por los campos que faltan. Para rechazar el pedido no hacen falta: van vacíos
            vacios = {nombre: [] if campo.annotation == list[str] else "" for nombre, campo in esquema.model_fields.items()}
            datos = {**vacios, **datos}
        return esquema.model_validate(datos).model_dump()
    except (ValueError, ValidationError):
        return None


def preguntar_estructurado(agente, mensaje: str, esquema: type[BaseModel], intentos: int = 3) -> tuple[dict, dict]:
    for intento in range(intentos):
        try:
            resultado = agente.invoke({"messages": [{"role": "user", "content": mensaje}]})
            return resultado["structured_response"].model_dump(), resultado
        except Exception as error:
            if not es_falla_de_generacion(error):
                raise
            rescatado = rescatar_generacion(error, esquema)
            if rescatado is not None:
                return rescatado, {"messages": [], "rescatado": True}
            if intento == intentos - 1:
                raise


def tools_llamadas(resultado) -> str:
    # Qué tools llamó un agente, con qué argumentos y el comienzo de lo que le devolvieron
    respuestas = {m.tool_call_id: m.content for m in resultado["messages"] if getattr(m, "type", "") == "tool"}
    lineas = []
    for msg in resultado["messages"]:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] in ("leer_historial", "buscar_guia"):
                salida = " ".join(str(respuestas.get(tc["id"], "")).split())
                lineas.append(f"→ {tc['name']}({tc['args']})\n   {salida[:300]}{'...' if len(salida) > 300 else ''}")
    return "\n".join(lineas) or "(no llamó a ninguna tool)"


def a_texto(datos: dict, campos: list[str]) -> str:
    return "\n".join(f"{c}: {', '.join(datos[c]) if isinstance(datos[c], list) else datos[c]}"
                     for c in campos if datos.get(c) not in (None, "", []))


@con_tiempo
def nodo_planificador(estado: EstadoMarketing) -> dict:
    mensaje = f"Fecha de hoy: {fecha_de_hoy.invoke({'formato': 'texto'})}.\nPedido: {estado['pedido']}"
    brief, resultado = preguntar_estructurado(planificador, mensaje, Brief)
    # Falla cerrada: canal y segmento tienen que ser valores válidos o la pieza no sigue
    brief["canal"] = normalizar(brief["canal"], CANALES)
    brief["segmento"] = normalizar(brief["segmento"], SEGMENTOS)
    brief["region"] = next((r for r in REGIONES if r.lower() == (brief.get("region") or "").strip().lower()), "")
    brief["situacion"] = normalizar(brief["situacion"], SITUACIONES) or "contenido"
    # Sin descuento, cuotas ni envío sin cargo en el pedido no hay promoción: el modelo marcaba así «anunciando la
    # nueva línea» y el revisor pedía legales de una promoción que no existe
    situacion_corregida = ""
    if brief["situacion"] == "promocion" and not PROMO_EN_EL_PEDIDO.search(estado["pedido"]):
        brief["situacion"] = situacion_corregida = ("lanzamiento" if re.search(r"\bnuev[oa]s?\b|lanz", estado["pedido"], re.IGNORECASE)
                                                    else "contenido")
    # El código valida lo que recibe de otro agente: un porcentaje, monto, fecha o cantidad de cuotas
    # que no está en el pedido no puede llegar al redactor como si fuera un dato (pasa a ser un marcador)
    reemplazos = []
    for campo, valor in brief.items():
        if campo == "angulo":      # habla del historial (p. ej., "la pieza del 23/10"): no es un dato de la pieza
            continue
        if isinstance(valor, str):
            brief[campo], hechos = quitar_datos_no_confirmados(valor, estado["pedido"])
            reemplazos += hechos
        elif isinstance(valor, list):
            limpios = [quitar_datos_no_confirmados(v, estado["pedido"]) for v in valor]
            brief[campo] = [texto for texto, _ in limpios]
            reemplazos += [r for _, hechos in limpios for r in hechos]
    tools = ("(Groq rechazó la respuesta del modelo: el brief vino incompleto o escrito como texto en vez de "
             "llamar a la tool. El código lo tomó de failed_generation y lo validó con el esquema)" if resultado.get("rescatado")
             else tools_llamadas(resultado))
    detalle = (f"Tools:\n{tools}\n\nBrief:\n"
               + a_texto(brief, ["fuera_de_alcance", "motivo", "canal", "segmento", "situacion", "objetivo",
                                 "mensaje_principal", "angulo", "llamado_a_la_accion", "datos_del_pedido",
                                 "datos_faltantes", "region"]))
    if situacion_corregida:
        detalle += f"\n\nEl planificador marcó «promocion», pero el pedido no trae descuento, cuotas ni envío sin cargo: el código la cambió a «{situacion_corregida}»."
    if reemplazos:
        detalle += "\n\nEl código reemplazó datos que no estaban en el pedido: " + "; ".join(reemplazos)
    return {"brief": brief, "pasos": [{"titulo": "1. Planificador (agente con tools)", "contenido": detalle}]}


@con_tiempo
def nodo_preparar_contexto(estado: EstadoMarketing) -> dict:
    b = estado["brief"]
    situaciones = {"promocion": "promociones", "lanzamiento": "lanzamientos",
                   "recuperacion": "recuperación de clientes detractores", "contenido": "contenido, tips y guías"}
    consultas = [f"reglas del canal {b['canal']}: largo máximo, hashtags, emojis y llamado a la acción",
                 f"cómo hablarle al segmento de clientes {b['segmento']}",
                 f"voz y tono para {situaciones[b['situacion']]}",
                 "palabras y frases prohibidas"]
    texto_pedido = " ".join([estado["pedido"]] + b["datos_del_pedido"] + b["datos_faltantes"]).lower()
    if b["situacion"] == "recuperacion":
        consultas.append("recuperación de detractores: cliente que tuvo una mala experiencia")
    if b["situacion"] in ("promocion", "recuperacion") or re.search(r"%|descuento|cuota|sin cargo|cup[oó]n", texto_pedido):
        consultas.append("legales obligatorios de promociones: descuentos, cuotas y envío sin cargo")
    if b["region"] or re.search(r"env[ií]o|entrega", texto_pedido):
        consultas.append("plazos de entrega: qué se puede decir sobre los envíos")
    secciones = fragmentos_guia(consultas)
    ejemplos = ejemplos_del_canal(b["canal"], b["segmento"])
    texto_ejemplos = "\n\n".join(f"[{e['canal']} · {e['segmento']} · {e['titulo']}]\n{e['texto']}" for e in ejemplos)
    detalle = ("Secciones de la guía (RAG): " + "; ".join(s["seccion"] for s in secciones)
               + "\nEjemplos del historial: " + ("; ".join(e["titulo"] for e in ejemplos) or "ninguno"))
    return {"contexto_guia": "\n\n".join(s["texto"] for s in secciones), "ejemplos": texto_ejemplos,
            "intentos": 0, "rechazos": 0, "correcciones": [],
            "pasos": [{"titulo": "2. Contexto para el redactor (código)", "contenido": detalle}]}


@con_tiempo
def nodo_redactor(estado: EstadoMarketing) -> dict:
    b = estado["brief"]
    mensaje = ("BRIEF\n" + a_texto(b, ["canal", "segmento", "situacion", "objetivo", "mensaje_principal", "angulo",
                                       "llamado_a_la_accion", "datos_del_pedido", "datos_faltantes", "region"])
               + f"\n\nSECCIONES DE LA GUÍA DE MARCA\n{estado['contexto_guia']}"
               + "\n\nPIEZAS YA ENVIADAS POR ESTE CANAL (copiá el formato, no sus frases)\n"
               + (estado["ejemplos"] or "ninguna"))
    if estado.get("correcciones"):
        # Las correcciones van primero: al final del mensaje, el modelo devolvía la misma versión
        mensaje = ("TU VERSIÓN ANTERIOR NO SE APROBÓ. Corregí todo esto y no devuelvas el mismo texto:\n"
                   + "\n".join(f"- {c}" for c in estado["correcciones"])
                   + f"\n\nVERSIÓN ANTERIOR\n{estado['validacion']['texto_final']}\n\n" + mensaje)
    pieza, _ = preguntar_estructurado(redactor, mensaje, Pieza)
    if b["canal"] not in ("Email", "Push"):
        pieza["encabezado"] = ""          # asunto y título solo existen en email y push
    intentos = estado.get("intentos", 0) + 1
    detalle = a_texto(pieza, ["idea", "titulo", "encabezado", "texto", "cta", "legales", "hashtags", "imagen_sugerida"])
    return {"pieza": pieza, "intentos": intentos,
            "pasos": [{"titulo": f"3. Redactor, versión {intentos} (agente)", "contenido": detalle}]}


@con_tiempo
def nodo_validar(estado: EstadoMarketing) -> dict:
    b, pieza = estado["brief"], estado["pieza"]
    pieza, voseo = pasar_a_voseo(pieza)
    recorte = recortar_push(pieza) if b["canal"] == "Push" else ""
    if recorte:
        pieza = {**pieza, "texto": recorte}
    validacion = validar_pieza(b["canal"], pieza, estado["pedido"], b["region"])
    detalle = ("Sin errores." if not validacion["errores"] else
               "Errores (vuelve al redactor):\n" + "\n".join(f"- {e}" for e in validacion["errores"]))
    if recorte:
        detalle = f"El texto del push pasaba de 90 caracteres: el código lo cortó en «{recorte}».\n" + detalle
    if voseo:
        detalle = f"Tenía tuteo: el código lo pasó a voseo ({', '.join(voseo)}).\n" + detalle
    if validacion["avisos"]:
        detalle += "\nAvisos para el revisor:\n" + "\n".join(f"- {a}" for a in validacion["avisos"])
    return {"pieza": pieza, "validacion": validacion, "correcciones": validacion["errores"],
            "pasos": [{"titulo": f"4. Validación automática, versión {estado['intentos']} (código)", "contenido": detalle}]}


@con_tiempo
def nodo_revisor(estado: EstadoMarketing) -> dict:
    b, pieza, validacion = estado["brief"], estado["pieza"], estado["validacion"]
    secciones = fragmentos_guia(["checklist antes de publicar", f"cómo hablarle al segmento de clientes {b['segmento']}"]
                                + (["recuperación de detractores"] if b["situacion"] == "recuperacion" else []))
    # La pieza va separada en partes: con el texto corrido, el revisor pedía legales que ya estaban
    partes = {"Asunto" if b["canal"] == "Email" else "Título": pieza["encabezado"],
              "Cuerpo": ("Hola, {nombre}: " if b["canal"] == "Email" else "") + cuerpo_de_la_pieza(b["canal"], pieza),
              "Botón (el único llamado a la acción)": pieza["cta"] if b["canal"] == "Email" else "",
              # Sin esta línea, el revisor rechazaba el push por no tener llamado a la acción
              "Llamado a la acción": "tocar la notificación abre la app (así lo define la guía para Push)"
                                     if b["canal"] == "Push" else "",
              "Legales": (LEGALES_WHATSAPP if b["canal"] == "WhatsApp" and pieza["legales"] else pieza["legales"]) or "(no lleva)",
              "Hashtags": " ".join(pieza["hashtags"]), "Idea": pieza["idea"], "Imagen sugerida": pieza["imagen_sugerida"]}
    mensaje = (f"PIEZA ({b['canal']}, segmento {b['segmento']})\n"
               + "\n".join(f"{nombre}: {valor}" for nombre, valor in partes.items() if valor)
               + "\n\nBRIEF\n" + a_texto(b, ["situacion", "objetivo", "mensaje_principal", "datos_del_pedido",
                                             "datos_faltantes"])
               + "\n\nAVISOS DEL CONTROL AUTOMÁTICO\n" + ("\n".join(f"- {a}" for a in validacion["avisos"]) or "ninguno")
               + "\n\nSECCIONES DE LA GUÍA DE MARCA\n" + "\n\n".join(s["texto"] for s in secciones))
    revision, _ = preguntar_estructurado(revisor, mensaje, Revision)
    # Aprueba el código, no el LLM: solo si todos los puntos del checklist dieron true
    fallidos = [p for p in PUNTOS_DEL_CHECKLIST if not revision[p]]
    revision["aprobada"] = not fallidos
    if fallidos and not revision["observaciones"]:
        revision["observaciones"] = [f"No cumple: {p.replace('_', ' ')}." for p in fallidos]
    detalle = "Checklist: " + ", ".join(f"{p.replace('_', ' ')} {'sí' if revision[p] else 'NO'}" for p in PUNTOS_DEL_CHECKLIST)
    detalle += ("\nAprobada." if revision["aprobada"] else
                "\nRechazada:\n" + "\n".join(f"- {o}" for o in revision["observaciones"]))
    return {"revision": revision, "correcciones": [] if revision["aprobada"] else revision["observaciones"],
            "rechazos": estado.get("rechazos", 0) + (0 if revision["aprobada"] else 1),
            "pasos": [{"titulo": f"5. Revisor, versión {estado['intentos']} (agente)", "contenido": detalle}]}


@con_tiempo
def nodo_guardar(estado: EstadoMarketing) -> dict:
    b, pieza = estado["brief"], estado["pieza"]
    resultado = guardar_pieza.invoke({"canal": b["canal"], "segmento": b["segmento"], "titulo": pieza["titulo"],
                                      "texto": estado["validacion"]["texto_final"]})
    return {"guardado": resultado, "pasos": [{"titulo": "6. Guardado (tool guardar_pieza)", "contenido": resultado}]}


def despues_del_planificador(estado: EstadoMarketing) -> str:
    b = estado.get("brief")
    return "preparar_contexto" if b and not b["fuera_de_alcance"] and b["canal"] and b["segmento"] else END


def despues_de_validar(estado: EstadoMarketing) -> str:
    if not estado["validacion"]["errores"]:
        return "revisor"
    return "redactor" if estado["intentos"] < MAX_VERSIONES else END


def despues_del_revisor(estado: EstadoMarketing) -> str:
    if estado["revision"]["aprobada"]:
        return "guardar"
    return "redactor" if estado["rechazos"] < MAX_RECHAZOS and estado["intentos"] < MAX_VERSIONES else END


grafo = StateGraph(EstadoMarketing)
grafo.add_node("planificador", nodo_planificador)
grafo.add_node("preparar_contexto", nodo_preparar_contexto)
grafo.add_node("redactor", nodo_redactor)
grafo.add_node("validar", nodo_validar)
grafo.add_node("revisor", nodo_revisor)
grafo.add_node("guardar", nodo_guardar)
grafo.add_edge(START, "planificador")
grafo.add_conditional_edges("planificador", despues_del_planificador, ["preparar_contexto", END])
grafo.add_edge("preparar_contexto", "redactor")
grafo.add_edge("redactor", "validar")
grafo.add_conditional_edges("validar", despues_de_validar, ["revisor", "redactor", END])
grafo.add_conditional_edges("revisor", despues_del_revisor, ["guardar", "redactor", END])
grafo.add_edge("guardar", END)
pipeline_marketing = grafo.compile()


# --------------------------------------------------------------------------
# Respuesta para la app
# --------------------------------------------------------------------------

def componer_respuesta(estado: EstadoMarketing) -> str:
    b = estado.get("brief")
    if not b or b["fuera_de_alcance"]:
        motivo = f" {b['motivo']}" if b and b.get("motivo") else ""
        return ("Solo puedo armar piezas de marketing de texto para Hornero (Instagram, email, WhatsApp, push o "
                f"LinkedIn).{motivo}")
    if not b["canal"] or not b["segmento"]:
        return ("No pude identificar el canal o el segmento de la pieza. Decime el canal (Instagram, email, WhatsApp, "
                "push o LinkedIn) y a quién va dirigida (clientes nuevos, recurrentes, premium o todos).")
    pieza, validacion = estado["pieza"], estado["validacion"]
    aprobada = bool(estado.get("revision", {}).get("aprobada")) and not validacion["errores"]
    if aprobada:
        guardada = estado.get("guardado", "").startswith("Pieza guardada")
        estado_pieza = ("Pieza aprobada y guardada en el historial" if guardada
                        else f"Pieza aprobada. {estado.get('guardado', '')}")
    else:
        motivo = ("el revisor la rechazó dos veces" if estado.get("rechazos", 0) >= MAX_RECHAZOS
                  else f"no pasó los controles de la guía en {estado['intentos']} versiones")
        estado_pieza = (f"La pieza no se guardó en el historial: {motivo}. "
                        "Te dejo la última versión con lo que falta corregir")
    lineas = [f"**{estado_pieza}** · {b['canal']} · {b['segmento']}", "",
              f"**Idea:** {pieza['idea']}", f"**Título:** {pieza['titulo']}"]
    if pieza.get("encabezado"):
        lineas.append(f"**{'Asunto' if b['canal'] == 'Email' else 'Título del push'}:** {pieza['encabezado']}")
    lineas += ["", "**Texto para publicar:**", "```text", validacion["texto_final"], "```",
               f"**Imagen sugerida:** {pieza['imagen_sugerida']}"]
    marcadores = sorted(set(re.findall(r"\[X\]%?|DD/MM|\$X", validacion["texto_final"])))
    if marcadores:
        lineas.append(f"**Antes de publicar, completá:** {', '.join(marcadores)} (datos que el pedido no daba).")
    if not aprobada:
        lineas += ["", "**Falta corregir:**"] + [f"- {c}" for c in estado.get("correcciones", [])]
    return "\n".join(lineas)


def responder(pregunta: str) -> dict:
    """Punto de entrada que usa main.py.

    Devuelve un diccionario con:
      - "respuesta": el texto final que se muestra en el chat.
      - "detalle": lista de pasos intermedios para mostrar en la app, cada uno como
        {"titulo": str, "contenido": str}. Sirve para ver qué agente o tool actuó.
    """
    inicio = perf_counter()
    contador.reiniciar()
    estado = {}
    try:
        # stream en vez de invoke: si algo falla en el medio, el detalle muestra hasta dónde llegó
        for estado in pipeline_marketing.stream({"pedido": pregunta, "pasos": []}, stream_mode="values"):
            pass
    except Exception as error:   # p. ej., Groq sin cupo o una salida que no respeta el esquema
        if type(error).__name__ == "RateLimitError":
            mensaje = "No pude armar la pieza: " + aviso_sin_cupo(error)
        else:
            mensaje = f"No pude armar la pieza: hubo un error ({type(error).__name__}). Probá de nuevo en un rato."
        return {"respuesta": mensaje,
                "detalle": estado.get("pasos", []) + [{"titulo": "Error", "contenido": str(error)[:1000]}]}
    uso = f"{perf_counter() - inicio:.1f} s de punta a punta · {contador.resumen()}"
    return {"respuesta": componer_respuesta(estado),
            "detalle": estado["pasos"] + [{"titulo": "Tiempo y uso del LLM", "contenido": uso}]}


if __name__ == "__main__":
    import sys
    salida = responder(" ".join(sys.argv[1:]) or EJEMPLOS[0])
    for paso in salida["detalle"]:
        print(f"\n=== {paso['titulo']}\n{paso['contenido']}")
    print(f"\n=== RESPUESTA\n{salida['respuesta']}")
