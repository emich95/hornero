"""
agentes.py (etapa 2): un orquestador que delega en especialistas.

    Consulta ──► Orquestador ──┬──► Marketing ............ grafo de la etapa 1 (marketing.py)
                (agente)       ├──► Políticas de RRHH .... agente con RAG del manual
                               ├──► People analytics ..... agente con modelo de attrition + subagente SQL (empleados.db)
                               └──► Supply chain ......... agente con modelo de ventas + MCP del clima + subagente SQL (ventas.db)

- Orquestador: el único punto de entrada. Sus tools son los especialistas; el docstring de cada una es
  todo lo que sabe de ellos. Puede llamar a más de uno si la consulta combina áreas. No tiene datos.
- Cada especialista tiene solo las tools y los datos de su área (mínimo privilegio) y rechaza lo que
  no es suyo. Los subagentes SQL abren su base en solo lectura y ven una sola tabla.
- Marketing reutiliza el pipeline de la etapa 1 sin cambios (marketing.py es una copia de etapa1/agentes.py).

Límites de alcance en tres capas: el orquestador rechaza lo que no es de ningún área; cada especialista
rechaza lo que no es del suyo; y cada uno accede solo a sus datos. Además, si el orquestador no delegó
en nadie, el código devuelve el mensaje de rechazo: el sistema nunca responde con lo que "sabe" el LLM.

main.py solo usa la función responder().
"""

import contextvars
import os
import re
from time import perf_counter

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_groq import ChatGroq

import marketing
from tools import (base_empleados, base_ventas, buscar_politica, explicar_factores_attrition,
                   explicar_factores_ventas, predecir_riesgo_attrition, predecir_ventas,
                   prever_ventas_semana, pronostico_semana, simular_escenario_attrition, simular_escenario_ventas)

load_dotenv()

MODELO = "openai/gpt-oss-20b"
# reasoning_effort="low": con el valor por defecto, gpt-oss-20b a veces razona hasta agotar los
# tokens de salida y devuelve una respuesta vacía. Además gasta muchos menos tokens.
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=MODELO, temperature=0.0, max_retries=5,
               reasoning_effort="low")

EJEMPLOS = [   # las 8 consultas de prueba de la consigna y las propias (main.py las muestra como botones)
    "Armá un post de Instagram para clientes Premium sobre la nueva línea de aires acondicionados.",
    "¿Cuántos días por semana puedo trabajar remoto y cómo pido el reintegro de internet?",
    "¿Qué área tiene la mayor tasa de renuncia? ¿Y qué centro?",
    "¿Qué factores explican más las renuncias? ¿Cuántos empleados en riesgo dejarían de estarlo si acercamos los sueldos 10 puntos al mercado?",
    "¿Qué categoría vendió más unidades en 2025 y en qué región?",
    "Si el precio del cobre sube 20%, ¿cuánto caerían las ventas de Electro y de Aires acondicionados?",
    "¿Cuántos aires acondicionados esperamos vender la semana que viene en Cuyo según el pronóstico?",
    "¿Quién ganó el Mundial 2022?",
    "Según el pronóstico, ¿en qué región conviene empujar los aires acondicionados la semana que viene? Armá un push para clientes recurrentes de esa región.",
    "¿Qué dice el manual sobre capacitación y cuántas capacitaciones promedio tuvieron en el último año los que renunciaron frente a los que no?",
    "Borrá de la base a los empleados que renunciaron.",
]

RECHAZO = ("Solo puedo ayudarte con temas internos de Hornero: piezas de marketing de texto, políticas de RRHH, "
           "datos y riesgo de renuncia del personal, y ventas y clima para supply chain. Esa consulta queda fuera "
           "de lo que puedo responder.")
SOLO_LECTURA = ("No puedo borrar ni modificar datos: las bases de empleados y de ventas se abren en modo solo "
                "lectura y los especialistas solo pueden consultarlas.")
PIDE_ESCRITURA = re.compile(r"\b(borr|elimin|modific|actualiz|insert|agreg|cambi|edit)\w*\b.*\b(bases?|tablas?|registros?|datos|empleados?|ventas?)\b",
                            re.IGNORECASE)


# --------------------------------------------------------------------------
# Contador de llamadas y tokens (el de la clase 4), compartido por todos los agentes
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
llm.callbacks = [contador]
# Marketing trae sus propios LLM (los de la etapa 1): se les cuelga el mismo contador
marketing.llm.callbacks = [contador]
marketing.llm_cuidadoso.callbacks = [contador]

# Lo que hizo cada especialista en la consulta en curso, para el detalle de la app. Es una ContextVar
# y no una lista global: así dos sesiones de Streamlit no mezclan sus pasos, y los especialistas que
# corren en paralelo (en threads) escriben en la lista de su consulta.
registro: contextvars.ContextVar[list] = contextvars.ContextVar("registro")
# La pregunta original del usuario: Marketing la usa para controlar que el orquestador no le agregue datos
pregunta_usuario: contextvars.ContextVar[str] = contextvars.ContextVar("pregunta_usuario", default="")


def registrar(titulo: str, contenido: str):
    pasos = registro.get(None)
    if pasos is not None:
        pasos.append({"titulo": titulo, "contenido": contenido})


def resumen_tools(resultado, max_caracteres: int = 400) -> str:
    """Qué tools llamó un agente, con qué argumentos y el comienzo de lo que le devolvieron."""
    respuestas = {m.tool_call_id: m.content for m in resultado["messages"] if getattr(m, "type", "") == "tool"}
    lineas = []
    for mensaje in resultado["messages"]:
        for tc in getattr(mensaje, "tool_calls", None) or []:
            salida = " ".join(str(respuestas.get(tc["id"], "")).split())
            lineas.append(f"→ {tc['name']}({tc['args']})\n   {salida[:max_caracteres]}"
                          f"{'...' if len(salida) > max_caracteres else ''}")
    return "\n".join(lineas) or "(no llamó a ninguna tool)"


# Números de 4 cifras o más, con o sin separador de miles (74894, 74.894, 74 894)
NUMERO_GRANDE = re.compile(r"(?<![\d.,])\d{1,3}(?:[.   ]\d{3})+(?![\d])|\d{4,}")


def numeros_sin_fuente(resultado, pregunta: str) -> list[str]:
    """Números grandes de la respuesta final de un agente que no aparecen ni en la pregunta, ni en lo que
    devolvieron sus tools, ni en los argumentos con que las llamó (p. ej., el año dentro del SQL)."""
    mensajes = resultado["messages"]
    fuentes = " ".join([pregunta] + [str(m.content) for m in mensajes if getattr(m, "type", "") == "tool"]
                       + [str(tc["args"]) for m in mensajes for tc in (getattr(m, "tool_calls", None) or [])])
    conocidos = {re.sub(r"\D", "", n) for n in NUMERO_GRANDE.findall(fuentes)}
    # Los años quedan afuera: en la consulta 9 marcó «2026», sacado de una fecha, y la corrección salió peor
    return [n for n in dict.fromkeys(NUMERO_GRANDE.findall(str(mensajes[-1].content)))
            if re.sub(r"\D", "", n) not in conocidos and not re.fullmatch(r"(19|20)\d\d", n)]


def controlar_numeros(agente, nombre: str, resultado, pregunta: str, otra_vuelta: str = "o consultá la tool que los da"):
    """Si la respuesta trae números que no salieron de ninguna tool, se le pide al mismo agente que la
    corrija (una vez). En la consulta 5, supply chain agregó un total de AMBA inventado (152.321)."""
    sin_fuente = numeros_sin_fuente(resultado, pregunta)
    if not sin_fuente:
        return resultado
    lista = ", ".join(sin_fuente)
    registrar(f"{nombre}: control de números (código)",
              f"Estos números de la respuesta no salieron de ninguna tool: {lista}. Se le pide que la corrija.")
    corregido = agente.invoke({"messages": resultado["messages"] + [{"role": "user", "content": (
        f"Control automático: estos números de tu respuesta no aparecen en lo que te devolvieron las tools: {lista}. "
        f"Reescribí la respuesta completa sin ellos, {otra_vuelta}. No agregues explicaciones que no "
        "salgan de las tools.")}]})
    if numeros_sin_fuente(corregido, pregunta):
        registrar(f"{nombre}: control de números (código)", "Después de la corrección siguen apareciendo números sin fuente.")
    return corregido


def consultar(agente, nombre: str, pregunta: str) -> str:
    """Invoca a un agente especialista y anota en el registro qué hizo y cuánto tardó."""
    inicio = perf_counter()
    registrar(f"→ {nombre}", f"Recibe: {pregunta}")
    try:
        resultado = agente.invoke({"messages": [{"role": "user", "content": pregunta}]})
        resultado = controlar_numeros(agente, nombre, resultado, pregunta)
    except Exception as error:                     # p. ej., Groq sin cupo después de los reintentos
        registrar(f"{nombre}: error", str(error)[:500])
        return f"{nombre} no pudo responder por un error ({type(error).__name__}). No inventes la respuesta."
    respuesta = resultado["messages"][-1].content
    registrar(f"{nombre} ({perf_counter() - inicio:.1f} s)", f"{resumen_tools(resultado)}\n\nResponde:\n{respuesta}")
    return respuesta


# --------------------------------------------------------------------------
# Subagentes SQL (SQLDatabaseToolkit, clase 4), uno por base
# --------------------------------------------------------------------------
# Del toolkit se usan sql_db_query y sql_db_schema. Se dejan afuera sql_db_query_checker, que gasta una
# llamada extra al LLM por consulta, y sql_db_list_tables, que no recibe parámetros (gpt-oss-20b genera
# mal esas llamadas). Como cada base tiene una sola tabla, su esquema va directo en el prompt.

def prompt_sql(base, contexto: str) -> str:
    return f"""ROL: Sos un analista de datos que responde preguntas con consultas SQL sobre una base SQLite \
de Hornero. {contexto}

ALCANCE: solo consultas de lectura (SELECT) sobre esta tabla. La base está en modo solo lectura: si te piden \
borrar, modificar o agregar datos, respondé que no se puede y no lo intentes.

ESQUEMA (con filas de ejemplo):
{base.get_table_info()}

CÓMO TRABAJÁS:
- Ejecutá la consulta con sql_db_query. Usá agregaciones (COUNT, SUM, AVG, GROUP BY), ORDER BY y LIMIT: \
nunca SELECT * ni listados de filas sueltas.
- Si piden un ranking por una dimensión y otro por otra ("¿qué área…? ¿y qué centro?"), hacé una consulta por \
dimensión (GROUP BY area y, aparte, GROUP BY centro). Agrupar por las dos juntas da cada combinación, no la tasa \
del área ni la del centro.
- Si la consulta da error, leé el mensaje, corregila y volvé a ejecutarla.
- Respondé con los números que salieron de la base y, al final, la consulta SQL que usaste. Nunca inventes \
números: si la base no tiene el dato, decilo.

TONO: castellano rioplatense, breve."""


def herramientas_sql(base) -> list:
    return [t for t in SQLDatabaseToolkit(db=base, llm=llm).get_tools() if t.name in ("sql_db_query", "sql_db_schema")]


sql_empleados = create_agent(
    model=llm, tools=herramientas_sql(base_empleados),
    system_prompt=prompt_sql(base_empleados,
                             "La tabla empleados tiene una fila por empleado. renuncio = 'Si' si renunció en el "
                             "último año. Tasa de renuncia = AVG(renuncio = 'Si') * 100, no la cantidad de renuncias. "
                             "Nunca muestres el sueldo de una persona: los sueldos, solo agregados por grupo."))

sql_ventas = create_agent(
    model=llm, tools=herramientas_sql(base_ventas),
    system_prompt=prompt_sql(base_ventas,
                             "La tabla ventas tiene una fila por semana, región y categoría; semana es la fecha del "
                             "lunes en formato 'AAAA-MM-DD' (texto). Para un año, filtrá con semana BETWEEN "
                             "'2025-01-01' AND '2025-12-31'."))


@tool
def consultar_datos_empleados(pregunta: str) -> str:
    """Responde preguntas exploratorias sobre los datos de los empleados (cantidades, tasas de renuncia,
    promedios por área, centro, nivel o modalidad, antigüedad, horas extra, capacitaciones, sueldos
    agregados) con un analista SQL sobre empleados.db, en solo lectura. Pasale la pregunta completa."""
    return consultar(sql_empleados, "Subagente SQL de empleados", pregunta)


@tool
def consultar_datos_ventas(pregunta: str) -> str:
    """Responde preguntas exploratorias sobre las ventas históricas (unidades por semana, categoría y
    región, de enero de 2023 a octubre de 2026, con su temperatura, lluvia y precio del cobre) con un
    analista SQL sobre ventas.db, en solo lectura. Pasale la pregunta completa."""
    return consultar(sql_ventas, "Subagente SQL de ventas", pregunta)


# --------------------------------------------------------------------------
# Especialistas
# --------------------------------------------------------------------------

PROMPT_POLITICAS = """ROL: Sos el especialista en políticas de RRHH de Hornero. Respondés consultas de los \
colaboradores con el manual de políticas.

ALCANCE: solo lo que dice el manual (trabajo híbrido, reintegros, capacitación, licencias, beneficios, código de \
conducta, contacto con RRHH). No tenés datos de empleados ni de ventas: si te preguntan eso, decí que no es tu área.

CÓMO TRABAJÁS:
- Siempre buscá con buscar_politica antes de responder; si la pregunta tiene varias partes, buscá cada una.
- Respondé solo con lo que dicen las secciones que encontraste y nombrá la sección (por ejemplo, "1.2 Días de \
trabajo remoto"). Nunca inventes una política, un monto ni un plazo.
- Si el manual no lo dice, respondé: "No encontré esa información en el manual de políticas" y sugerí el \
contacto de RRHH que figura en el manual.

TONO: claro y cordial, en voseo, con viñetas si hay varios puntos."""

PROMPT_PEOPLE = """ROL: Sos el especialista en people analytics de Hornero. Respondés preguntas sobre los \
empleados con datos y con el modelo de attrition (probabilidad de renunciar) que entrenó el equipo.

ALCANCE: solo datos de empleados y riesgo de renuncia. No tenés las políticas del manual ni datos de ventas: si te \
preguntan eso, decí que no es tu área. La base es de solo lectura: no se pueden borrar ni modificar datos.

CÓMO TRABAJÁS:
- Preguntas exploratorias sobre lo que pasó (cuántos, qué área, promedios, tasas): consultar_datos_empleados, \
con la pregunta del usuario tal cual, sin reformularla. "Tasa de renuncia" es el porcentaje que renunció, no la cantidad.
- Riesgo de renuncia de los empleados activos: predecir_riesgo_attrition.
- Qué factores explican las renuncias: explicar_factores_attrition. Para cada variable da su peso y su sentido \
tal cual lo dice la tool (si sube, las chances de renunciar aumentan o disminuyen, y cuánto). Decí "chances", \
como la tool, y no "probabilidad": el porcentaje es sobre las chances y no vale para la probabilidad.
- Escenarios ("qué pasaría si"): simular_escenario_attrition. "Acercar los sueldos N puntos al mercado" es \
variable='sueldo_vs_mercado_pct', cambio=N, limite=0: sube a quien está debajo del mercado sin pasarlo y no \
toca a quien ya está arriba. Explicá esa interpretación en la respuesta.
- Decí siempre el umbral de riesgo que usaste (0,5 si no te piden otro) y que el riesgo se calcula sobre los \
empleados activos.
- Usá solo los números que devuelven las tools. Nunca muestres el sueldo de una persona.

TONO: castellano rioplatense, concreto, con los números clave primero."""

PROMPT_SUPPLY = """ROL: Sos el especialista en supply chain de Hornero. Respondés preguntas sobre ventas con los \
datos históricos, el modelo de ventas que entrenó el equipo y el pronóstico del clima.

ALCANCE: solo ventas, demanda, clima y precio del cobre. No tenés datos de empleados ni políticas de RRHH: si te \
preguntan eso, decí que no es tu área. La base es de solo lectura.

CÓMO TRABAJÁS:
- Preguntas sobre lo que ya se vendió (rankings, totales por año, categoría o región): consultar_datos_ventas. \
"Qué categoría vendió más en un año y en qué región" = la categoría líder y la región donde esa categoría vendió \
más; mencioná también la región que más vendió en total y aclarás la interpretación. Pedile al analista las tres \
cosas en una sola pregunta: el total por categoría, la región donde más vendió la categoría líder y el total por región.
- Ventas esperadas para condiciones dadas: predecir_ventas.
- Qué mueve las ventas: explicar_factores_ventas. Da el sentido y el tamaño de cada efecto tal cual lo dice la tool.
- Escenarios ("si el cobre sube 20 %"): simular_escenario_ventas, con todas las categorías que pregunten en una \
sola llamada.
- Ventas de la semana que viene según el pronóstico: prever_ventas_semana(categoria, regiones). Decí de qué \
fuente salió el pronóstico y qué fechas cubre. Para el clima solo, sin ventas: pronostico_semana(region).
- "En qué región conviene empujar una categoría": prever_ventas_semana con todas las regiones. Recomendá una \
región y explicá el criterio: dónde el pronóstico sube más la demanda frente a un clima normal para la época, \
sin olvidar el volumen (AMBA siempre vende más por tamaño). Si la tool avisa que la estimación de una región \
extrapola, no la recomiendes por ese aumento y decilo.
- Usá solo los números que devuelven las tools; mostrá unidades y porcentajes, y avisá si algo extrapola. Si te \
falta un número, pedíselo a la tool o no lo des. No agregues explicaciones de por qué se vende más o menos \
(población, crédito, tipo de productos) que no salgan de las tools.

TONO: castellano rioplatense, concreto, con los números clave primero."""

agente_politicas = create_agent(model=llm, tools=[buscar_politica], system_prompt=PROMPT_POLITICAS)
agente_people = create_agent(model=llm, system_prompt=PROMPT_PEOPLE,
                             tools=[consultar_datos_empleados, predecir_riesgo_attrition,
                                    explicar_factores_attrition, simular_escenario_attrition])
agente_supply = create_agent(model=llm, system_prompt=PROMPT_SUPPLY,
                             tools=[consultar_datos_ventas, predecir_ventas, explicar_factores_ventas,
                                    simular_escenario_ventas, prever_ventas_semana, pronostico_semana])


# --------------------------------------------------------------------------
# Las tools del orquestador: un especialista cada una
# --------------------------------------------------------------------------

# Datos de promoción que Marketing solo usa si los dijo el usuario
DATOS_DE_PROMO = re.compile(r"\d+(?:[.,]\d+)?\s?%|\d+\s+cuotas|sin cargo|env[ií]o gratis|cup[oó]n|\$\s?\d[\d.]*", re.IGNORECASE)


@tool
def consultar_marketing(pedido: str) -> str:
    """Equipo de Marketing: arma UNA pieza de marketing de texto (post de Instagram, email, WhatsApp, push o
    LinkedIn) para clientes de Hornero, la revisa contra la guía de marca y, si se aprueba, la guarda en el
    historial de contenidos. Pasale el pedido completo del usuario, con el canal, el segmento y los datos
    (producto, categoría, descuento) tal cual los dijo."""
    inicio = perf_counter()
    registrar("→ Marketing (grafo de la etapa 1)", f"Recibe: {pedido}")
    # Falla cerrada: en la consulta 9 el orquestador le agregó «Descuento: 15%» a un pedido sin descuento, y la
    # validación de la etapa 1 compara contra el pedido que recibe. Lo que el usuario no dijo no pasa
    plano = lambda t: re.sub(r"\s", "", t.lower())
    agregados = [d for d in dict.fromkeys(DATOS_DE_PROMO.findall(pedido)) if plano(d) not in plano(pregunta_usuario.get())]
    if agregados:
        registrar("Marketing: control del pedido (código)", f"El pedido trae datos que el usuario no dio: {', '.join(agregados)}. "
                                                            "No se arma la pieza y se le devuelve al orquestador.")
        return (f"Marketing no armó la pieza: el pedido trae datos que el usuario no dio ({', '.join(agregados)}). "
                "Volvé a llamar a Marketing solo con lo que pidió el usuario y el dato del otro especialista, sin agregar "
                "descuentos ni condiciones.")
    estado = {}
    try:
        for estado in marketing.pipeline_marketing.stream({"pedido": pedido, "pasos": []}, stream_mode="values"):
            pass
    except Exception as error:
        for paso in estado.get("pasos", []):
            registrar(f"Marketing · {paso['titulo']}", paso["contenido"])
        registrar("Marketing: error", str(error)[:500])
        return f"Marketing no pudo armar la pieza por un error ({type(error).__name__}). No la inventes."
    for paso in estado["pasos"]:
        registrar(f"Marketing · {paso['titulo']}", paso["contenido"])
    respuesta = marketing.componer_respuesta(estado)
    registrar(f"Marketing ({perf_counter() - inicio:.1f} s)", respuesta)
    return respuesta


@tool
def consultar_politicas_rrhh(pregunta: str) -> str:
    """Políticas de RRHH: responde con el manual de políticas de Hornero (trabajo híbrido y remoto, reintegros
    de gastos, capacitación, licencias, beneficios, código de conducta, contacto con RRHH). No tiene datos
    de empleados ni de ventas."""
    return consultar(agente_politicas, "Políticas de RRHH", pregunta)


@tool
def consultar_people_analytics(pregunta: str) -> str:
    """People analytics: datos de los empleados de Hornero (cantidades, tasas de renuncia por área o centro,
    antigüedad, horas extra, capacitaciones, sueldos agregados) y el modelo de attrition (riesgo de renuncia,
    qué factores lo explican, escenarios como subir sueldos). No conoce las políticas del manual."""
    return consultar(agente_people, "People analytics", pregunta)


@tool
def consultar_supply_chain(pregunta: str) -> str:
    """Supply chain: ventas semanales por categoría y región (2023 a 2026), el modelo de ventas (predicción,
    efecto del clima y del precio del cobre, escenarios) y el pronóstico del clima de la semana que viene para
    estimar ventas."""
    return consultar(agente_supply, "Supply chain", pregunta)


ESPECIALISTAS = [consultar_marketing, consultar_politicas_rrhh, consultar_people_analytics, consultar_supply_chain]
AREAS = {"consultar_marketing": "Marketing", "consultar_politicas_rrhh": "Políticas de RRHH",
         "consultar_people_analytics": "People analytics", "consultar_supply_chain": "Supply chain"}


def cifras(texto: str) -> set[str]:
    """Los números de un texto sin separadores, para comparar «74.894» con «74 894» o «1,40» con «1.40»."""
    return {re.sub(r"\D", "", n) for n in re.findall(r"\d+(?:[.,\u00a0\u202f ]\d+)*", texto)}


def respuestas_de_especialistas(mensajes) -> list[tuple[str, str]]:
    """La última respuesta de cada especialista, en el orden en que se los llamó."""
    ultimas = {}
    for m in mensajes:
        if getattr(m, "type", "") == "tool" and m.name in AREAS:
            ultimas.pop(m.name, None)
            ultimas[m.name] = str(m.content)
    return list(ultimas.items())


def unir_respuestas(especialistas: list[tuple[str, str]]) -> str:
    if len(especialistas) == 1:
        return especialistas[0][1]
    return "\n\n---\n\n".join(f"**{AREAS[nombre]}**\n\n{texto}" for nombre, texto in especialistas)

PROMPT_ORQUESTADOR = f"""ROL: Sos el punto de entrada del asistente interno de Hornero, un e-commerce argentino de \
artículos para el hogar. No respondés vos: decidís qué especialista responde cada consulta.

ALCANCE: marketing (piezas de texto para clientes), políticas de RRHH, datos y riesgo de renuncia de los \
empleados, y ventas, demanda y clima para supply chain. Todo lo demás (deportes, noticias, cultura general, \
programación, consejos personales) está fuera de alcance: respondé exactamente "{RECHAZO}" sin llamar a nadie.

CÓMO TRABAJÁS:
- Llamá solo a los especialistas que hagan falta. Si la consulta combina áreas, llamá a cada uno con su parte.
- Pasale a cada especialista la parte de la consulta que le toca, con todos los datos que dio el usuario.
- Si una parte depende del resultado de otra (por ejemplo, una pieza de marketing para la región que recomiende \
supply chain), llamalos de a uno: primero el que da el dato y después el otro, con ese dato en el pedido.
- A Marketing pasale el pedido con las palabras del usuario, cambiando solo lo que dio otro especialista (por \
ejemplo, "Armá un push para clientes recurrentes de AMBA sobre aires acondicionados"). No agregues descuentos, \
porcentajes, cuotas, categorías ni aclaraciones (tampoco "sin descuentos").
- A cada especialista llamalo una sola vez por consulta.
- No pidas aclaraciones: los especialistas saben qué suponer.
- Si piden borrar, modificar o agregar datos de empleados o de ventas, pasáselo al especialista de esa área: él \
explica que la base es de solo lectura.
- Si una parte de la consulta está fuera de alcance, respondé el resto y aclará que esa parte no la podés responder.
- Nunca agregues datos por tu cuenta. Si un especialista no tiene el dato o dio un error, decilo.

RESPUESTA: si preguntan dónde o qué conviene, empezá con la recomendación en una frase y el número que la \
sostiene. Incluí completo lo que te devolvieron los especialistas: números, umbrales, supuestos, fuentes (las \
secciones del manual con su número, la consulta SQL, de dónde salió el pronóstico) y las piezas de marketing tal \
cual, sin reescribirlas. Si respondieron dos especialistas, separá sus partes con un subtítulo.

TONO: castellano rioplatense, claro y breve."""

orquestador = create_agent(model=llm, tools=ESPECIALISTAS, system_prompt=PROMPT_ORQUESTADOR)


# --------------------------------------------------------------------------
# Punto de entrada para la app
# --------------------------------------------------------------------------

def responder(pregunta: str) -> dict:
    """Punto de entrada que usa main.py.

    Devuelve un diccionario con:
      - "respuesta": el texto final que se muestra en el chat.
      - "detalle": lista de pasos intermedios para mostrar en la app, cada uno como
        {"titulo": str, "contenido": str}. Sirve para ver qué agente o tool actuó.
    """
    inicio = perf_counter()
    contador.reiniciar()
    pasos = []
    registro.set(pasos)
    pregunta_usuario.set(pregunta)
    try:
        resultado = orquestador.invoke({"messages": [{"role": "user", "content": pregunta}]})
        # El orquestador corrige sin volver a llamar a los especialistas (en la consulta 9 los volvió a llamar y
        # supply chain cambió de recomendación)
        resultado = controlar_numeros(orquestador, "Orquestador", resultado, pregunta,
                                      "sin volver a llamar a los especialistas: usá solo lo que ya te respondieron")
    except Exception as error:
        if type(error).__name__ == "RateLimitError":
            mensaje = "No pude responder: " + marketing.aviso_sin_cupo(error)
        else:
            mensaje = f"No pude responder: hubo un error ({type(error).__name__}). Probá de nuevo en un rato."
        return {"respuesta": mensaje, "detalle": pasos + [{"titulo": "Error", "contenido": str(error)[:1000]},
                                                          {"titulo": "⏱ Tiempo total", "contenido": f"{perf_counter() - inicio:.1f} s"}]}
    segundos = perf_counter() - inicio

    nombres = {t.name for t in ESPECIALISTAS}
    delegaciones = [tc for m in resultado["messages"] for tc in (getattr(m, "tool_calls", None) or [])
                    if tc["name"] in nombres]
    respuesta = resultado["messages"][-1].content
    respuesta_orquestador = respuesta
    if not delegaciones:
        # Falla cerrada: si no delegó en ningún especialista, la respuesta no sale de ningún dato de Hornero. Si
        # pedía borrar o modificar datos, se aclara que las bases son de solo lectura (en la corrida oficial, «Borrá
        # de la base a los empleados que renunciaron» se respondió con el rechazo genérico)
        respuesta = SOLO_LECTURA if PIDE_ESCRITURA.search(pregunta) else RECHAZO
    elif respuesta.strip() == RECHAZO:
        # Delegó, pero contestó con el rechazo fijo en vez de lo que dijo el especialista (pasó en «Borrá de la
        # base a los empleados que renunciaron»): va la respuesta del especialista
        respuesta = unir_respuestas(respuestas_de_especialistas(resultado["messages"]))
    else:
        # Si el orquestador resumió y dejó afuera algún número de los especialistas, va lo que dijeron ellos tal cual
        # (en la corrida oficial sacó la tabla de regiones de la consulta 9 y la sección 3.2 de la consulta 10)
        especialistas = respuestas_de_especialistas(resultado["messages"])
        faltan = {AREAS[n]: sorted(cifras(t) - cifras(respuesta)) for n, t in especialistas}
        faltan = {area: numeros for area, numeros in faltan.items() if numeros}
        if faltan:
            respuesta = unir_respuestas(especialistas)
            pasos.append({"titulo": "Control de la respuesta (código)", "contenido": (
                "El orquestador dejó afuera números de " + "; ".join(f"{a} ({', '.join(n[:8])})" for a, n in faltan.items())
                + ": la respuesta es la de los especialistas tal cual.")})
    # La pieza de marketing tiene que llegar tal cual (es la que quedó guardada): si el orquestador la
    # reescribió o la recortó, se agrega la respuesta original de Marketing
    for paso in pasos:
        if paso["titulo"].startswith("Marketing (") and "```text" in paso["contenido"]:
            texto_pieza = paso["contenido"].split("```text", 1)[1].split("```", 1)[0]
            if " ".join(texto_pieza.split()) not in " ".join(respuesta.split()):
                respuesta += "\n\n---\n**Pieza de Marketing (tal cual la aprobó y guardó el equipo):**\n\n" + paso["contenido"]

    orquestador_paso = {"titulo": "Orquestador", "contenido": (
        "Delegó en: " + "; ".join(f"{tc['name']}({tc['args']})" for tc in delegaciones) if delegaciones
        else ("No delegó en ningún especialista: la consulta pide borrar o modificar datos y el código responde que "
              "las bases son de solo lectura. " if respuesta == SOLO_LECTURA
              else "No delegó en ningún especialista: la consulta está fuera de alcance. ")
             + ("El orquestador había respondido con el mensaje de rechazo." if respuesta_orquestador.strip() == RECHAZO
                else f"El orquestador había respondido «{respuesta_orquestador[:300]}» y el código lo reemplazó."))}
    uso = f"{segundos:.1f} s de punta a punta · {contador.resumen()}"
    return {"respuesta": respuesta,
            "detalle": [orquestador_paso] + pasos + [{"titulo": "⏱ Tiempo total", "contenido": uso}]}


if __name__ == "__main__":
    import sys
    salida = responder(" ".join(sys.argv[1:]) or EJEMPLOS[1])
    for paso in salida["detalle"]:
        print(f"\n=== {paso['titulo']}\n{paso['contenido']}")
    print(f"\n=== RESPUESTA\n{salida['respuesta']}")
