# TP Hornero: estado

Fuente de verdad: `../TP_Aplicaciones IA Generativa_2026/Consigna_TP.pdf` (material original, sin tocar).
Plan de trabajo: `PLAN_TP_Hornero.md` (documento de traspaso, fuera de esta carpeta).
Entrega: domingo 29/11/2026, 23:59 (hora de Buenos Aires). La tabla de demoras de la consigna dice
que no hay descuento hasta el 6/12, pero planificamos para el 29/11.

## Reglas de trabajo

- El informe se escribe en un Google Doc que vamos iterando y al final se exporta a PDF.
- Antes de cerrar cada fase, se repasa el checklist de abajo contra la consigna.

## Entregables (consigna, sección 5)

| # | Entregable | Estado |
|---|---|---|
| E1 | `etapa1/` y `etapa2/`, cada una con `tools.py`, `agentes.py`, `main.py` (+ módulos extra, p. ej. el script que entrena los modelos). Cada una corre sola con `streamlit run main.py` desde su carpeta. `RUTA_DATOS` ajustado | ✅ código (rutas relativas a cada carpeta); falta la validación de la etapa 2 en la app |
| E2 | `contenidos_aprobados.csv`: las 8 filas originales + las piezas de las consultas de prueba de **las dos etapas** | ✅ 21/11 en la raíz, con `consolidar_csv.py` (15 filas: 8 + 4 de la etapa 1 + 3 de la etapa 2, la última de la prueba de Emiliano en la app). Volver a correrlo si se aprueban más piezas |
| E3 | Informe en PDF que cubre las dos etapas (ver E3 abajo) | ⬜ |
| E4 | `TP_IAGen_2026_APELLIDOS.zip` con `etapa1/`, `etapa2/`, el CSV y el PDF; **sin `.env` ni venv**. Lo sube un solo integrante al campus, en "Entregar TP" | ⬜ |

La etapa 1 se evalúa aparte: no alcanza con que la etapa 2 la reutilice.

### E3: contenido obligatorio del informe

- [x] Diagrama de la arquitectura de cada etapa (el de la etapa 2 actualizado el 21/11 con los controles nuevos)
- [x] Qué agentes hay, qué tools y datos tiene cada uno, y por qué
- [x] Cómo se entrenaron y validaron los dos modelos, y qué se encontró al interpretarlos (+ figuras 3 y 4, 21/11)
- [x] Decisiones de diseño y alternativas descartadas (tablas 2 y 4 con las decisiones hasta el 21/11)
- [x] Cómo se implementaron los límites de alcance (con el resultado de la consulta 11)
- [x] Tabla de tiempos de respuesta de la etapa 2, con la explicación de las diferencias (tabla 9)
- [ ] Si se hace A2A: qué cambió
- [x] Limitaciones conocidas
- [ ] Formato: A4, letra de 11 pt o más, máximo 10 páginas sin contar la carátula; anexo opcional de hasta 5. **Al 21/11 el borrador tiene ~6.900 palabras y la vista previa da 17 páginas: hay que recortar después de que Emiliano lo lea**
- [ ] Carátula con nombres completos y legajos de los 4 integrantes (**pendiente: datos del grupo**)

## Requisitos

### Transversal (pág. 2)
- [x] Contrato: `responder(pregunta) -> {"respuesta": str, "detalle": [{"titulo", "contenido"}]}`

### Etapa 1: Marketing (págs. 3-4)
- [x] 1.1 Pipeline de **3 agentes en secuencia**; justificar qué hace cada uno
- [x] 1.2 Solo texto: idea, copy, hashtags, asunto del email, una línea de imagen sugerida
- [x] 1.3 RAG sobre `guia_de_marca.pdf` (tono, reglas del canal, legales)
- [x] 1.4 Índice FAISS guardado en disco (vectores + chunks en JSON); si existe, se levanta; si no, se arma
- [x] 1.5 Tool de **lectura** del historial (no repetir mensajes por segmento; usar ejemplos)
- [x] 1.6 Tool de **escritura**: fila nueva con `fecha, canal, segmento, titulo, texto`
- [x] 1.7 System prompt con rol, alcance y tono en cada agente + "no inventar datos" (precios, plazos, códigos)
- [x] 1.8 Flujo con `StateGraph` de LangGraph
- [x] 1.9 El código decide qué recibe cada agente (el revisor recibe la pieza, no el historial)
- [x] 1.10 Consultas de prueba 1 a 4 resueltas y guardadas en el CSV

### Etapa 2: RRHH y Supply chain (págs. 4-6)
- [x] 2.1 Un **único punto de entrada** que decide a quién delegar
- [x] 2.2 Estructura multiagente **distinta** a la de la etapa 1
- [x] 2.3 Marketing de la etapa 1 integrado como un área más
- [x] 2.4 RAG sobre `manual_politicas_rrhh.pdf`, índice persistido, nunca inventar políticas
- [x] 2.5 Modelo de **attrition** propio (`empleados.csv`): logística, AUC test 0,81 (18/11)
- [x] 2.6 Modelo de **ventas** propio (`ventas.csv`: categoría, región, temperatura, lluvia, cobre): lineal-log, MAPE test 12 % (18/11)
- [x] 2.7 Para cada modelo, tools para **predecir**, **interpretar** (qué variables y en qué sentido) y **simular**
- [x] 2.8 Tools **lo más generales posible** (una de simulación con variable y cambio, no una por variable)
- [x] 2.9 Agentes SQL (`SQLDatabaseToolkit`) para las preguntas exploratorias; **base en solo lectura**
- [x] 2.10 MCP `mcp_weather_server` con `MultiServerMCPClient`; salida resumida; pronóstico → ventas de la semana que viene. Si falla: respaldo con el CSV, documentado en el informe
- [x] 2.11 Mínimo privilegio: cada agente accede solo a los datos y documentos de su área
- [x] 2.12 Rechazo de las preguntas fuera de alcance
- [x] 2.13 Tiempo de punta a punta con `time.perf_counter()`, visible en el detalle de la app
- [x] 2.14 Consultas de prueba 1 a 8 + **al menos 2 propias** (p. ej., que combinen especialistas o un caso límite) → propias: 9 (supply + marketing), 10 (políticas + people), 11 (borrar datos). Corrida oficial 21/11
- [ ] 2.15 (Opcional) A2A con al menos un especialista

## Patrones de las prácticas a replicar

Fuente: `MiM/AIAG/prácticas/` (clases 1, 2 y 4). Seguirlos y nombrarlos en el informe.

- **System prompts** con bloques ROL / SCOPE / TONO, la regla "siempre usá las tools, nunca inventes
  números" y "si no tenés el dato, decilo" (clase 2).
- **Tools:** devuelven texto corto; docstring claro; al menos un parámetro; mensajes explícitos ante
  errores (un id inexistente, un valor fuera de rango) para que el LLM no invente; texto libre
  normalizado (`.lower()` + `.get()` con un valor por defecto); se prueban solas con `.invoke({...})` (clase 4).
- **Modelos:** se entrenan una vez y se guardan con `joblib`; se cargan una sola vez; el
  preprocesamiento es idéntico al del entrenamiento; las predicciones fijas se calculan una vez a nivel
  de módulo (clases 2 y 4).
- **Dividir por responsabilidad:** un lector por agente, pocas tools, prompt corto. Quien no tiene
  acceso a un dato no lo puede filtrar (mínimo privilegio) (clase 4, parte 1).
- **El código decide qué pasa entre agentes:** cortes determinísticos y "falla cerrada" (si falta lo
  esperado, no se pasa nada). Cada agente valida lo que recibe de otro (clase 4, partes 1 y 2).
- **Orquestador:** sus tools son los subagentes; el docstring es todo lo que sabe de cada uno;
  "llamá solo a los especialistas que hagan falta"; "incluí completo lo que te devolvieron" (clase 4, parte 2).
- **Medir costo:** callback `ContadorUso` (`on_llm_end`) colgado del `llm` compartido, que cuenta
  llamadas y tokens, también los de los subagentes. Sirve para la tabla de tiempos (clase 4).
- **Evaluar con varias corridas** y chequeos baratos con regex, no con otro LLM (clase 4, parte 1).
- **LangGraph:** `TypedDict` como estado; los nodos pueden ser funciones comunes sin LLM;
  diagrama con `draw_mermaid_png()` (si no hay internet, el texto Mermaid) (clase 4, parte 3).
- **SQL:** se escribe la tabla con un engine de escritura, que después se descarta; el agente abre
  `sqlite:///file:x.db?mode=ro&uri=true`; se muestra que un `DELETE` falla; los números del agente se
  verifican con pandas (clase 4, extra).
- **MCP:** el host elige qué tools del server le expone al agente (mínimo privilegio y tamaño de
  respuesta); en un `.py`, `asyncio.run(...)` envuelto en un `@tool` (clase 4, parte 4; `ejemplo_mcp.py`).
- **A2A** (si se hace): Starlette + uvicorn, Agent Card en `/.well-known/agent-card.json`,
  `message/send` en JSON-RPC, chequear `status.state == "completed"`, la descripción de la tool sale
  de la Agent Card (clase 4, parte 5).
- **RAG:** `paraphrase-multilingual-MiniLM-L12-v2`, chunks de 500 caracteres con 80 de solapamiento,
  `IndexFlatIP` sobre vectores normalizados, k=3; responder solo con el contexto y una frase fija
  cuando el dato no está ("No encontré esa información en el manual...") (clase 1).
- **Groq gratis:** ~8.000 tokens por minuto con gpt-oss-20b (lo dice la clase 4); `max_retries=5`.
- **Limitaciones para el informe** (clase 2): predecir sobre los mismos empleados del entrenamiento
  infla las probabilidades; la simulación muestra correlación, no causalidad.

## Estructura

```
TP_Hornero/
├── venv/          # Python 3.11.13, torch CPU + requirements.txt (NO va al zip)
├── etapa1/        # copia del código base + datos/etapa1
├── etapa2/        # copia del código base + datos/etapa1 y etapa2 + ejemplo_mcp.py
└── STATUS.md      # este archivo (NO va al zip)
```

`etapa2/marketing.py` y `etapa2/tools_marketing.py` son copias de `etapa1/agentes.py` y `etapa1/tools.py` (D8);
`marketing.py` solo cambia el import. Si se toca la etapa 1, volver a copiar:
`cp etapa1/tools.py etapa2/tools_marketing.py && sed 's/^from tools import (/from tools_marketing import (/' etapa1/agentes.py > etapa2/marketing.py`
(`etapa2/rag.py` también es una copia exacta).

La `GROQ_API_KEY` se toma de `MiM/AIAG/.env`: `load_dotenv()` sube por las carpetas hasta encontrarlo.

## Fases

| Fase | Estado |
|---|---|
| 0. Setup | ✅ 18/11, validado por Emiliano en la app |
| 1. Exploración y modelos | ✅ 18/11 (falta que Emiliano lo revise): `etapa2/preparacion.py` + `entrenar_modelos.py` → `modelos/attrition.joblib`, `ventas.joblib`, `metricas.json`. Hallazgos en `informe/notas_modelos.md`. Gráficos en la fase 5 |
| 2. Etapa 1 | ✅ 20/11, validada por Emiliano en la app: corridas oficiales 1 a 3 destaparon E1-22 a E1-29; la final (`e1_oficial_v4.log`) aprobó las 4 consultas y rechazó las 2 fuera de alcance, pero el push y el WhatsApp se rehicieron por E1-30 y E1-31 (`e1_oficial_v5.log` la 3, `e1_oficial_v6.log` la 4, 20/11 17:01 UTC). CSV = 8 originales + 4 piezas (respaldo `csv_e1_oficial_v6.csv`). |
| 3. Etapa 2 | funcionando (20/11), falta la corrida oficial de las 11 y la validación de Emiliano en la app. Primera ronda con Groq (logs `e2_*.log`). Bien, contra las referencias: 2 manual, 3 tasas (40,0 % Atención al cliente, 43,1 % CD Córdoba), 4 factores + 122→52, 5 Electro 179.728 / AMBA 74.894 / AMBA total 198.133, 6 cobre (−27,3 % aires, −24,4 % Electro, extrapola), 7 Cuyo 100 u (82-122, MCP), 8 Mundial rechazado, 10 manual + capacitaciones 1,40 vs 1,67, 11 borrar → «no puedo borrar». 1 Instagram aprobado y guardado en el CSV de la etapa 2. Arreglos E2-13 a E2-21 (abajo). 9 (tercera corrida, 23:24 UTC): recomienda AMBA (422 u), descarta Patagonia por extrapolar, push «Eficiencia que refresca» sin datos inventados, aprobado y guardado; 81,6 s, 10 llamadas, ~18.900 tokens; supply y marketing en vueltas separadas. Detalle menor: la «idea» de la pieza dice «mantén» (no se publica). Corrida oficial 21/11 14:52-15:05 UTC (`e2_oficial.log`, ~90K tokens): bien 1-8 y 10; la 11 salió con el rechazo genérico. Arreglos E2-22 a E2-25 y re-corrida de 2, 4, 9, 10 y 11 (`e2_oficial_b.log`, ~42K): todas bien en contenido; la 9 y la 10 perdían partes en el resumen del orquestador → E2-26 (probado sobre los logs, sin re-correr). CSV de la etapa 2: 8 originales + Instagram de la 1 (14:53) + push de la 9 «Aires eficientes para tu hogar» (15:14). Falta: validación de Emiliano en la app (con cupo o key de un compañero) |
| 4. Pruebas | en curso: corrida oficial de la etapa 2 hecha (21/11) y tabla de tiempos lista (abajo); faltan la validación de Emiliano en la app, el CSV consolidado y las capturas |
| 5. Informe (Google Doc → PDF) y entrega | borrador actualizado 21/11 (resumen, decisiones hasta E1-31 y E2-27, figuras 3 y 4, resultados y tiempos, consulta 11, limitaciones, anexo A.1); faltan: que Emiliano lo lea, recortar a 10 páginas, capturas (A.2), carátula, A2A o sacar la sección 7, PDF y zip final |

## Verificado contra los datos (2026-11-18)

- `renuncio` = Si en 28,3 % de los 2.000 empleados.
- `ventas.csv`: 200 semanas (2023-01-02 a 2026-10-26) × 6 regiones × 6 categorías = 7.200 filas.
- Región → ciudad (pronóstico): AMBA Buenos Aires, Centro Córdoba, Cuyo Mendoza, Litoral Rosario,
  NOA San Miguel de Tucumán, Patagonia Neuquén.
- `pronostico_clima.csv` cubre del 16 al 22/11/2026: es la semana en curso, no "la que viene".
- Historial: ya hay un email a Recurrentes "Recuperación: entrega con problemas" (2026-10-23),
  que importa para la consulta 2 de la etapa 1, y el push "Preparate para el calor" (2026-10-20),
  para la consulta 4.
- Manual RRHH 1.1: el híbrido aplica solo a roles de oficina; los de los CD son presenciales.
- Consulta 4 de la etapa 2: "acercar 10 puntos al mercado" = +10 con tope en 0 para quien está por
  debajo del mercado; sin cambio para quien está por encima.
- Conectividad: Groq, HuggingFace y Open-Meteo responden.
- Límites de la cuenta de Groq (18/11): 8.000 tokens por minuto, 1.000 pedidos por día y **200.000 tokens
  por día**, que es el que manda. No se reinicia a medianoche: es un balde que se recarga a ~2,3 tokens por
  segundo (~8.300 por hora), deducido de los 429 ("faltan 359 tokens → esperá 2 min 35 s"). El cupo es por
  modelo y por organización (cada key de un integrante es otra organización).
  Una consulta de la etapa 1 gasta ~7.000 tokens si sale en una versión y hasta ~18.000 con correcciones. El
  18/11 se agotó en la ronda 3. Hay que dosificar las pruebas y repartirlas con las keys de los otros integrantes.
- Si la app se cae con un segfault de pyarrow: `ARROW_DEFAULT_MEMORY_POOL=system` antes de `streamlit run`.

## Respuestas de referencia (para validar a los agentes en la fase 4)

Calculadas con pandas o con los modelos guardados, no con un LLM.

| Consulta | Referencia |
|---|---|
| E2-3 tasa de renuncia | Área: **Atención al cliente 40,0 %** (2.ª Operaciones 33,5 %). Centro: **CD Córdoba 43,1 %** (2.º CD Tucumán 37,6 %). Ojo: por cantidad absoluta ganan Operaciones (229) y CD AMBA (178), que **no** es lo que se pregunta |
| E2-4 factores + sueldos | Factores: sueldo vs. mercado, satisfacción con el clima, meses sin aumento, horas extra (después antigüedad y capacitaciones). Escenario: **122 activos en riesgo (u = 0,5) → 52 dejan de estarlo** |
| E2-5 ventas 2025 | **Electro, 179.728 unidades**; donde más vendió Electro: **AMBA (74.894)**. AMBA también es la región líder en total (198.133) |
| E2-6 cobre +20 % | Base = últimas 8 semanas: **Electro −24,4 %** (3.344 → 2.529 u/semana), **Aires −27,3 %** (419 → 304). Cantidades con el factor de Duan (sin él: 3.303 → 2.498 y 413 → 301) |
| E2-7 aires en Cuyo | Depende del pronóstico del momento. El 18/11, el MCP para el 23 al 29/11 en Mendoza dio 22,2 °C y 10,5 mm → **96 aires** (80 %: 78 a 118), 0 % frente a un clima normal para la época. Con el CSV (16 al 22/11): 98 |
| Propia: dónde empujar aires + push | 18/11, semana del 23 al 29/11: más volumen AMBA (446, +4 % vs. clima normal); mayor suba Patagonia (102, +378 %: 23,9 °C pronosticados contra 14,8 °C normales). Cambia con el pronóstico |

## Decisiones cerradas (18/11/2026)

| # | Decisión | Por qué |
|---|---|---|
| D1 | Attrition: **regresión logística** (LightGBM queda como alternativa descartada) | AUC CV 0,79 vs 0,75; test 0,81 vs 0,79; coeficientes interpretables |
| D2 | Ventas: **lineal sobre log(unidades) con interacciones** categoría × (región, clima, cobre) | Split temporal (últimas 40 semanas): MAPE 12 % vs 19 % de LightGBM. Cobre +20 % ≈ 11.400 USD/t > máximo histórico 11.160: LightGBM no extrapola |
| D3 | RRHH en **dos agentes**: políticas (solo el manual) y people analytics (base + modelo) | Mínimo privilegio, patrón de la clase 4 |
| D4 | Revisor que rechaza 2 veces: **no se guarda**; se devuelve la pieza con observaciones avisando que no quedó en el CSV. Toda pieza **aprobada se guarda siempre** | La consigna: la tool de escritura agrega "la pieza aprobada"; el E2 pide las piezas de todas las consultas |
| D5 | Consulta 4: "acercar 10 puntos" = +10 a quien está por debajo del mercado, con tope en 0. Consulta 5: categoría líder 2025 + región donde más vendió esa categoría (y se menciona la región líder total) | Interpretación explícita en la respuesta |
| D6 | Umbral de riesgo **0,5** por defecto; parámetro de la tool y siempre explícito en la respuesta (0,33 = F1 máx., va al informe como alternativa) | Intuitivo, el de la clase 2 |
| D7 | Riesgo calculado **solo sobre activos** (`renuncio = No`, 1.434); el modelo se entrena con los 2.000 | "Los que están fuera ya están fuera" (Emiliano). Limitación: son predicciones dentro de la muestra de entrenamiento |
| D8 | Marketing en la etapa 2: **copia** idéntica del módulo de la etapa 1; cada carpeta escribe su CSV y se consolidan al final | Cada carpeta corre sola |
| D9 | A2A: nice to have, solo si el 25/11 está todo lo obligatorio andando; candidato Supply chain | — |
| D10 | Entrega: **29/11/2026 23:59** | Confirmado por Emiliano |

## Decisiones de diseño de la etapa 1 (18/11/2026, para la sección 3 y 5 del informe)

| # | Decisión | Por qué / alternativa descartada |
|---|---|---|
| E1-1 | Grafo: planificador (agente + tools) → contexto (código) → redactor (agente) → validar (código) → revisor (agente) → guardar (código); hasta 4 versiones del redactor y 2 rechazos del revisor (D4) | 3 agentes en secuencia como pide la consigna; los nodos de código deciden qué recibe cada agente (clase 4). Con un tope único de 3 versiones, las correcciones baratas del validador se comían el margen del revisor |
| E1-2 | Solo el planificador ve el historial (`leer_historial`); el revisor recibe la pieza y el brief **sin el ángulo** (que nombra piezas del historial) | Requisito 1.9 y mínimo privilegio |
| E1-3 | RAG: tablas extraídas con `find_tables` y escritas fila por fila; chunks por sección numerada (no cada 500 caracteres) | Con el texto plano las columnas de la tabla de canales salen mezcladas ("Asunto de 50 no hasta 1 en el Un único botón"). Corte por sección = una regla no queda partida |
| E1-4 | Índice en `indices/<nombre>.faiss` + `.json` con el MD5 del PDF: si el PDF cambia, se rearma solo | La consigna pide borrarlo a mano; esto lo automatiza |
| E1-5 | Salida estructurada (`response_format` con pydantic) en los 3 agentes | El código lee campos fijos (canal, segmento, encabezado, legales...) en vez de parsear texto; 1 llamada por agente sin tools |
| E1-6 | Validador determinístico (`validar_pieza`): largos, emojis, hashtags, CTA, prohibidas, plazos, velocidad en Cuyo/NOA/Centro, competidores, mayúsculas, "!!", frases ≥ 25 palabras, legales y **datos que no estaban en el pedido** (%, $, fechas, N cuotas). Si falla, vuelve al redactor sin gastar al revisor | Chequeos baratos con regex en vez de otro LLM (clase 4). Prueba: sin contexto, gpt-oss-20b inventó "15 % de descuento exclusivo" en un push |
| E1-7 | `guardar_pieza` es una tool, pero la llama el código solo tras la aprobación; ningún agente la tiene | Un LLM no puede guardar una pieza no aprobada (falla cerrada) |
| E1-8 | Formato del texto guardado = el del historial (Asunto/Hola, {nombre}/[Botón]/legales; Título/Texto en push; hashtags al final en IG). Saludo del email y link de legales de WhatsApp los pone el código | Reglas fijas de la guía; evita reintentos por un formato |
| E1-9 | La fecha de hoy la inyecta el código en el pedido al planificador (con `fecha_de_hoy.invoke`) | Ahorra una llamada al LLM y sirve para el ángulo de temporada |
| E1-10 | Planificador con `ToolStrategy` (el brief es una tool más); redactor y revisor con `ProviderStrategy(strict=True)` y `extra="forbid"` | Groq no combina el modo JSON con tools. Sin strict, gpt-oss-20b a veces devolvía el esquema en vez de la pieza; strict exige `additionalProperties: false` |
| E1-11 | El revisor responde un checklist de 4 puntos (true/false) y **aprueba el código**: solo si todo es true | Con un "aprobada" libre, el LLM aprobaba con observaciones pendientes o rechazaba sin decir qué cambiar |
| E1-12 | `reasoning_effort`: "low" en planificador y redactor, "medium" en el revisor | Con "low", el revisor pidió "Sabés" en vez de "Sabemos" y un mail de contacto inventado. "medium" en el redactor no corregía mejor y generaba hasta 7.000 tokens de salida (cupo de 8.000 por minuto) |
| E1-13 | Voseo, contactos inventados y parecido con el historial (secuencias de 4 palabras, umbral 0,30) se chequean en código | El revisor LLM confundía la 1.ª persona con falta de voseo y dejó pasar "vuelve"; el redactor copió frases del email del 23/10. Umbral calibrado sobre las 8 piezas originales |
| E1-14 | El código limpia el brief antes del redactor: %, $, fechas y cuotas que no están en el pedido pasan a marcador | Cada agente valida lo que recibe de otro (clase 4). En las pruebas el planificador inventó "15 %" y "10 %" |
| E1-15 | El brief describe la pieza (mensaje principal como idea, sin características del producto); no la redacta | Ronda 2: el planificador escribió "tecnología inverter", "garantía extendida" y "mantén" (tuteo); el redactor lo copió y el revisor no podía marcarlo porque estaba en el brief, y exigía su frase textual |
| E1-16 | Fallas de generación de Groq: la tool se llama `brief` en minúsculas; reintentos ante `tool_use_failed` y `json_validate_failed`; si Groq devuelve en `failed_generation` un JSON que valida contra el esquema, se usa | Ronda 2: ante el pedido fuera de alcance, el modelo escribió el brief correcto como texto y los 3 reintentos (temperatura 0) dieron lo mismo |
| E1-17 | `responder()` recorre el grafo con `stream` en vez de `invoke` | Si algo falla en el medio, el detalle muestra hasta qué paso llegó |
| E1-18 | Si el texto del push pasa de 90 caracteres, el código lo corta en sus primeras oraciones completas que entran (`recortar_push`, en el nodo validar) y el revisor juzga el resultado | 19/11: en dos corridas de la consulta 4 ninguna de las 8 versiones bajó de 150 caracteres (166, 151, 151, 151 y 155 ×4): tras la primera corrección devolvía el mismo texto, también con un recorte concreto sugerido |
| E1-19 | En el push no se chequea el campo `cta` (no se publica: el formato es Título + Texto) | La guía escribe el CTA del push «Abre la app» (tuteo); el redactor lo copiaba y el validador lo rechazaba |
| E1-20 | Al revisor de un push se le dice que el llamado a la acción es tocar la notificación | 19/11: con el texto recortado, el revisor rechazó el push por «falta un llamado a la acción» y propuso «Descubre más» |
| E1-21 | El recorte del push saltea las oraciones que repiten el título (60 % o más de sus palabras de 4+ letras ya están en él); si sin ellas no entra nada, vuelve al recorte simple | 19/11: la consulta 4 se aprobó y guardó con Título «Aires que ahorran energía y dinero» y Texto «Aires acondicionados que ahorran energía y dinero.». Con el cambio, la misma pieza queda «Elegí el que mejor se adapte a tu hogar y disfrutá de un clima perfecto sin gastar de más.» (90 caracteres, sin errores de validación) |
| E1-22 | Rescate del brief fuera de alcance incompleto: si `failed_generation` trae `fuera_de_alcance: true` y le faltan campos, se completan vacíos y se valida (un brief en alcance incompleto se sigue descartando) | 19/11, corrida oficial: las dos consultas fuera de alcance terminaron en «hubo un error»: el modelo llamó a la tool solo con `fuera_de_alcance` y `motivo` y Groq rechazó la llamada por los campos que faltaban |
| E1-23 | El contador de uso cuenta también las llamadas que rechaza Groq (`on_llm_error`) y lo aclara: «N llamadas al LLM (M rechazadas por Groq, sin tokens informados)». En las dos etapas | 19/11: la consulta rescatada mostraba «0 llamadas al LLM» aunque el modelo había respondido |
| E1-24 | Los códigos de cupón (mayúsculas con letras y números, 4+ caracteres) son datos a confirmar: si no están en el pedido, error en la validación y marcador `[CÓDIGO]` en el brief | 19/11, corrida oficial: el WhatsApp de la consulta 3 se aprobó con «Usá el cupón NUEVO15», que no está en el pedido ni en la guía (la consigna prohíbe inventar códigos). Sin falsos positivos en las 8 piezas originales |
| E1-25 | Instagram y LinkedIn no llevan saludo con `{nombre}` (error en la validación; además, la descripción del campo `texto` lo dice) | 19/11, corrida oficial: el post de Instagram de la consulta 1 empezaba con «Hola, {nombre}:». La guía pide el saludo personalizado en el email; ninguna pieza original de Instagram o LinkedIn lo usa |
| E1-26 | El tuteo de la tabla `VOSEO` lo pasa a voseo el código (`pasar_a_voseo`, en el nodo validar, sobre encabezado, texto y CTA) y el revisor juzga el resultado; «usted» sigue siendo error. Se suman crea, explora, encuentra y accede | 19/11, segunda corrida oficial: el email de la consulta 2 no se guardó porque las 4 versiones repitieron «Visita la app» aunque la corrección nombraba la palabra; el WhatsApp se aprobó con «Crea momentos especiales», que no estaba en la lista |
| E1-27 | Si el planificador marca «promocion» y el pedido no trae descuento, cuotas, envío sin cargo, cupón u oferta, el código la cambia a «lanzamiento» (si el pedido dice nuevo/nueva o lanzamiento) o «contenido», y lo avisa en el detalle | 19/11, segunda corrida oficial: «anunciando la nueva línea de muebles de exterior» salió como promoción y el revisor rechazó dos veces por falta de legales |
| E1-28 | El prompt del revisor aclara que el canal y el segmento los eligió quien hizo el pedido y no se cuestionan | Misma consulta: el segundo rechazo dijo que «para clientes Premium el canal preferido es email» |
| E1-29 | En Instagram, si el código tiene que agregar el CTA de la guía y la última oración ya mandaba a la app o a la bio, la reemplaza en vez de sumar otra | 19/11, tercera corrida oficial: el post se aprobó terminando en «Descubrí la colección en la app. Encontralo en la app.» (el segundo lo agregó el código; el revisor recibe el CTA aparte y no vio la repetición) |
| E1-30 | Las formas de la tabla `VOSEO` que coinciden con la tercera persona del presente (visita, ahorra, crea, transforma, elige…) solo cuentan como tuteo al principio de una oración, que es donde va el imperativo (`AMBIGUAS`, `formas_de_tuteo`); tienes, prepárate y las demás sin doble lectura se corrigen en cualquier lugar. Costo: un imperativo con tú en medio de una oración («Mirá y descubre») ya no se detecta | 19/11, corrida final: el push se guardó con «Aire acondicionado que ahorrá energía» (E1-26 había pasado a voseo un «ahorra» en tercera persona). De las tres conversiones del día, las dos de «ahorra» estaban mal y la de «Visita» bien, y esa se sigue haciendo |
| E1-31 | Si el pedido da el beneficio (%, cuotas, sin cargo) y no habla de cupón ni de código, que la pieza los nombre es error de validación: es inventar cómo se aplica el descuento. En recuperación no aplica, porque ahí el pedido no trae beneficio y el cupón con [X]% es el que pide la guía | 19/11, corrida final: el WhatsApp se aprobó con «Usá el cupón al pagar» (en la primera corrida había inventado «NUEVO15») |

## Decisiones de diseño de la etapa 2 (18/11/2026)

| # | Decisión | Por qué / alternativa descartada |
|---|---|---|
| E2-1 | Orquestador (`create_agent`) cuyas tools son los 4 especialistas; puede llamar a más de uno | Patrón "orquestador con subagentes" de la clase 4: estructura distinta de la secuencial y combina áreas |
| E2-2 | People analytics y Supply chain delegan lo exploratorio en un subagente SQL propio (3 niveles: orquestador → especialista → SQL) | Requisito de agentes SQL; cada subagente ve una sola base |
| E2-3 | SQL: `empleados.db` y `ventas.db` separadas, abiertas con `mode=ro` (URI con `as_uri()`, sirve con espacios en la ruta). Del toolkit, solo `sql_db_query` y `sql_db_schema`; el esquema va en el prompt. Resultados recortados a 2.500 caracteres | `DELETE`/`DROP` fallan con "readonly database" y ventas.db no ve empleados (verificado). `sql_db_query_checker` gasta una llamada más por consulta; `sql_db_list_tables` no tiene parámetros (gpt-oss-20b los genera mal). Un `SELECT *` de ventas da ~450.000 caracteres |
| E2-4 | MCP: de las 8 tools del server se usa solo `get_weather_byDateTimeRange`. La respuesta (~100.000 caracteres hora por hora + instrucciones para el LLM) se resume en código a temperatura media/mín/máx y lluvia total de la semana que viene. Chequeo de hemisferio, 2 intentos y respaldo en el CSV avisando sus fechas | Mínimo privilegio y tamaño de respuesta (clase 4). Probado en las 6 regiones: 1,6 s cada una. Un primer intento falló por un corte de red: el respaldo se probó también forzando la falla |
| E2-5 | Tools de los modelos generales: `simular_escenario_attrition(variable, cambio, tipo_cambio, limite, nuevo_valor, filtros)` y `simular_escenario_ventas(variable, cambio, tipo_cambio, categorias, regiones, semanas_base)` | Requisito 2.8. `limite` resuelve "acercar al mercado" (D5) sin una tool especial |
| E2-6 | Ventas: el factor de Duan (1,012) corrige las cantidades absolutas; los % no cambian | Al volver de log a unidades la predicción queda en la mediana |
| E2-7 | Falla cerrada: si el orquestador no delegó en ningún especialista, el código devuelve el mensaje de rechazo | El sistema nunca responde con lo que "sabe" el LLM (capa extra de los límites de alcance) |
| E2-8 | La pieza de Marketing llega tal cual: si el orquestador la reescribe o la recorta, el código agrega la respuesta original | Es la que quedó guardada en el CSV |
| E2-9 | Los pasos de cada especialista se anotan en una `ContextVar`, no en una lista global | Dos sesiones de Streamlit no se mezclan, y los especialistas que corren en paralelo (threads) escriben en su consulta |
| E2-10 | Un solo `ContadorUso`, colgado también de los LLM de Marketing | El detalle muestra las llamadas y los tokens de toda la consulta |
| E2-12 | `prever_ventas_semana(categoria, regiones)`: pronóstico de las regiones en paralelo (MCP) + modelo, en una sola tool; compara con un clima normal para la época (mismas semanas de 2023-2025) | Para "dónde conviene empujar" el especialista hacía 6 pronósticos y 6 predicciones (12 llamadas a tools). Comparar contra las últimas 8 semanas (sep-oct) daba +1.216 % en Patagonia: cualquier semana de fin de noviembre parecía una explosión |
| E2-11 | Las tools aceptan texto libre: sin tildes, guion bajo = espacio, coincidencia parcial si es única, listas como texto separado por comas | Los LLM escriben "satisfaccion clima" o "aires, calefaccion"; con un error de validación, la llamada fallaba |
| E2-13 | El prompt SQL pide una consulta por dimensión cuando hay dos rankings («¿qué área? ¿y qué centro?») y people analytics le pasa la pregunta tal cual | Consulta 3: agrupó por área × centro y respondió la combinación (55 %) |
| E2-14 | Control de números en código (`controlar_numeros`): todo número de 4 cifras o más de la respuesta de un agente tiene que estar en la pregunta, en lo que devolvieron sus tools o en sus argumentos; si no, una vuelta de corrección. Los años quedan afuera | Consulta 5: supply chain inventó «AMBA, 152.321 unidades» y una explicación de crédito y heladeras |
| E2-15 | `explicar_factores_attrition` dice «si sube 1 unidad, las chances aumentan/disminuyen X %» | Consulta 4: «cada unidad más baja las chances» se leyó al revés en capacitaciones |
| E2-16 | Los prompts piden dar el sentido de cada factor (attrition y ventas) | Consulta 4: la respuesta perdió el sentido de las variables |
| E2-17 | `simular_escenario_ventas` escribe la diferencia («caen 115 unidades por semana») | Consulta 6: el LLM tomó el «después» (304) como la caída |
| E2-18 | `prever_ventas_semana` etiqueta «Pronóstico de la semana» y «Clima normal para la época», y avisa cuando la temperatura pronosticada sale del rango que registró la región (Patagonia: 24,5 °C contra un máximo de 21,9) | Consulta 7: leyó 22,2 °C normal como mínima e inventó 22,8; consulta 9: recomendó Patagonia por un +431 % extrapolado |
| E2-19 | Pedidos de borrar o modificar datos van al especialista del área; si el orquestador delega pero contesta con el rechazo fijo, va la respuesta del especialista | Consulta 11: se rechazaba con «solo temas internos de Hornero», que no es el motivo |
| E2-20 | Falla cerrada en `consultar_marketing`: si el pedido trae descuentos, cuotas, envío sin cargo, cupón o montos que no están en la pregunta original del usuario (`pregunta_usuario`), no se arma la pieza y vuelve al orquestador. El prompt del orquestador pide pasar el pedido con las palabras del usuario | Consulta 9: el orquestador agregó «Descuento: 15%» y salió un push con un 15 % inventado (se sacó del CSV) |
| E2-21 | El orquestador corrige números sin volver a llamar a los especialistas y llama a cada uno una sola vez | Consulta 9, segunda corrida: un falso positivo («2026») hizo que volviera a llamar a supply chain, que cambió de recomendación; 311 s y ~38.000 tokens |
| E2-22 | Si el orquestador no delega y el pedido es borrar o modificar datos (`PIDE_ESCRITURA`), el código responde que las bases son de solo lectura (`SOLO_LECTURA`) en vez del rechazo genérico | Corrida oficial (21/11): «Borrá de la base a los empleados que renunciaron» no se delegó y salió «queda fuera de lo que puedo responder»; en las corridas del 20/11 sí delegaba (variabilidad del modelo) |
| E2-23 | `buscar_politica`: si dos de los 3 resultados son del mismo tema del manual, suma el resto del tema | Corrida oficial, consulta 10: «capacitación» trajo 3.1, 3.2 y 3.4, y quedó afuera 3.3 Posgrados (50 % del costo, 18 meses de permanencia) |
| E2-24 | Prompt del orquestador: si preguntan dónde o qué conviene, la respuesta arranca con la recomendación en una frase y el número que la sostiene | Corrida oficial, consulta 9: tabla y push para AMBA, pero ninguna frase decía «conviene AMBA» |
| E2-25 | Prompt de people analytics: decir «chances», como la tool, y no «probabilidad» | Corrida oficial, consulta 4: «cada punto reduce la probabilidad un 52,7 %», cuando el porcentaje es sobre las chances (odds) |
| E2-26 | Si a la respuesta final del orquestador le falta algún número que dio un especialista (`cifras`), la respuesta pasa a ser la de los especialistas tal cual, con subtítulo por área, y queda un paso «Control de la respuesta (código)» en el detalle | Re-corridas del 21/11: en la 9 el orquestador sacó la tabla de regiones y el descarte de Patagonia, en la 10 la sección 3.2. Probado sobre los 16 logs del día: repone en 9 y 10 (re-corrida), y en 2 (números de sección) y 7 (pronóstico del MCP) de la oficial; en los otros 12 deja la del orquestador |
| E2-27 | El aviso de cupo dice cuál tope se agotó (diario o por minuto) y cuánto falta, leído del error de Groq (`aviso_sin_cupo` en `etapa1/agentes.py`, que la etapa 2 usa por `marketing`) | Decía siempre «probá en unos minutos», y con el cupo diario agotado la espera puede ser de más de una hora. README.md en la raíz para instalar y correr desde el repo |


## Tiempos de la etapa 2 (21/11, para el informe)

Corrida oficial (`e2_oficial.log`) y, para 2, 4, 9, 10 y 11, la re-corrida con E2-22 a E2-25 (`e2_oficial_b.log`).
60 s entre consultas; Groq gratis: 8.000 tokens por minuto.

| # | Consulta | Quién responde | Tiempo | Llamadas al LLM | Tokens (entrada + salida) |
|---|---|---|---|---|---|
| 1 | Instagram Premium aires | Marketing | 11,4 s | 6 | 8.094 + 1.290 |
| 2 | Remoto e internet | Políticas | 10,5 s | 4 | 3.771 + 424 |
| 3 | Tasa de renuncia por área y centro | People + SQL | 7,8 s | 7 | 7.960 + 600 |
| 4 | Factores y sueldos +10 | People | 5,3 s | 5 | 7.337 + 1.119 |
| 5 | Categoría y región 2025 | Supply + SQL (2 pedidos) | 54,4 s | 12 | 15.303 + 1.701 |
| 6 | Cobre +20 % | Supply | 1,6 s | 4 | 5.532 + 463 |
| 7 | Aires en Cuyo (MCP) | Supply | 5,4 s | 4 | 5.506 + 380 |
| 8 | Mundial 2022 | Rechazo | 0,2 s | 1 | 998 + 61 |
| 9 | Región + push | Supply + Marketing | 69,4 s | 10 | 16.138 + 1.826 |
| 10 | Manual y capacitaciones | Políticas + People + SQL | 13,0 s | 9 | 9.735 + 883 |
| 11 | Borrar empleados | Código (solo lectura) | 0,5 s | 1 | 1.028 + 80 |

Lectura: el tiempo sigue al número de llamadas y al tope de tokens por minuto. Con el balde lleno una pieza de
Marketing sale en 11 s (la 1, primera de la corrida); en la 5 el segundo pedido al subagente SQL esperó 22,9 s, y en
la 9 oficial el revisor tardó 22 s con una llamada rechazada por Groq. Los rechazos son una sola llamada. Ojo: en la 9
el paso «Contexto para el redactor» (solo código: 5 búsquedas de embeddings en la guía) tardó 24-28 s porque la
máquina estaba saturada (medido a las 15:20 UTC); el mismo paso tardó 2,4 s el 20/11 y
4,5 s en la 1. No es Groq.

Ojo: las pruebas de desarrollo escriben en `etapa1/datos/etapa1/contenidos_aprobados.csv`. **Antes de la corrida
oficial de la fase 4, restaurarlo** desde `../TP_Aplicaciones IA Generativa_2026/datos/etapa1/` (8 filas; fin de
fila `\r\n` y saltos internos `\n`, igual que el original).

Pendiente: integrantes y legajos (hacen falta recién para el informe).
