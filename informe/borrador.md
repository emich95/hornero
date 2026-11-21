# Trabajo Práctico: Sistema multiagente para Hornero

<!-- Carátula (no cuenta para las 10 páginas): UTDT, Escuela de Negocios · Master in Management, Analytics & AI ·
Aplicaciones de IA Generativa, 2026 · Integrantes: 4 nombres completos con legajo · Fecha de entrega -->

## 1. Resumen

(pendiente: se escribe al final)

## 2. Etapa 1: Marketing

### 2.1 Arquitectura

Figura 1. Arquitectura de la etapa 1 (`diagramas/etapa1.png`).

El pedido entra a un StateGraph de LangGraph con seis nodos: tres son agentes y tres son pasos de código. Los tres agentes usan el mismo modelo, openai/gpt-oss-20b en Groq con temperatura 0, y se diferencian por el prompt, las tools y lo que reciben. Cada agente devuelve un objeto con campos fijos (salida estructurada con pydantic), así el código lee el canal, el segmento o el asunto sin tener que interpretar texto libre.

El planificador convierte el pedido en un brief: canal, segmento, situación (promoción, lanzamiento, recuperación de clientes o contenido), objetivo, mensaje principal, ángulo, llamado a la acción y los datos que trajo el pedido, como un porcentaje o una categoría. Es el único agente con tools: `leer_historial`, para ver qué recibió el segmento en las últimas semanas y elegir un ángulo que no lo repita, y `buscar_guia`, para consultar la guía cuando el pedido lo necesita. Si lo que se pide no es una pieza de marketing, lo marca como fuera de alcance y el grafo termina sin redactar nada.

Entre el planificador y el redactor hay un paso de código que decide qué recibe el redactor. Primero limpia el brief: todo porcentaje, precio, fecha o cantidad de cuotas que no estaba en el pedido original pasa a un marcador ([X]%, DD/MM). Después arma entre cuatro y siete consultas a la guía según el canal, el segmento y la situación (si hay un descuento, por ejemplo, agrega una sobre legales de promociones) y toma la sección más parecida a cada una. Por último, suma dos piezas ya aprobadas del mismo canal, de otros segmentos, para que el redactor copie el formato.

El redactor no tiene tools. Escribe con lo que le pasa el código y devuelve la idea, el título, el asunto (email) o título (push), el texto, el llamado a la acción, los legales, los hashtags y una línea de imagen sugerida.

La pieza pasa primero por un validador en código que controla las reglas de la guía que se pueden verificar sin un LLM: largo por canal, cantidad de emojis y hashtags, voseo, palabras prohibidas, menciones a plazos de entrega, legales obligatorios, links o contactos inventados, datos que no estaban en el pedido y parecido con piezas del historial. Si encuentra errores, la pieza vuelve al redactor con la lista de correcciones. Si no, la recibe el revisor, que responde un checklist de cuatro puntos tomado de la guía (un solo mensaje y un solo llamado a la acción, tono adecuado a la situación, sin promesas ni datos inventados, cumple el brief) y anota qué cambiar cuando algo no cumple. El código aprueba la pieza solo si los cuatro puntos dan verdadero, y en ese caso la agrega a `contenidos_aprobados.csv` con la tool `guardar_pieza`, en el mismo formato que las piezas que ya estaban.

El ciclo tiene dos topes: cuatro versiones del redactor y dos rechazos del revisor. Si se agotan, el sistema devuelve la última versión con lo que falta corregir y avisa que no la guardó. Los topes van por separado porque, con uno solo de tres versiones, las correcciones del validador, que son baratas, se comían el margen del revisor.

### 2.2 Agentes, tools y datos

Tabla 1. Qué recibe cada agente.

| Agente | Tools | Recibe | No recibe | Por qué |
|---|---|---|---|---|
| Planificador | `leer_historial`, `buscar_guia` | El pedido y la fecha de hoy | — | Es el único que necesita el historial completo, para no repetir mensajes |
| Redactor | Ninguna | El brief limpio, las secciones de la guía que eligió el código y dos piezas de ejemplo | El historial y el pedido original | Escribe solo con datos confirmados; los ejemplos son de otros segmentos para que no copie lo que ese segmento ya recibió |
| Revisor | Ninguna | La pieza separada en partes (asunto, cuerpo, botón, legales, hashtags), el brief sin el ángulo, el checklist y cómo hablarle al segmento | El historial y los ejemplos | El ángulo nombra piezas del historial; el revisor juzga la pieza terminada |

La escritura del historial no la tiene ningún agente: la hace el código después de la aprobación.

### 2.3 RAG sobre la guía de marca

La guía tiene diez secciones numeradas, todas cortas. Cortar el texto cada 500 caracteres, como en la clase 1, deja reglas partidas entre dos chunks, así que usamos un chunk por sección (once en total, con la introducción). La tabla de reglas por canal fue el caso más delicado: al extraer el texto plano del PDF, las columnas salían mezcladas («Asunto de 50 no hasta 1 en el Un único botón»). La extrajimos con `find_tables` de pdfplumber y la escribimos fila por fila, con el nombre de cada columna, por ejemplo: «Canal: Push | Largo máximo: Título de 40 caracteres; texto de 90 | Hashtags: no | Emojis: hasta 1».

Para los embeddings usamos paraphrase-multilingual-MiniLM-L12-v2, que corre localmente, sin llamar a una API, y está entrenado en español. El índice es un IndexFlatIP de FAISS sobre vectores normalizados, que equivale a la similitud coseno. Se guarda en disco junto con los textos de los chunks y el hash MD5 del PDF: si el PDF cambia, el índice se vuelve a armar solo, sin borrarlo a mano.

### 2.4 Decisiones de diseño y alternativas descartadas

Tabla 2. Decisiones de la etapa 1.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Los controles mecánicos los hace el código y el revisor se ocupa del criterio | Un revisor LLM que controle todo | En las pruebas, el revisor tomó la primera persona del plural («sabemos») como falta de voseo y dejó pasar un «vuelve». Con expresiones regulares esos errores se detectan siempre y no gastan una llamada al LLM |
| El revisor responde un checklist de verdadero o falso y la aprobación la decide el código | Un campo libre «aprobada» | Con el campo libre, el modelo aprobaba piezas con observaciones pendientes o rechazaba sin decir qué cambiar |
| `guardar_pieza` la llama el código, después de la aprobación | Darle la tool de escritura al revisor | Ningún agente puede guardar una pieza no aprobada, aunque se equivoque |
| El brief describe la pieza y no la redacta; el código borra los datos que no estaban en el pedido | Confiar en la instrucción de no inventar | En las pruebas, el planificador agregó «tecnología inverter», «garantía extendida» y descuentos de 15 % y 10 % que nadie había pedido. El redactor los copió y el revisor no podía objetarlos porque venían en el brief |
| Si el texto del push pasa de 90 caracteres, el código lo corta en sus primeras oraciones completas, salteando la que repite el título, y el revisor juzga el resultado | Pedirle al redactor que lo acorte | En dos corridas de la consulta 4, ninguna de las ocho versiones bajó de 150 caracteres: después de la primera corrección, el redactor devolvía el mismo texto, también cuando la corrección le proponía un recorte concreto |
| Salida estructurada: el brief se devuelve como una tool y la pieza y la revisión en JSON estricto | Texto libre y parseo | Groq no permite combinar el modo JSON con tools en la misma llamada. Sin el modo estricto, el modelo a veces devolvía el esquema en lugar de la pieza |

<!-- Fuera del informe por espacio (quedan en STATUS.md si hace falta): reasoning_effort "medium" solo en el revisor
(E1-12), rescate de failed_generation de Groq (E1-16), fecha inyectada por el código (E1-9), detección de parecido
con el historial con secuencias de 4 palabras y umbral 0,30 (E1-13). -->

## 3. Etapa 2: arquitectura

### 3.1 Un orquestador con especialistas como tools

Figura 2. Arquitectura de la etapa 2 (`diagramas/etapa2.png`).

La etapa 2 sigue el patrón de orquestador con subagentes de la clase 4. El orquestador es un agente de `create_agent` cuyas cuatro tools son los especialistas: `consultar_marketing`, `consultar_politicas_rrhh`, `consultar_people_analytics` y `consultar_supply_chain`. Cada tool recibe la parte de la consulta que le toca al área, corre el agente especialista y devuelve su respuesta. El orquestador no tiene acceso a ningún dato: decide a quién delegar, le pasa a cada especialista su parte con los datos que dio el usuario y arma la respuesta final con lo que le devolvieron. Si una parte depende de otra (por ejemplo, una pieza de marketing para la región que recomiende supply chain), llama primero al que da el dato y después al otro.

A diferencia de la etapa 1, donde el código fija el orden de los agentes, acá el orden lo decide el LLM en cada consulta. Por eso el código hace un control al final: si el orquestador no llamó a ningún especialista, la respuesta es el mensaje de rechazo, aunque el modelo haya contestado otra cosa; y si el orquestador recortó o reescribió una pieza de marketing, se agrega la pieza original, que es la que quedó guardada en el historial.

### 3.2 Especialistas, tools y datos

Tabla 3. Qué tiene cada especialista.

| Especialista | Tools | Datos y permiso |
|---|---|---|
| Marketing | El grafo completo de la etapa 1 | Guía de marca (índice FAISS) y `contenidos_aprobados.csv` (lectura, y escritura solo de piezas aprobadas) |
| Políticas de RRHH | `buscar_politica` (RAG, 3 fragmentos) | Solo el manual de RRHH, con un chunk por subsección numerada (20 en total) |
| People analytics | `predecir_riesgo_attrition`, `explicar_factores_attrition`, `simular_escenario_attrition`, `consultar_datos_empleados` | Modelo de attrition; la base de empleados solo a través de su agente SQL |
| Supply chain | `predecir_ventas`, `explicar_factores_ventas`, `simular_escenario_ventas`, `prever_ventas_semana`, `pronostico_semana`, `consultar_datos_ventas` | Modelo de ventas, server MCP de clima con `pronostico_clima.csv` como respaldo; la base de ventas solo a través de su agente SQL |
| Agente SQL de empleados | `sql_db_schema`, `sql_db_query` | `empleados.db`, abierta en solo lectura |
| Agente SQL de ventas | `sql_db_schema`, `sql_db_query` | `ventas.db`, abierta en solo lectura |

Marketing se integra como un área más: el módulo de la etapa 1 está copiado dentro de `etapa2/` y la tool del orquestador recorre el mismo grafo. Lo copiamos en lugar de importarlo desde `etapa1/` porque la consigna pide que cada carpeta corra por separado.

### 3.3 Agentes SQL

Cada CSV se vuelca una sola vez a su propia base SQLite (`empleados.db` y `ventas.db`) y la base se abre en modo de solo lectura (`mode=ro`). Lo verificamos: un `DELETE` o un `DROP` fallan con «readonly database», y el agente de ventas no ve la tabla de empleados. Del SQLDatabaseToolkit usamos solo `sql_db_query` y `sql_db_schema`, y el esquema va en el prompt; descartamos `sql_db_query_checker` porque suma una llamada al LLM por consulta. El prompt de cada agente lleva además las reglas del área: la tasa de renuncia se calcula como el promedio de `renuncio = 'Si'` y nunca se muestra el sueldo de un empleado en particular; en ventas, la semana es un texto con formato fecha y un año se filtra con `BETWEEN`. Los resultados se recortan a 2.500 caracteres, porque un `SELECT *` de la tabla de ventas devuelve unos 450.000.

### 3.4 Pronóstico con el server MCP de clima

El cliente es un `MultiServerMCPClient` que levanta `mcp_weather_server` por stdio. De las ocho tools del server usamos solo `get_weather_byDateTimeRange`, para la ciudad principal de cada región (Buenos Aires, Córdoba, Mendoza, Rosario, San Miguel de Tucumán y Neuquén) y la semana que viene, de lunes a domingo. Cada respuesta trae unos 100.000 caracteres con datos hora por hora e instrucciones dirigidas al LLM, así que el código no se la pasa al agente: la resume en cuatro números (temperatura media, mínima y máxima y lluvia total de la semana). Además verifica que la ciudad devuelta esté en el hemisferio sur, para descartar una ciudad homónima de otro país. Cada consulta al server tarda alrededor de 1,6 segundos, y las de varias regiones van en paralelo. Si el server falla dos veces, el sistema usa `pronostico_clima.csv` y lo dice en la respuesta, junto con las fechas de ese pronóstico, que no son las de la semana que viene. El segundo intento lo agregamos después de que una prueba fallara por un corte de red; el respaldo lo probamos forzando la falla.

La tool `prever_ventas_semana` junta el pronóstico y el modelo en un solo paso: trae el clima de las regiones pedidas, predice las unidades de la categoría y compara el clima con el normal para esa época del año (las mismas semanas de 2023 a 2025).

### 3.5 Decisiones de diseño y alternativas descartadas

Tabla 4. Decisiones de la etapa 2.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Orquestador que llama a los especialistas como tools | Un router que clasifica la consulta y la manda a un solo agente | Con un router, una consulta que combina dos áreas tiene que ir a una sola. El orquestador puede llamar a dos, y en orden si una parte depende de la otra |
| RRHH en dos agentes: políticas y people analytics | Un solo agente de RRHH | Mínimo privilegio: el que responde políticas ve solo el manual, y el que analiza renuncias, solo los datos y el modelo. Cada uno tiene un prompt más corto y menos tools entre las que elegir |
| Las preguntas sueltas a los datos las resuelve un agente SQL dentro de cada especialista | Una tool por tipo de pregunta, o las tools SQL dadas directo al especialista | La consigna pide un agente que escriba SQL. Como subagente, cada uno ve una sola base y lleva su esquema en el prompt; el especialista decide qué preguntar |
| Pronóstico y predicción en una sola tool (`prever_ventas_semana`) | Que el especialista encadene `pronostico_semana` y `predecir_ventas` | Para «¿en qué región conviene empujar los aires?», el especialista hacía seis pronósticos y seis predicciones: doce llamadas a tools en lugar de una |
| Comparar el pronóstico con el clima normal para la época | Compararlo con las últimas ocho semanas | En noviembre, contra septiembre y octubre, Patagonia daba +1.216 % de aires: cualquier semana de fin de noviembre parecía una explosión de demanda |
| El rechazo final lo decide el código si no hubo delegación | Confiar en que el orquestador rechace | El sistema nunca responde con lo que «sabe» el modelo, aunque el prompt falle |

## 4. Modelos de machine learning

Los dos modelos se entrenan con `etapa2/entrenar_modelos.py`, que guarda cada modelo en un `.joblib` y sus métricas en `modelos/metricas.json`. Los agentes no entrenan nada: cargan el modelo ya entrenado y lo usan a través de tools.

### 4.1 Attrition

El modelo estima la probabilidad de que un empleado renuncie (en los datos renunció el 28,3 % de los 2.000). Usa el área, el centro, el nivel y la modalidad como variables categóricas, y la antigüedad, el sueldo respecto del mercado, los meses sin aumento, las horas extra, la distancia, la última evaluación, las capacitaciones y la satisfacción con el clima laboral, estandarizadas. Dejamos afuera el sueldo bruto: depende del nivel, no mejora el AUC (0,7875 con él y 0,7882 sin él) y la posición frente al mercado ya está en otra variable.

Validamos de dos maneras: con una partición estratificada 80/20 y con validación cruzada estratificada de cinco folds sobre los 2.000 empleados.

Tabla 5. Modelos de attrition comparados.

| Modelo | AUC (validación cruzada) | AUC (test) | F1 con umbral 0,5 |
|---|---|---|---|
| Regresión logística (elegido) | 0,788 ± 0,033 | 0,814 | 0,54 |
| LightGBM | 0,749 ± 0,038 | 0,787 | 0,52 |

Elegimos la regresión logística porque predice mejor y se interpreta directamente: el signo de cada coeficiente dice en qué sentido actúa la variable. Los efectos en los datos son monótonos (la tasa de renuncia baja del 46 % al 14 % entre el quintil peor pago y el mejor pago respecto del mercado), que es lo que un modelo lineal capta bien.

Un empleado está «en riesgo» si su probabilidad es de 0,5 o más. Con ese umbral, en test el modelo marca al 20,5 % de los empleados, con precisión 0,65 y recall 0,47. El umbral que maximiza F1 es 0,33 (marca al 34 %, con precisión 0,55 y recall 0,67). Nos quedamos con 0,5 por ser el más fácil de explicar, pero la tool lo recibe como parámetro y la respuesta siempre dice cuál usó. El modelo se entrena con los 2.000 empleados y el riesgo se calcula solo sobre los 1.434 activos, porque los que ya renunciaron no están en riesgo. Como el modelo vio a esos empleados al entrenar, comparamos con predicciones fuera de muestra (de la validación cruzada): también dan 122 activos en riesgo.

Para interpretarlo medimos la importancia de cada variable como la caída del AUC en test al desordenarla, y su sentido con el odds ratio.

Tabla 6. Qué explica las renuncias.

| Variable | Caída del AUC | Efecto sobre las chances de renunciar |
|---|---|---|
| Sueldo respecto del mercado | 0,060 | −4,5 % por punto |
| Satisfacción con el clima laboral (1 a 5) | 0,055 | −53 % por punto |
| Meses sin aumento | 0,038 | +5,2 % por mes |
| Horas extra por mes | 0,025 | +5,2 % por hora |
| Antigüedad | 0,014 | −8,3 % por año |
| Capacitaciones en el último año | 0,013 | −21 % por capacitación |
| Distancia al trabajo | 0,005 | +2,3 % por km |

El área suma poco una vez que se controla por lo anterior (Atención al cliente es la de mayor riesgo), y el centro y la modalidad casi nada, porque se superponen con el área: Logística y Operaciones son totalmente presenciales y trabajan en los centros de distribución.

### 4.2 Ventas

El modelo predice las unidades vendidas por semana de cada categoría en cada región. Es una regresión lineal sobre el logaritmo de las unidades, con un nivel propio para cada combinación de categoría y región y un efecto propio de cada categoría para la temperatura, la lluvia y el precio del cobre. No sumamos variables de calendario porque los datos no tienen tendencia ni estacionalidad que el clima no explique: el residuo promedio por mes da prácticamente cero.

Como son series de tiempo, validamos con una partición temporal: entrenamos con las primeras 160 semanas (hasta enero de 2026) y evaluamos con las últimas 40.

Tabla 7. Modelos de ventas comparados (test, últimas 40 semanas).

| Modelo | MAE (unidades) | MAPE | R² |
|---|---|---|---|
| Lineal sobre log con interacciones por categoría (elegido) | 22,8 | 12,2 % | 0,980 |
| Lineal sobre log con todas las interacciones de a pares | 23,1 | 12,2 % | 0,980 |
| LightGBM | 26,5 | 18,6 % | 0,975 |

Además de predecir mejor, el modelo lineal puede extrapolar. El cobre está en 9.470 USD por tonelada y una suba del 20 % lo lleva a unos 11.400, por encima del máximo histórico de 11.160. Un modelo de árboles no puede ir más allá de lo que vio y devolvería casi la misma venta. Entre los dos lineales elegimos el más simple, porque predicen igual. El error varía por categoría: el MAPE va de 8,5 % en Electro a 17,3 % en Aires acondicionados.

Tabla 8. Efecto de cada factor sobre las unidades vendidas.

| Categoría | +1 °C | +10 mm de lluvia | +1.000 USD/t de cobre |
|---|---|---|---|
| Aires acondicionados | +18,4 % | 0 % | −15,1 % |
| Calefacción | −14,7 % | 0 % | −1,7 % |
| Jardín | +6,1 % | −11,4 % | −0,9 % |
| Electro | 0 % | 0 % | −13,6 % |
| Cocina | 0 % | 0 % | 0 % |
| Muebles | 0 % | 0 % | +1,0 % |

La temperatura mueve Aires, Calefacción y Jardín, la lluvia solo Jardín, y el cobre pega en Electro y Aires. Cocina y Muebles dependen solo de la región; AMBA vende más del doble que la segunda región en todas las categorías. Al volver del logaritmo a unidades, la predicción queda en la mediana, un poco por debajo de la media, así que corregimos las cantidades con el factor de Duan (1,012); los porcentajes no cambian.

### 4.3 Tools de predicción, interpretación y simulación

Cada modelo tiene tres tools generales. La de predicción recibe filtros opcionales: `predecir_riesgo_attrition` cuenta y lista a los activos en riesgo por área, centro, nivel o modalidad, y `predecir_ventas` estima las unidades de una categoría y región para un clima y un precio del cobre dados, con un rango del 80 %; si no se le pasan, usa el promedio de las últimas ocho semanas. La de interpretación devuelve la tabla de factores, con su peso y su sentido. La de simulación es una sola por modelo y recibe la variable y el cambio, en lugar de una tool por variable: `simular_escenario_attrition(variable, cambio, tipo_cambio, limite, ...)` y `simular_escenario_ventas(variable, cambio, tipo_cambio, categorias, regiones)`. El parámetro `limite` resuelve pedidos como «acercar los sueldos 10 puntos al mercado» (+10 a quien está por debajo, sin pasar de 0) sin una tool especial, y los valores simulados se mantienen dentro del rango posible de cada variable (la satisfacción no pasa de 5, por ejemplo).

Con esas tools se responden las consultas 4 y 6. Si se acercan los sueldos 10 puntos al mercado, de los 122 activos en riesgo dejan de estarlo 52 (43 %). Si el cobre sube 20 %, sobre la base de las últimas ocho semanas Electro pasa de 3.344 a 2.529 unidades por semana (−24,4 %) y Aires acondicionados de 419 a 304 (−27,3 %).

## 5. Límites de alcance y seguridad

En la etapa 1, el planificador decide si el pedido es una pieza de marketing de texto. Si no lo es, marca el brief como fuera de alcance y el grafo termina antes del redactor, sin gastar más llamadas. Para que no se inventen datos hay tres barreras: la instrucción explícita en el prompt de los tres agentes, la limpieza del brief (lo que no estaba en el pedido pasa a marcador) y el validador, que rechaza porcentajes, precios, fechas o cuotas que no vinieron en el pedido, y mails, teléfonos o sitios inventados. La única escritura del sistema, agregar una pieza al historial, la hace el código y solo con una pieza aprobada.

En la etapa 2 los límites están en tres capas:

1. Prompt del orquestador. Define el alcance (marketing, políticas de RRHH, datos y riesgo de renuncia, ventas y clima) y le pide responder un mensaje de rechazo fijo, sin llamar a ningún especialista, ante cualquier otra cosa. Si una consulta mezcla una parte válida con otra fuera de alcance, responde la primera y aclara que la segunda no la puede responder. La consulta 8 («¿Quién ganó el Mundial 2022?») se rechaza con una sola llamada al LLM.
2. Control en código. Si el orquestador no llamó a ningún especialista, la respuesta es el mensaje de rechazo, diga lo que diga el modelo. Así el sistema nunca contesta con lo que el LLM sabe por su cuenta, ni siquiera cuando el prompt falla.
3. Mínimo privilegio. Cada especialista ve solo los datos de su área (Tabla 3) y el orquestador no ve ninguno. Las bases están separadas por área y abiertas en solo lectura, de modo que un agente SQL no puede modificar datos aunque lo intente, ni leer los de otra área. El agente SQL de empleados tiene además la regla de no mostrar el sueldo de una persona en particular.

<!-- PENDIENTE fase 4: resultado de la consulta propia «Borrá de la base a los empleados que renunciaron» (qué hizo el
sistema y en qué capa se frenó). -->

## 6. Resultados y tiempos de respuesta

(pendiente: fase 4)

## 7. Extra A2A

(si se hace)

## 8. Limitaciones conocidas

- El modelo. gpt-oss-20b es un modelo chico para coordinar agentes. A veces no sigue una corrección (el caso del push), devuelve la salida estructurada como texto o necesita reintentos. Lo compensamos con controles en código, salida estructurada y el rescate de la respuesta que Groq devuelve cuando falla la validación del esquema, pero con un modelo más grande varios de esos parches sobrarían.
- Los límites de Groq. El plan gratuito permite 8.000 tokens por minuto y 200.000 por día por modelo. Una consulta de marketing consume entre 8.000 y 12.000 tokens, así que varias seguidas generan esperas por el límite por minuto, que se ven en los tiempos, y alcanzan para unas veinte consultas de marketing por día.
- Los datos. Son sintéticos y los efectos son limpios. Con datos reales habría más ruido, más variables y relaciones menos estables.
- Asociación no es causalidad. Las simulaciones muestran qué asocia el modelo, no qué pasaría si se interviene: que el mejor sueldo se asocie con menos renuncias no garantiza que una suba las reduzca en esa medida.
- Extrapolación. El escenario de cobre +20 % queda por encima del máximo histórico; el resultado supone que la relación sigue siendo la misma fuera del rango observado.
- El umbral de riesgo. Con 0,5, el modelo de attrition detecta menos de la mitad de los que renuncian (recall 0,47). Si al negocio le importa más no dejar escapar casos, conviene bajar el umbral.
- El pronóstico depende de un servicio externo. Si el server MCP falla, el respaldo en CSV es de otra semana; el sistema lo avisa, pero la predicción ya no es de la semana que viene.
- Las piezas. Los legales salen con marcadores (DD/MM, [X]%) que tiene que completar el equipo, el recorte automático del push puede dejar afuera parte del mensaje y el control de parecido con el historial compara secuencias de palabras, así que no detecta una paráfrasis.
- Código duplicado. El módulo de Marketing está copiado en las dos carpetas para que cada una corra sola; un cambio en uno hay que llevarlo al otro.

## Anexo

(pendiente: capturas de la app y conversaciones de ejemplo)
