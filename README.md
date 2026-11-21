# TP Hornero: sistema multiagente

TP grupal de Aplicaciones de IA Generativa (MiM UTDT, 2026). Hay dos apps de Streamlit, una por etapa:

- `etapa1/`: Marketing. Arma piezas de texto (Instagram, email, WhatsApp, push, LinkedIn) siguiendo la guía de marca y las guarda en el historial.
- `etapa2/`: orquestador con cuatro especialistas: Marketing (el de la etapa 1), Políticas de RRHH, People analytics y Supply chain.

El estado del trabajo y las decisiones de diseño están en `STATUS.md`. El borrador del informe está en `informe/`.

## Qué hace falta

- Python 3.11 (lo probamos con 3.11.13).
- Una API key de Groq propia, que se saca gratis en console.groq.com, en la sección API Keys.
- Internet: la app usa Groq; la primera vez baja de HuggingFace el modelo de embeddings (unos 470 MB), y el pronóstico del clima sale de Open-Meteo.

## Instalación

Una sola vez, desde la carpeta del repo.

Mac o Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r etapa2/requirements.txt
cp etapa2/.env.example .env
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r etapa2\requirements.txt
copy etapa2\.env.example .env
```

Después abrí `.env` y reemplazá `tu_api_key_de_groq` por tu key. Ese archivo no se sube al repo (está en `.gitignore`). Un solo `.env` en la carpeta del repo sirve para las dos etapas.

Los `requirements.txt` de las dos etapas son iguales. En Linux, `sentence-transformers` instala torch con soporte de GPU, que pesa varios GB. Si no tenés GPU, instalá antes la versión liviana:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Cómo correr cada etapa

Con el entorno activado:

```bash
cd etapa1
streamlit run main.py
```

Para la etapa 2, lo mismo desde `etapa2/`. Se abre en el navegador, en http://localhost:8501. En la barra lateral están las consultas de prueba como botones, y debajo de cada respuesta, en «Ver detalle», qué agentes y tools actuaron, con tiempos y tokens.

La primera vez tarda más, porque baja el modelo de embeddings. Los índices de los PDF, las bases SQLite y los modelos entrenados ya vienen en el repo. Si cambia un PDF o un CSV, los índices y las bases se rearman solos. Para volver a entrenar los modelos: `python entrenar_modelos.py` desde `etapa2/`.

## El cupo de Groq

El plan gratis da 8.000 tokens por minuto y 200.000 por día por cuenta, y se va recargando de a poco. Una vuelta completa de las 11 consultas de la etapa 2 gasta unos 90.000 tokens, y una pieza de Marketing entre 8.000 y 16.000. Si se agota, la app avisa cuál cupo se agotó y cuánto falta para volver a probar. Conviene que cada uno use su propia key.

## Las piezas que se aprueban

Cada pieza que aprueba Marketing se suma a `datos/etapa1/contenidos_aprobados.csv` de la etapa en la que corriste. Ese archivo es parte de la entrega y ya tiene las piezas de las corridas oficiales. Antes de hacer commit, mirá `git status`: si el CSV cambió solo por tus pruebas y no lo hablamos, descartalo con

```bash
git checkout -- etapa1/datos/etapa1/contenidos_aprobados.csv etapa2/datos/etapa1/contenidos_aprobados.csv
```

## Si algo falla

- La app se cae al arrancar con un error de memoria de pyarrow (segmentation fault): corré `ARROW_DEFAULT_MEMORY_POOL=system streamlit run main.py`.
- El puerto 8501 está ocupado: `streamlit run main.py --server.port 8502`.
- «No pude responder: se agotó el cupo...»: esperá el tiempo que indica o usá otra key.
- Algo no anda con versiones nuevas de las librerías: estas son las que probamos.

| Librería | Versión |
|---|---|
| langchain | 1.5.0 |
| langchain-groq | 1.2.0 |
| langgraph | 1.2.22 |
| langchain-community | 0.4.2 |
| langchain-mcp-adapters | 0.3.2 |
| mcp_weather_server | 0.6.1 |
| groq | 1.7.0 |
| streamlit | 1.67.0 |
| sentence-transformers | 6.3.0 |
| faiss-cpu | 1.16.0 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| SQLAlchemy | 2.1.9 |
