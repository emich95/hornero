"""
main.py: aplicación de Streamlit para probar los agentes.

Cómo correrla (desde esta carpeta, con el venv activado):
    streamlit run main.py

Respecto del original solo se agregó la barra lateral con las consultas de prueba (la lista
EJEMPLOS de agentes.py), para no tener que copiarlas a mano. El contrato con responder() no cambia.
"""

import streamlit as st

import agentes
from agentes import responder

st.set_page_config(page_title="Asistente Hornero", page_icon="🏠")
st.title("🏠 Asistente Hornero")
st.caption("TP Aplicaciones de IA Generativa: sistema multiagente")

if "historial" not in st.session_state:
    st.session_state.historial = []

with st.sidebar:
    st.subheader("Consultas de prueba")
    for i, ejemplo in enumerate(getattr(agentes, "EJEMPLOS", [])):
        if st.button(ejemplo, key=f"ejemplo_{i}", width="stretch"):
            st.session_state.pendiente = ejemplo


def mostrar_detalle(detalle: list[dict]):
    if not detalle:
        return
    with st.expander("Ver detalle (agentes y tools)"):
        for paso in detalle:
            st.markdown(f"**{paso['titulo']}**")
            st.text(paso["contenido"])


# Mostrar la conversación
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        mostrar_detalle(mensaje.get("detalle", []))

# Nueva pregunta (escrita o elegida en la barra lateral)
pregunta = st.chat_input("Escribí tu consulta...") or st.session_state.pop("pendiente", None)
if pregunta:
    st.session_state.historial.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resultado = responder(pregunta)
        st.markdown(resultado["respuesta"])
        mostrar_detalle(resultado.get("detalle", []))

    st.session_state.historial.append({
        "rol": "assistant",
        "contenido": resultado["respuesta"],
        "detalle": resultado.get("detalle", []),
    })
