import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Ensayo de Calentamiento 600A")

st.title("Estabilidad Térmica - Inyección 600A")

# 1. Parámetros Generales (Estáticos)
st.subheader("Condiciones de Ensayo")
col1, col2, col3 = st.columns(3)
t_amb = col1.number_input("Temp Ambiente Inicial (°C)", value=25.0)
r_ini = col2.number_input("R Contacto Inicio (µΩ)", value=0.0)
r_fin = col3.number_input("R Contacto Fin (µΩ)", value=0.0)

# Inicializar dataframe en Session State si no existe
if 'df_ensayo' not in st.session_state:
    # Columnas: Hora y las 9 posiciones
    cols = ['Hora (HH:MM)', 'Ua', 'Ub', 'Uc', 'Va', 'Vb', 'Vc', 'Wa', 'Wb', 'Wc']
    st.session_state.df_ensayo = pd.DataFrame(columns=cols)
    # Fila vacía para empezar
    st.session_state.df_ensayo.loc[0] = ['14:00'] + [t_amb]*9

st.subheader("Registro de Mediciones")
st.markdown("Cargá la hora en formato `HH:MM` y las temperaturas. El tiempo transcurrido se calcula solo.")

# 2. Ingreso de datos (Editor interactivo)
df_editado = st.data_editor(st.session_state.df_ensayo, num_rows="dynamic", use_container_width=True)

# 3. Procesamiento y Cálculo de Tiempo
def calcular_minutos(hora_str, hora_base_str):
    formato = '%H:%M'
    try:
        t_actual = datetime.strptime(hora_str, formato)
        t_base = datetime.strptime(hora_base_str, formato)
        # Manejo de cruce de medianoche
        if t_actual < t_base:
            return (t_actual - t_base).seconds / 60 + 1440
        return (t_actual - t_base).seconds / 60
    except:
        return np.nan

if len(df_editado) > 0:
    hora_cero = df_editado['Hora (HH:MM)'].iloc[0]
    df_editado['t_min'] = df_editado['Hora (HH:MM)'].apply(lambda x: calcular_minutos(x, hora_cero))
    
    # Limpiar datos incompletos para el modelo
    df_modelo = df_editado.dropna(subset=['t_min'])

# 4. Ajuste de Curva y Proyección
def modelo_calentamiento(t, delta_T, tau):
    return t_amb + delta_T * (1 - np.exp(-t / tau))

st.subheader("Análisis de Estabilidad y Proyección")
if len(df_modelo) > 2: # Se necesitan al menos 3 puntos para ajustar la curva
    fig = go.Figure()
    
    # Tiempo extendido para proyectar el futuro (ej. 300 minutos)
    t_proyeccion = np.linspace(0, max(df_modelo['t_min']) + 120, 100)
    
    # Analizamos, por ejemplo, el sensor más caliente (ej. U-A) o hacemos un loop
    sensor_critico = 'U-A'  # Podés iterar sobre todos
    
    t_data = df_modelo['t_min'].values
    T_data = df_modelo[sensor_critico].values
    
    try:
        # Ajuste no lineal
        popt, pcov = curve_fit(modelo_calentamiento, t_data, T_data, p0=[50, 60], bounds=(0, [200, 500]))
        delta_T_fit, tau_fit = popt
        
        # Graficar puntos reales
        fig.add_trace(go.Scatter(x=t_data, y=T_data, mode='markers', name=f'Datos Reales {sensor_critico}', marker=dict(size=10)))
        
        # Graficar proyección
        T_proy = modelo_calentamiento(t_proyeccion, delta_T_fit, tau_fit)
        fig.add_trace(go.Scatter(x=t_proyeccion, y=T_proy, mode='lines', name=f'Proyección (Tau={tau_fit:.1f} min)', line=dict(dash='dash')))
        
        # Marcar los 5 Tau
        cinco_tau = 5 * tau_fit
        T_cinco_tau = modelo_calentamiento(cinco_tau, delta_T_fit, tau_fit)
        fig.add_vline(x=cinco_tau, line_width=1, line_dash="dot", line_color="red", annotation_text=f"5 Tau ({cinco_tau:.0f} min)")
        
        # Evaluar criterio de 1°C/hr
        # Derivada: dT/dt = (Delta_T / tau) * exp(-t/tau). Lo pasamos a °C/hr multiplicando por 60
        derivada_actual = (delta_T_fit / tau_fit) * np.exp(-t_data[-1] / tau_fit) * 60
        
        st.metric(label="Variación actual proyectada (°C/hr)", value=f"{derivada_actual:.2f} °C/hr")
        if derivada_actual <= 1.0:
            st.success("¡Criterio de estabilidad térmica alcanzado! Variación ≤ 1°C/hr.")
        else:
            st.warning("Aún no se alcanza la estabilidad.")
            
    except Exception as e:
        st.error("No hay suficientes datos o dispersión para ajustar la curva todavía.")

    fig.update_layout(xaxis_title="Tiempo Transcurrido (min)", yaxis_title="Temperatura (°C)")
    st.plotly_chart(fig, use_container_width=True)
