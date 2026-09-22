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
col1, col2 = st.columns(2)
r_ini = col1.number_input("R Contacto Inicio (µΩ)", value=0.0)
r_fin = col2.number_input("R Contacto Fin (µΩ)", value=0.0)

# Inicializar dataframe en Session State si no existe
if 'df_ensayo' not in st.session_state:
    # Agregada columna 'Temp Amb' y nombres actualizados
    cols = ['Hora (HH:MM)', 'Temp Amb', 'Ua', 'Ub', 'Uc', 'Va', 'Vb', 'Vc', 'Wa', 'Wb', 'Wc']
    st.session_state.df_ensayo = pd.DataFrame(columns=cols)
    # Fila vacía para empezar (se puede dejar vacía o poner un valor de ejemplo)
    st.session_state.df_ensayo.loc[0] = ['14:00', 25.0] + [25.0]*9

st.subheader("Registro de Mediciones")
st.markdown("Cargá la hora en formato `HH:MM` y las temperaturas. Si no querés medir la Temp Ambiente en cada paso, dejala en blanco.")

# 2. Ingreso de datos
df_editado = st.data_editor(st.session_state.df_ensayo, num_rows="dynamic", use_container_width=True)

# 3. Procesamiento y Cálculo de Tiempo
def calcular_minutos(hora_str, hora_base_str):
    formato = '%H:%M'
    try:
        t_actual = datetime.strptime(str(hora_str).strip(), formato)
        t_base = datetime.strptime(str(hora_base_str).strip(), formato)
        if t_actual < t_base:
            return (t_actual - t_base).seconds / 60 + 1440
        return (t_actual - t_base).seconds / 60
    except:
        return np.nan

if len(df_editado) > 0:
    hora_cero = df_editado['Hora (HH:MM)'].iloc[0]
    df_editado['t_min'] = df_editado['Hora (HH:MM)'].apply(lambda x: calcular_minutos(x, hora_cero))
    
    # Limpiar filas donde no se pudo calcular el tiempo (datos incompletos)
    df_modelo = df_editado.dropna(subset=['t_min']).copy()

# 4. Ajuste de Curva y Proyección
st.subheader("Análisis de Estabilidad y Proyección")

# Selección dinámica del sensor a graficar/analizar
sensores = ['Ua', 'Ub', 'Uc', 'Va', 'Vb', 'Vc', 'Wa', 'Wb', 'Wc']
sensor_critico = st.selectbox("Seleccioná el sensor para calcular la estabilidad:", sensores)

if len(df_modelo) > 2:
    fig = go.Figure()
    
    t_proyeccion = np.linspace(0, max(df_modelo['t_min']) + 120, 100)
    
    # Datos del sensor elegido
    # pd.to_numeric fuerza a que sean números (por si hay algún espacio al tipear)
    t_data = df_modelo['t_min'].values
    T_data = pd.to_numeric(df_modelo[sensor_critico]).values
    
    # Obtenemos la temperatura inicial del sensor elegido para arrancar la curva desde ahí
    T_inicial = T_data[0]
    
    # Redefinimos el modelo para que arranque desde la temperatura de la primera medición
    def modelo_calentamiento(t, delta_T, tau):
        return T_inicial + delta_T * (1 - np.exp(-t / tau))
    
    # Graficar Temp Ambiente como referencia (si hay datos válidos)
    df_modelo['Temp Amb'] = pd.to_numeric(df_modelo['Temp Amb'], errors='coerce')
    if not df_modelo['Temp Amb'].isna().all():
        fig.add_trace(go.Scatter(
            x=df_modelo['t_min'], 
            y=df_modelo['Temp Amb'], 
            mode='lines+markers', 
            name='Temp Ambiente', 
            line=dict(color='gray', dash='dot')
        ))

    try:
        # Ajuste matemático
        popt, pcov = curve_fit(modelo_calentamiento, t_data, T_data, p0=[50, 60], bounds=(0, [200, 500]))
        delta_T_fit, tau_fit = popt
        
        # Graficar puntos reales del sensor
        fig.add_trace(go.Scatter(
            x=t_data, y=T_data, 
            mode='markers+lines', 
            name=f'Datos {sensor_critico}', 
            marker=dict(size=8)
        ))
        
        # Graficar proyección
        T_proy = modelo_calentamiento(t_proyeccion, delta_T_fit, tau_fit)
        fig.add_trace(go.Scatter(
            x=t_proyeccion, y=T_proy, 
            mode='lines', 
            name=f'Proyección (Tau={tau_fit:.1f} min)', 
            line=dict(dash='dash', color='blue')
        ))
        
        # Marcar 5 Tau
        cinco_tau = 5 * tau_fit
        fig.add_vline(x=cinco_tau, line_width=1, line_dash="dot", line_color="red", annotation_text=f"5 Tau ({cinco_tau:.0f} min)")
        
        # Evaluar criterio de 1°C/hr (Derivada)
        derivada_actual = (delta_T_fit / tau_fit) * np.exp(-t_data[-1] / tau_fit) * 60
        
        st.metric(label=f"Variación actual ({sensor_critico})", value=f"{derivada_actual:.2f} °C/hr")
        if derivada_actual <= 1.0:
            st.success("¡Criterio de estabilidad térmica alcanzado! Variación ≤ 1°C/hr.")
        else:
            st.warning("Aún no se alcanza la estabilidad.")
            
    except Exception as e:
        st.error(f"Faltan datos o dispersión para ajustar la curva del sensor {sensor_critico}.")

    fig.update_layout(xaxis_title="Tiempo (min)", yaxis_title="Temperatura (°C)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Cargá al menos 3 mediciones del tiempo para que el modelo pueda calcular la proyección.")
