import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Ensayo de Calentamiento 600A")
st.title("Estabilidad Térmica - Inyección 600A")

# 1. Parámetros Generales
st.subheader("Condiciones de Ensayo")
col1, col2 = st.columns(2)
r_ini = col1.number_input("R Contacto Inicio (µΩ)", value=0.0)
r_fin = col2.number_input("R Contacto Fin (µΩ)", value=0.0)

# Inicialización
sensores = ['Ua', 'Ub', 'Uc', 'Va', 'Vb', 'Vc', 'Wa', 'Wb', 'Wc']
if 'df_ensayo' not in st.session_state:
    cols = ['Hora (HH:MM)', 'Temp Amb'] + sensores
    st.session_state.df_ensayo = pd.DataFrame(columns=cols)
    st.session_state.df_ensayo.loc[0] = ['14:00', 25.0] + [25.0]*9

st.subheader("Registro de Mediciones")
df_editado = st.data_editor(st.session_state.df_ensayo, num_rows="dynamic", use_container_width=True)

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
    df_modelo = df_editado.dropna(subset=['t_min']).copy()
    
    # Asegurar que todas las columnas de sensores sean numéricas
    for s in sensores:
        df_modelo[s] = pd.to_numeric(df_modelo[s], errors='coerce')

# ==========================================
# CÁLCULOS DINÁMICOS Y BÚSQUEDA DEL PEOR CASO
# ==========================================
st.subheader("Análisis de Estabilidad y Proyección")

sensor_por_defecto = "Ua"
max_temp_actual = 0
delta_max_medido = 0
delta_sensor = ""

if len(df_modelo) > 1:
    # 1. Encontrar el sensor más caliente en la ÚLTIMA medición
    ultima_fila = df_modelo.iloc[-1]
    sensor_por_defecto = ultima_fila[sensores].astype(float).idxmax()
    max_temp_actual = ultima_fila[sensor_por_defecto]
    
    # 2. Calcular la mayor diferencia de T entre el último punto y el anterior
    fila_anterior = df_modelo.iloc[-2]
    diferencias = ultima_fila[sensores].astype(float) - fila_anterior[sensores].astype(float)
    delta_sensor = diferencias.idxmax()
    delta_max_medido = diferencias[delta_sensor]
    tiempo_transcurrido = ultima_fila['t_min'] - fila_anterior['t_min']

# Selector de sensor (pre-seteado en el más crítico)
try:
    idx_defecto = sensores.index(sensor_por_defecto)
except:
    idx_defecto = 0
sensor_critico = st.selectbox("Sensor bajo análisis:", sensores, index=idx_defecto)

# Mostrar Delta Medido (paso a paso)
if len(df_modelo) > 1:
    st.info(f"**Mayor incremento en último intervalo:** {delta_max_medido:.1f} °C en el sensor {delta_sensor} (pasaron {tiempo_transcurrido:.0f} min desde la lectura anterior).")

# ==========================================
# AJUSTE Y GRÁFICO
# ==========================================
if len(df_modelo) > 2:
    fig = go.Figure()
    t_proyeccion = np.linspace(0, max(df_modelo['t_min']) + 120, 100)
    
    t_data = df_modelo['t_min'].values
    T_data = df_modelo[sensor_critico].values
    T_inicial = T_data[0]
    
    def modelo_calentamiento(t, delta_T, tau):
        return T_inicial + delta_T * (1 - np.exp(-t / tau))
    
    df_modelo['Temp Amb'] = pd.to_numeric(df_modelo['Temp Amb'], errors='coerce')
    if not df_modelo['Temp Amb'].isna().all():
        fig.add_trace(go.Scatter(x=df_modelo['t_min'], y=df_modelo['Temp Amb'], mode='lines+markers', name='Temp Ambiente', line=dict(color='gray', dash='dot')))

    try:
        popt, pcov = curve_fit(modelo_calentamiento, t_data, T_data, p0=[50, 60], bounds=(0, [200, 500]))
        delta_T_fit, tau_fit = popt
        
        # Gráficos
        fig.add_trace(go.Scatter(x=t_data, y=T_data, mode='markers+lines', name=f'Medición {sensor_critico}', marker=dict(size=8)))
        T_proy = modelo_calentamiento(t_proyeccion, delta_T_fit, tau_fit)
        fig.add_trace(go.Scatter(x=t_proyeccion, y=T_proy, mode='lines', name=f'Proyección Modelo', line=dict(dash='dash', color='blue')))
        
        cinco_tau = 5 * tau_fit
        fig.add_vline(x=cinco_tau, line_width=1, line_dash="dot", line_color="red", annotation_text=f"5 Tau ({cinco_tau:.0f} min)")
        
        # ==========================================
        # ANÁLISIS DE CRITERIO: PROYECTADO VS REAL
        # ==========================================
        col_m1, col_m2 = st.columns(2)
        
        # A) PROYECTADO (Derivada matemática actual)
        derivada_actual = (delta_T_fit / tau_fit) * np.exp(-t_data[-1] / tau_fit) * 60
        col_m1.metric(label=f"Variación PROYECTADA ({sensor_critico})", value=f"{derivada_actual:.2f} °C/hr")
        
        if derivada_actual <= 1.0:
            col_m1.success("El modelo matemático asume que ya se alcanzó la estabilidad térmica (≤ 1°C/hr).")
        else:
            col_m1.warning("Según la curva, aún falta para la estabilidad.")

        # B) REAL MEDIDO (Requiere al menos 60 min de ensayo)
        if t_data[-1] >= 60:
            # Buscar el punto medido que esté aprox 60 mins atrás del último
            t_actual = t_data[-1]
            idx_1h = np.argmin(np.abs(t_data - (t_actual - 60)))
            t_hace_1h = t_data[idx_1h]
            
            # Si el punto encontrado está entre 50 y 70 min de diferencia, lo damos por válido
            if 50 <= (t_actual - t_hace_1h) <= 70:
                variacion_real = T_data[-1] - T_data[idx_1h]
                col_m2.metric(label=f"Variación REAL última hora ({sensor_critico})", value=f"{variacion_real:.1f} °C/hr")
                if variacion_real <= 1.0:
                    col_m2.success("¡Comprobación empírica! En la última hora real la variación fue ≤ 1°C.")
                else:
                    col_m2.warning("La medición física marca que aún no se estabilizó.")
            else:
                col_m2.info("Falta una lectura cercana a 60 min atrás para verificar el dato real.")
        else:
            col_m2.info("El ensayo debe superar los 60 min para calcular la variación real.")
            
    except Exception as e:
        st.error(f"El modelo necesita más dispersión térmica. Esperá a cargar otra lectura del sensor {sensor_critico}.")

    fig.update_layout(xaxis_title="Tiempo (min)", yaxis_title="Temperatura (°C)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Cargá al menos 3 mediciones de tiempo para ajustar la proyección matemática.")
