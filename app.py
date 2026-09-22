import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Ensayo de Calentamiento")
st.title("Estabilidad Térmica")

# ==========================================
# METADATOS DEL ENSAYO
# ==========================================
st.subheader("Información General del Ensayo")
col_info1, col_info2, col_info3 = st.columns(3)
fecha_ensayo = col_info1.date_input("Fecha de medición", datetime.today())
lugar_ensayo = col_info2.text_input("Lugar de ensayo", placeholder="Ej: Laboratorio Central / Planta...")
datos_equipo = col_info3.text_input("Datos del equipo", placeholder="Ej: Transformador TX-01, Interruptor...")

# ==========================================
# PARÁMETROS DE INYECCIÓN
# ==========================================
st.subheader("Condiciones de Contacto")
col_param1, col_param2 = st.columns(2)
r_ini = col_param1.number_input("R Contacto Inicio (µΩ)", value=0.0)
r_fin = col_param2.number_input("R Contacto Fin (µΩ)", value=0.0)

# ==========================================
# INICIALIZACIÓN DE DATOS
# ==========================================
sensores = ['Ua', 'Ub', 'Uc', 'Va', 'Vb', 'Vc', 'Wa', 'Wb', 'Wc']
if 'df_ensayo' not in st.session_state:
    cols = ['Nº Medición', 'Hora (HH:MM)', 'Corriente [A]', 'Temp Amb'] + sensores
    st.session_state.df_ensayo = pd.DataFrame(columns=cols)
    # Se agrega 600.0 como valor por defecto inicial para la corriente
    st.session_state.df_ensayo.loc[0] = [1, '14:00', 600.0, 25.0] + [25.0]*9

# ==========================================
# REGISTRO DE MEDICIONES (EVITANDO REFRESH)
# ==========================================
st.subheader("Registro de Mediciones")

def obtener_hora_local():
    hora_arg = datetime.utcnow() - timedelta(hours=3)
    return hora_arg.strftime('%H:%M')

if st.button("➕ Agregar medición con hora actual"):
    hora_actual = obtener_hora_local()
    df_temp = st.session_state.df_ensayo.copy()
    
    if len(df_temp) > 0:
        nueva_fila = df_temp.iloc[-1].copy()
        nueva_fila['Hora (HH:MM)'] = hora_actual
    else:
        nueva_fila = pd.Series([1, hora_actual, 600.0, 25.0] + [25.0]*9, index=df_temp.columns)
    
    df_temp.loc[len(df_temp)] = nueva_fila
    df_temp.reset_index(drop=True, inplace=True)
    df_temp['Nº Medición'] = range(1, len(df_temp) + 1)
    
    st.session_state.df_ensayo = df_temp

with st.form("formulario_mediciones"):
    st.markdown("✏️ **Modo de edición:** Cambiá los valores tranquilamente. La pantalla no se va a actualizar hasta que guardes los cambios.")
    
    df_editado_form = st.data_editor(
        st.session_state.df_ensayo, 
        num_rows="dynamic", 
        use_container_width=True,
        hide_index=True,
        disabled=["Nº Medición"]
    )
    
    guardar_cambios = st.form_submit_button("💾 Guardar Cambios y Actualizar Gráficos", type="primary")

if guardar_cambios:
    df_editado_form.reset_index(drop=True, inplace=True)
    df_editado_form['Nº Medición'] = range(1, len(df_editado_form) + 1)
    st.session_state.df_ensayo = df_editado_form

df_modelo = st.session_state.df_ensayo.copy()

# ==========================================
# PROCESAMIENTO MATEMÁTICO DE TIEMPOS
# ==========================================
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

if len(df_modelo) > 0:
    hora_cero = df_modelo['Hora (HH:MM)'].iloc[0]
    df_modelo['t_min'] = df_modelo['Hora (HH:MM)'].apply(lambda x: calcular_minutos(x, hora_cero))
    df_modelo = df_modelo.dropna(subset=['t_min']).copy()
    
    df_modelo['Corriente [A]'] = pd.to_numeric(df_modelo['Corriente [A]'], errors='coerce')
    for s in sensores:
        df_modelo[s] = pd.to_numeric(df_modelo[s], errors='coerce')

# ==========================================
# ANÁLISIS DE ESTABILIDAD
# ==========================================
st.subheader("Análisis de Estabilidad y Proyección")

sensor_por_defecto = "Ua"
max_temp_actual = 0
delta_max_medido = 0
delta_sensor = ""
tiempo_transcurrido = 0

if len(df_modelo) > 1:
    ultima_fila = df_modelo.iloc[-1]
    fila_anterior = df_modelo.iloc[-2]
    
    valores_ultima = ultima_fila[sensores].astype(float)
    valores_anterior = fila_anterior[sensores].astype(float)
    
    if not valores_ultima.isna().all() and not valores_anterior.isna().all():
        sensor_por_defecto = valores_ultima.idxmax()
        max_temp_actual = valores_ultima[sensor_por_defecto]
        
        diferencias = valores_ultima - valores_anterior
        delta_sensor = diferencias.idxmax()
        delta_max_medido = diferencias[delta_sensor]
        tiempo_transcurrido = ultima_fila['t_min'] - fila_anterior['t_min']

try:
    idx_defecto = sensores.index(sensor_por_defecto)
except:
    idx_defecto = 0
sensor_critico = st.selectbox("Sensor bajo análisis (se actualiza automáticamente al de mayor temperatura):", sensores, index=idx_defecto)

if len(df_modelo) > 1 and delta_sensor != "":
    st.info(f"**Mayor incremento en último intervalo:** {delta_max_medido:.1f} °C en el sensor {delta_sensor} (pasaron {tiempo_transcurrido:.0f} min desde la lectura anterior).")

# ==========================================
# AJUSTE Y GRÁFICOS
# ==========================================
if len(df_modelo) > 2:
    # Se crea un gráfico con un eje Y secundario
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    t_proyeccion = np.linspace(0, max(df_modelo['t_min']) + 120, 100)
    
    t_data = df_modelo['t_min'].values
    T_data = df_modelo[sensor_critico].values
    
    mascara_validos = ~np.isnan(T_data)
    t_data = t_data[mascara_validos]
    T_data = T_data[mascara_validos]
    
    if len(T_data) > 2:
        T_inicial = T_data[0]
        
        def modelo_calentamiento(t, delta_T, tau):
            return T_inicial + delta_T * (1 - np.exp(-t / tau))
        
        # 1. Graficar Corriente (Eje Y Secundario)
        if not df_modelo['Corriente [A]'].isna().all():
            fig.add_trace(go.Scatter(
                x=df_modelo['t_min'], 
                y=df_modelo['Corriente [A]'], 
                mode='lines+markers', 
                name='Corriente Inyectada [A]', 
                line=dict(color='orange', dash='dashdot')
            ), secondary_y=True)

        # 2. Graficar Temp Ambiente (Eje Y Principal)
        df_modelo['Temp Amb'] = pd.to_numeric(df_modelo['Temp Amb'], errors='coerce')
        if not df_modelo['Temp Amb'].isna().all():
            fig.add_trace(go.Scatter(
                x=df_modelo['t_min'], 
                y=df_modelo['Temp Amb'], 
                mode='lines+markers', 
                name='Temp Ambiente', 
                line=dict(color='gray', dash='dot')
            ), secondary_y=False)

        # 3. Graficar Datos y Proyección de Temperatura (Eje Y Principal)
        try:
            popt, pcov = curve_fit(modelo_calentamiento, t_data, T_data, p0=[50, 60], bounds=(0, [200, 500]))
            delta_T_fit, tau_fit = popt
            
            fig.add_trace(go.Scatter(
                x=t_data, y=T_data, 
                mode='markers+lines', 
                name=f'Medición {sensor_critico}', 
                marker=dict(size=8)
            ), secondary_y=False)
            
            T_proy = modelo_calentamiento(t_proyeccion, delta_T_fit, tau_fit)
            fig.add_trace(go.Scatter(
                x=t_proyeccion, y=T_proy, 
                mode='lines', 
                name=f'Proyección Modelo', 
                line=dict(dash='dash', color='blue')
            ), secondary_y=False)
            
            cinco_tau = 5 * tau_fit
            fig.add_vline(x=cinco_tau, line_width=1, line_dash="dot", line_color="red", annotation_text=f"5 Tau ({cinco_tau:.0f} min)")
            
            col_m1, col_m2 = st.columns(2)
            
            # A) PROYECTADO
            derivada_actual = (delta_T_fit / tau_fit) * np.exp(-t_data[-1] / tau_fit) * 60
            col_m1.metric(label=f"Variación PROYECTADA ({sensor_critico})", value=f"{derivada_actual:.2f} °C/hr")
            
            if derivada_actual <= 1.0:
                col_m1.success("El modelo matemático asume que ya se alcanzó la estabilidad térmica (≤ 1°C/hr).")
            else:
                col_m1.warning("Según la curva, aún falta para la estabilidad.")

            # B) REAL MEDIDO
            if t_data[-1] >= 60:
                t_actual = t_data[-1]
                idx_1h = np.argmin(np.abs(t_data - (t_actual - 60)))
                t_hace_1h = t_data[idx_1h]
                
                if 50 <= (t_actual - t_hace_1h) <= 70:
                    T_actual_medida = T_data[-1]
                    T_hace_1h_medida = T_data[idx_1h]
                    variacion_real = T_actual_medida - T_hace_1h_medida
                    
                    col_m2.metric(label=f"Variación REAL última hora ({sensor_critico})", value=f"{variacion_real:.1f} °C/hr")
                    col_m2.caption(f"*Comparando sensor {sensor_critico}: {T_actual_medida:.1f} °C (Min {t_actual:.0f}) vs {T_hace_1h_medida:.1f} °C (Min {t_hace_1h:.0f})*")
                    
                    if variacion_real <= 1.0:
                        col_m2.success(f"¡Comprobación empírica! El sensor {sensor_critico} alcanzó la estabilidad (variación ≤ 1°C).")
                    else:
                        col_m2.warning(f"La medición física marca que {sensor_critico} aún no se estabilizó.")
                else:
                    col_m2.info("Falta una lectura cercana a 60 min atrás para verificar el dato real.")
                    col_m2.caption(f"*Última lectura en min {t_actual:.0f}. Lectura más cercana a la hora: min {t_hace_1h:.0f} (fuera de rango).*")
            else:
                col_m2.info("El ensayo debe superar los 60 min para calcular la variación real.")
                
        except Exception as e:
            st.error(f"El modelo necesita más dispersión térmica. Esperá a cargar otra lectura del sensor {sensor_critico}.")

        # Configuración de los ejes
        fig.update_layout(xaxis_title="Tiempo transcurrido (min)")
        fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=False)
        # Se fuerza a que el eje de corriente arranque desde 0
        fig.update_yaxes(title_text="Corriente [A]", secondary_y=True, rangemode="tozero")
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# EXPORTACIÓN
# ==========================================
st.divider()
st.subheader("Finalizar y Exportar")
@st.cache_data
def convertir_df_a_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convertir_df_a_csv(df_modelo)
st.download_button(
    label="📥 Descargar mediciones (CSV)",
    data=csv,
    file_name='ensayo_estabilidad.csv',
    mime='text/csv'
)
