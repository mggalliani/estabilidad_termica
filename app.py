# B) REAL MEDIDO (Requiere al menos 60 min de ensayo)
        if t_data[-1] >= 60:
            # Buscar el punto medido que esté aprox 60 mins atrás del último
            t_actual = t_data[-1]
            idx_1h = np.argmin(np.abs(t_data - (t_actual - 60)))
            t_hace_1h = t_data[idx_1h]
            
            # Si el punto encontrado está entre 50 y 70 min de diferencia, lo damos por válido
            if 50 <= (t_actual - t_hace_1h) <= 70:
                T_actual = T_data[-1]
                T_hace_1h = T_data[idx_1h]
                variacion_real = T_actual - T_hace_1h
                
                col_m2.metric(label=f"Variación REAL última hora ({sensor_critico})", value=f"{variacion_real:.1f} °C/hr")
                
                # Detalle de la comparación
                st.caption(f"*Comparando sensor {sensor_critico}: {T_actual:.1f} °C (Min {t_actual:.0f}) vs {T_hace_1h:.1f} °C (Min {t_hace_1h:.0f})*")
                
                if variacion_real <= 1.0:
                    col_m2.success(f"¡Comprobación empírica! El sensor {sensor_critico} alcanzó la estabilidad (variación ≤ 1°C).")
                else:
                    col_m2.warning(f"La medición física marca que {sensor_critico} aún no se estabilizó.")
            else:
                col_m2.info("Falta una lectura cercana a 60 min atrás para verificar el dato real.")
                st.caption(f"*Última lectura en min {t_actual:.0f}. Lectura más cercana a la hora: min {t_hace_1h:.0f} (fuera de rango).*")
        else:
            col_m2.info("El ensayo debe superar los 60 min para calcular la variación real.")
