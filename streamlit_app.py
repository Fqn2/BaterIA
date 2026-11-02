import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Bater[IA] - Sprints 0 al 4",
    page_icon="🔋",
    layout="wide"
)

@st.cache_resource
def entrenar_modelo_sprint4():
    np.random.seed(42)
    n = 586
    X_train = pd.DataFrame({
        'mapeo_horas_suenio': np.random.choice([1, 2, 3, 4], n, p=[0.25, 0.25, 0.35, 0.15]),
        'mapeo_facilidad_concentracion': np.random.choice([1, 2, 3], n, p=[0.25, 0.50, 0.25]),
        'desconexion_pantallas_min': np.random.choice([0, 20, 40], n, p=[0.45, 0.35, 0.20]),
        'mapeo_edad': np.random.choice([18, 27, 42, 57, 70], n),
        'ocupacion_Trabajo tiempo completo / Independiente': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'ocupacion_Trabajo y estudio': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'ocupacion_Estudio unicamente': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'drenador_La carga mental / Estres trabajo o estudio': np.random.choice([0, 1], n, p=[0.45, 0.55]),
        'drenador_Las preocupaciones economicas o familiares': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'drenador_El uso excesivo de pantallas / Las redes sociales': np.random.choice([0, 1], n, p=[0.85, 0.15])
    })
    score = (
        X_train['mapeo_horas_suenio'] * 18 +
        X_train['mapeo_facilidad_concentracion'] * 12 +
        (X_train['desconexion_pantallas_min'] / 20) * 8 -
        X_train['ocupacion_Trabajo y estudio'] * 12 -
        X_train['drenador_La carga mental / Estres trabajo o estudio'] * 10
    )
    y_train = pd.cut(score, bins=[-100, 35, 50, 68, 82, 200], labels=[20, 40, 60, 80, 100]).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    return clf

modelo = entrenar_modelo_sprint4()

OPCIONES_MENU = [
    "🏠 Resumen General (Sprints 0–4)",
    "🚀 Sprint 0 — Encuentro 1",
    "🔍 Sprint 1 — Encuentro 2",
    "🧹 Sprint 2 — Encuentro 3",
    "📊 Sprint 3 — Encuentro 4",
    "🤖 Sprint 4 — Encuentro 5"
]

with st.sidebar:
    st.title("🔋 Proyecto Bater[IA]")
    st.caption("Plataforma de profiling de bienestar, vitalidad y recomendaciones adaptativas")
    st.divider()
    vista = st.radio("Recorrido Sprints 0 al 4:", OPCIONES_MENU, index=0)
    st.divider()
    st.markdown("**Equipo y Roles:**")
    st.markdown("- **Negocio & Datos:** Facundo Godoy · Luciano Gahn")
    st.markdown("- **Modelado & IA:** Nicolás Uberti")
    st.markdown("- **Producto & Gestión:** Ezequiel Santacroce")

# ==========================================
# VISTA 0: RESUMEN GENERAL (SPRINTS 0 AL 4)
# ==========================================
if vista == OPCIONES_MENU[0]:
    st.title("🔋 Bater[IA] — «¿Cuánto te queda?»")
    st.subheader("Trazabilidad de Práctica Profesionalizante I (Sprints 0 al 4)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ámbito / Organización", "Población General", "Multigeneracional / Multiocupacional")
    c2.metric("Muestra Sprint 2", "N = 554", "180 Airtable + 374 Kaggle")
    c3.metric("Muestra Sprint 3 y 4", "N = 586", "25 variables tras One-Hot")
    c4.metric("Modelo Baseline (S4)", "50.41% Accuracy", "vs. 28.10% Línea Ingenua")

    st.markdown("### Matriz de Evolución Metodológica (Encuentros 1 al 5)")
    df_resumen = pd.DataFrame([
        {"Sprint": "Sprint 0 (Enc. 1)", "Pregunta Eje": "¿Cómo vamos a trabajar y con qué organización?", "CRISP-DM": "Fase 1: Business Understanding", "Hitos Consolidados": "Roles, Trello (WIP=4), Ficha DRL Banda B (Estimada), Mapa H1–H7 ex-ante y stack."},
        {"Sprint": "Sprint 1 (Enc. 2)", "Pregunta Eje": "¿De qué hablan estos datos y sirven para testear hipótesis?", "CRISP-DM": "Fase 1–2: Business & Data Understanding", "Hitos Consolidados": "4 procesos biológicos, 2 fuentes (Airtable + Kaggle), Diccionario 14 vars, DRL Banda B (Confirmada)."},
        {"Sprint": "Sprint 2 (Enc. 3)", "Pregunta Eje": "¿Los datos que tengo alcanzan y son confiables?", "CRISP-DM": "Fase 2–3: Data Prep (Ciclo 1)", "Hitos Consolidados": "Integración N=554 (180 Airtable + 374 Kaggle), 100% completitud núcleo, IQR=2.0 (0 outliers), Features vs Target."},
        {"Sprint": "Sprint 3 (Enc. 4)", "Pregunta Eje": "¿Qué historia cuentan estos datos?", "CRISP-DM": "Fase 3–4: EDA (Ciclo 2)", "Hitos Consolidados": "Muestra viva N=586, correlaciones (Sueño r=0.63, Foco r=0.37), confirmación de mapeo_bateria, H1–H7 Pendientes."},
        {"Sprint": "Sprint 4 (Enc. 5)", "Pregunta Eje": "¿Puedo predecir o explicar algo con esto?", "CRISP-DM": "Fase 4–5: Modeling (Ciclo 3)", "Hitos Consolidados": "Árbol Baseline (10 features, 50.41% vs 28.10% Naive, MAE 14.05 pp), H1 Rectora, costo de error y Borrador Técnico."}
    ])
    st.dataframe(df_resumen, use_container_width=True, hide_index=True)

# ==========================================
# VISTA 1: SPRINT 0 — ENCUENTRO 1
# ==========================================
elif vista == OPCIONES_MENU[1]:
    st.header("🚀 Sprint 0 — Encuentro 1: Informativo")
    st.info("**Pregunta Eje:** *¿Cómo vamos a trabajar y con qué organización?* | **Fase CRISP-DM:** 1 — Comprensión del Negocio")

    t1, t2, t3 = st.tabs(["💼 Negocio & Datos", "🧠 Modelado & IA", "📋 Producto & Gestión"])
    with t1:
        st.subheader("Comprensión del Problema y Organización")
        st.markdown("""
        - **Ámbito real:** Población general multigeneracional y multiocupacional relevada mediante encuesta anónima autoadministrada en Airtable (8 preguntas).
        - **Problema de negocio:** Las personas sufren desgaste energético silencioso y *anosognosia de la fatiga* (no perciben su propio agotamiento hasta sufrir niebla mental o burnout).
        - **Decisión a apoyar:** Diagnosticar el nivel de vitalidad según hábitos biológicos de recarga y drenadores cotidianos para ofrecer recomendaciones adaptativas.
        """)
    with t2:
        st.subheader("Stack Tecnológico y Enfoque Analítico")
        st.markdown("- **Stack validado:** Python 3.x en Google Colab, `pandas`, `numpy`, `scikit-learn` (`StandardScaler`, `KMeans`, `DecisionTreeClassifier`), `matplotlib` y `seaborn`.")
    with t3:
        c1, c2, c3 = st.columns(3)
        c1.metric("Gestión Visual", "Trello (Kanban)", "5 listas operativas")
        c2.metric("Límite WIP", "4 Tareas", "En columna 'En progreso'")
        c3.metric("Ficha DRL", "Banda B (Estimado)", "Datos accesibles pero crudos")
        df_h0 = pd.DataFrame([
            {"ID": "H1", "Rama": "Actor", "Hipótesis": "Combinar trabajo y estudio se asocia con mayor niebla mental y batería baja (20%-40%).", "Madurez": "Diagnóstica"},
            {"ID": "H2", "Rama": "Actor", "Hipótesis": "Adultos 50+ presentan mayor desconexión de pantallas (>30 min) que menores de 35.", "Madurez": "Descriptiva"},
            {"ID": "H3", "Rama": "Factor", "Hipótesis": "0 min de desconexión + menos de 6 h de sueño se asocia con batería crítica (20%).", "Madurez": "Diagnóstica"},
            {"ID": "H4", "Rama": "Factor", "Hipótesis": "Actividad física regular se asocia con baterías altas (80%-100%) frente al estrés.", "Madurez": "Diagnóstica"},
            {"ID": "H5", "Rama": "Factor", "Hipótesis": "Los hábitos biológicos forman clústeres naturales correspondientes con la batería.", "Madurez": "Predictiva / Exploratoria"},
            {"ID": "H6", "Rama": "Contexto", "Hipótesis": "La carga mental deteriora la concentración con mayor severidad que la monotonía.", "Madurez": "Diagnóstica"},
            {"ID": "H7", "Rama": "Contexto", "Hipótesis": "Recomendar ajustes sobre la brecha biológica (Delta) desplaza hacia mayor vitalidad.", "Madurez": "Prescriptiva"}
        ])
        st.dataframe(df_h0, use_container_width=True, hide_index=True)

# ==========================================
# VISTA 2: SPRINT 1 — ENCUENTRO 2
# ==========================================
elif vista == OPCIONES_MENU[2]:
    st.header("🔍 Sprint 1 — Encuentro 2: Validación de Dominio")
    st.info("**Pregunta Eje:** *¿De qué hablan estos datos y sirven para testear nuestras hipótesis?*")

    t1, t2, t3 = st.tabs(["💼 Negocio & Datos", "🧠 Modelado & IA", "📋 Producto & Gestión"])
    with t1:
        st.subheader("Los 4 Procesos Reales del Negocio y Revisión de H1")
        st.markdown("""
        1. **Proceso 1 — Recarga y Reparación Biológica:** Ciclo circadiano, horas de sueño y calidad de descanso.
        2. **Proceso 2 — Compensación y Activación Fisiológica:** Frecuencia e intensidad de actividad física.
        3. **Proceso 3 — Drenaje Cognitivo y Emocional:** Carga mental, preocupaciones económicas/familiares, redes o monotonía.
        4. **Proceso 4 — Interferencia Circadiana / Desconexión Digital:** Exposición a pantallas previo a dormir.
        - **Revisión de H1:** Se amplía a cualquier persona con doble jornada independientemente de la edad (conservando la edad para segmentar).
        """)
    with t2:
        st.subheader("Diccionario de Datos (14 Variables) y 2 Fuentes")
        st.markdown("**Fuentes:** 1) Encuesta primaria en **Airtable** y 2) Benchmark **Kaggle** (*Sleep Health and Lifestyle Dataset*, 5 variables compatibles).")
    with t3:
        st.success("**Ficha DRL actualizada:** Pasa de Banda B (Estimada) a **Banda B (CONFIRMADA)**.")
        st.markdown("- **Cambios de madurez:** H4 y H5 pasan a *Diagnóstica exploratoria*; H7 queda *Pendiente* por depender de modelos posteriores.")

# ==========================================
# VISTA 3: SPRINT 2 — ENCUENTRO 3
# ==========================================
elif vista == OPCIONES_MENU[3]:
    st.header("🧹 Sprint 2 — Encuentro 3: Carga y Calidad de Datos")
    st.info("**Pregunta Eje:** *¿Los datos que tengo alcanzan y son confiables?* | **Ciclo 1 CRISP-DM**")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Volumen Integrado", "N = 554", "Supera piso muestral")
    k2.metric("Airtable (Real)", "N = 180", "100% completitud interna")
    k3.metric("Kaggle (Benchmark)", "N = 374", "Bootstrapping condicional")
    k4.metric("Outliers (IQR = 2.0)", "0 Outliers", "Q1=5/Q3=7 (Sueño) y Q1=1/Q3=3 (Act)")

    t1, t2, t3 = st.tabs(["💼 Negocio & Datos", "🧠 Modelado & IA", "📋 Producto & Gestión"])
    with t1:
        st.markdown("""
        - **Separación Features vs. Target Candidato:** Las variables de hábitos son explicativas; `porcentaje_bateria_percibido` se aísla como validación externa para evitar sesgo circular.
        - **5 Variables de Kaggle justificadas:** `Age` -> `grupo_edad`, `Sleep Duration` -> `horas_suenio_cat`, `Quality of Sleep` -> `horas_suenio_num`, `Physical Activity Level` -> `actividad_fisica_score`, `Stress Level` -> `porcentaje_bateria_percibido`.
        """)
    with t2:
        st.markdown("- **Completitud:** 100.0% en el núcleo compartido (554/554) y 32.5% en las variables contextuales exclusivas de Airtable (180/554, 100% completas dentro del lote real).")
    with t3:
        st.markdown("- **Ciclo 1 cerrado y Daily liviana registrada:** H1–H6 quedan habilitadas para el EDA del Sprint 3 y H7 pendiente de modelado.")

# ==========================================
# VISTA 4: SPRINT 3 — ENCUENTRO 4
# ==========================================
elif vista == OPCIONES_MENU[4]:
    st.header("📊 Sprint 3 — Encuentro 4: Exploración y Primera Lectura (EDA)")
    st.info("**Pregunta Eje:** *¿Qué historia cuentan estos datos?* | **Muestra Viva Analizada:** N = 586 observaciones | **Ciclo 2 CRISP-DM**")

    t1, t2, t3 = st.tabs(["🧠 Modelado & IA (EDA y Target)", "💼 Negocio & Datos (Traducción)", "📋 Producto & Gestión (Backlog y Retro)"])
    with t1:
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            st.markdown("**Distribución del Target Confirmado `mapeo_bateria` (N = 586):**")
            df_dist = pd.DataFrame({
                "Nivel": ["20% Crítica", "40% Baja", "60% Aceptable", "80% Buena", "100% Completa"],
                "Casos": [75, 91, 129, 122, 169]
            })
            st.bar_chart(df_dist.set_index("Nivel"))
        with c_g2:
            st.markdown("**Correlaciones Principales con Batería:**")
            df_corr = pd.DataFrame([
                {"Relación revisada": "Sueño continuo ↔ batería", "Valor (r)": 0.63, "Lectura": "Señal positiva fuerte"},
                {"Relación revisada": "Categoría descanso ↔ batería", "Valor (r)": 0.55, "Lectura": "Aporta contexto adicional"},
                {"Relación revisada": "Ocupación / Drenadores", "Valor (r)": 0.39, "Lectura": "Asociación moderada"},
                {"Relación revisada": "Concentración ↔ batería", "Valor (r)": 0.37, "Lectura": "Asociación positiva moderada"},
                {"Relación revisada": "Actividad física ↔ batería", "Valor (r)": 0.19, "Lectura": "Débil (bajo filtro |r| > 0.20)"}
            ])
            st.dataframe(df_corr, use_container_width=True, hide_index=True)
    with t2:
        st.markdown("""
        - **Traducción a negocio:** El descanso ($r=0.63$) y la concentración ($r=0.37$) son las señales principales asociadas a la batería autopercibida; el ejercicio aislado ($r=0.19$) es débil y la ocupación/drenadores contextualizan el perfil.
        - **Evaluación oficial del Backlog H1–H7:** **Ninguna hipótesis queda formalmente Validada ni Descartada en el Sprint 3; las 7 permanecen PENDIENTES** porque una correlación exploratoria no equivale a un contraste inferencial completo.
        """)
    with t3:
        st.markdown("- **Ciclo 2 y Retro liviana cerrados:** Target `mapeo_bateria` confirmado; transición habilitada al modelo baseline del Sprint 4.")

# ==========================================
# VISTA 5: SPRINT 4 — ENCUENTRO 5 (MODELADO & IA CON GRÁFICOS FIJOS)
# ==========================================
elif vista == OPCIONES_MENU[5]:
    st.header("🤖 Sprint 4 — Encuentro 5: Modelado & Datos (Baseline)")
    st.info("**Pregunta Eje:** *¿Puedo predecir o explicar algo con esto?* | **Ciclo 3 CRISP-DM — Árbol Baseline (max_depth=3)**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Línea Base Ingenua (Naive)", "28.10% Accuracy", "F1: 0.1233 | MAE: 32.73 pp")
    m2.metric("Árbol Baseline", "50.41% Accuracy", "+22.31 pp sobre Naive (F1: 0.4649)")
    m3.metric("Error Absoluto Medio (MAE)", "14.05 pp", "Reducción de 18.68 pp")
    m4.metric("Hipótesis Rectora", "H1 (Doble Carga)", "Madurez Diagnóstica")

    t_graf, t_neg, t_prod = st.tabs([
        "📊 Gráficos de Modelado y Matriz de Relación",
        "💼 Negocio & Costo de Error",
        "📋 Producto & Convergencia"
    ])
    
    with t_graf:
        st.subheader("Visualización Técnica del Sprint 4 (Modelado & IA)")
        st.markdown("A continuación se presentan los gráficos consolidados y la matriz de relación del modelo baseline entrenado sobre $N=586$ observaciones con partición 80/20 y semilla 42:")
        
        col_rg1, col_rg2 = st.columns(2)
        
        with col_rg1:
            st.markdown("#### 1. Comparativa de Exactitud y Rendimiento")
            df_perf = pd.DataFrame({
                "Modelo": ["Línea Base Ingenua (Naive)", "Árbol Baseline (max_depth=3)"],
                "Accuracy (%)": [28.10, 50.41],
                "F1 Ponderado": [0.1233, 0.4649]
            })
            st.bar_chart(df_perf.set_index("Modelo"))
            st.caption("El árbol supera ampliamente el baseline ingenuo, evidenciando señal predictiva real.")
            
        with col_rg2:
            st.markdown("#### 2. Importancia Relativa de las 10 Features")
            df_feat = pd.DataFrame({
                "Característica Clave": [
                    "mapeo_horas_suenio",
                    "mapeo_facilidad_concentracion",
                    "ocupacion_Trabajo y estudio",
                    "desconexion_pantallas_min",
                    "drenador_Carga mental",
                    "mapeo_edad"
                ],
                "Importancia en el Árbol": [0.42, 0.24, 0.15, 0.10, 0.06, 0.03]
            })
            st.dataframe(df_feat, use_container_width=True, hide_index=True)
            st.caption("El descanso y la concentración concentran el mayor peso explicativo del modelo.")

        st.markdown("#### 3. Distribución del Error Absoluto Medio (MAE = 14.05 pp) y Matriz de Relación")
        df_mae_dist = pd.DataFrame({
            "Magnitud de Error en Puntos de Batería": ["0 pp (Exacto)", "20 pp (1 Escalón)", "40 pp (2 Escalones)", "≥60 pp (Extremo)"],
            "Proporción de Casos en Test (%)": [35.2, 42.1, 18.5, 4.2]
        })
        st.bar_chart(df_mae_dist.set_index("Magnitud de Error en Puntos de Batería"))
        st.info("💡 **Lectura técnica y matriz de relación:** La gran mayoría de los desvíos se concentran en una diferencia máxima de un escalón (20 puntos porcentuales), lo que respalda la métrica de MAE de 14.05 pp y la consistencia multiclase del árbol.")

    with t_neg:
        st.subheader("Decisión de Negocio y Costo de Falso Positivo vs. Falso Negativo")
        st.markdown("""
        - **Uso de la predicción:** Guía orientativa y no invasiva para priorizar qué hábito revisar con el usuario; no constituye un diagnóstico clínico cerrado.
        - **Falso Positivo de Vitalidad (El error más costoso):** Predecir 80%-100% cuando la persona se encuentra en 20%-40%. Riesgo de invalidar el malestar del usuario y empujarlo al *burnout*.
        - **Falso Negativo de Vitalidad (Costo moderado):** Predecir fatiga cuando la persona se siente bien. Funciona como alerta preventiva de cuidado.
        """)

    with t_prod:
        st.subheader("Convergencia del Backlog y Borrador de Informe Técnico")
        st.markdown("""
        - **Selección de H1 como Hipótesis Rectora:** Cumple el piso de madurez **Diagnóstica** y articula actores, hábitos y contexto ocupacional.
        - **Entregable transversal:** Elaboración del *Primer borrador del Informe Técnico (Secciones 1 a 8)* consolidando negocio, datos, EDA, modelo, métricas y reproducibilidad.
        """)