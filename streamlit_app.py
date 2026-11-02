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
def entrenar_demo_sintetica():
    rng = np.random.default_rng(42)
    n = 586
    X_train = pd.DataFrame({
        'mapeo_horas_suenio': rng.choice([1, 2, 3, 4], n, p=[0.25, 0.25, 0.35, 0.15]),
        'mapeo_facilidad_concentracion': rng.choice([1, 2, 3], n, p=[0.25, 0.50, 0.25]),
        'desconexion_pantallas_min': rng.choice([0, 20, 40], n, p=[0.45, 0.35, 0.20]),
        'mapeo_edad': rng.choice([18, 27, 42, 57, 70], n),
        'ocupacion_Trabajo tiempo completo / Independiente': rng.choice([0, 1], n, p=[0.4, 0.6]),
        'ocupacion_Trabajo y estudio': rng.choice([0, 1], n, p=[0.75, 0.25]),
        'ocupacion_Estudio unicamente': rng.choice([0, 1], n, p=[0.85, 0.15]),
        'drenador_La carga mental / Estres trabajo o estudio': rng.choice([0, 1], n, p=[0.45, 0.55]),
        'drenador_Las preocupaciones economicas o familiares': rng.choice([0, 1], n, p=[0.75, 0.25]),
        'drenador_El uso excesivo de pantallas / Las redes sociales': rng.choice([0, 1], n, p=[0.85, 0.15])
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

modelo_demo = entrenar_demo_sintetica()

OPCIONES_MENU = [
    "🏠 Resumen General",
    "🚀 Sprint 0",
    "🔍 Sprint 1",
    "🧹 Sprint 2",
    "📊 Sprint 3",
    "🤖 Sprint 4"
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
    st.title("🔋 Bater[IA] — ¿Cuánto te queda?")
    st.subheader("Trazabilidad de Práctica Profesionalizante I (Sprints 0 al 4)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ámbito / Organización", "Población General", "Multigeneracional / Multiocupacional")
    c2.metric("Muestra Sprint 2", "N = 554", "180 Airtable + 374 Kaggle")
    c3.metric("Muestra Sprint 3 y 4", "N = 586", "25 variables tras One-Hot")
    c4.metric("Modelo (S4)", "Demo sintética", "Sin métricas reales verificables")

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
# VISTA 5: SPRINT 4 — ENCUENTRO 5
# ==========================================
elif vista == OPCIONES_MENU[5]:
    st.header("🤖 Sprint 4 — Encuentro 5: Modelo Baseline y Convergencia")
    st.info("**Pregunta Eje:** *¿Puedo predecir o explicar algo con esto?* | **Prototipo ilustrativo; requiere datos reales para validar**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Datos de entrenamiento", "No incluidos", "No hay dataset en el repositorio")
    m2.metric("Desempeño medido", "No disponible", "No se puede validar con datos sintéticos")
    m3.metric("Error medio (MAE)", "No disponible", "Requiere evaluación con datos reales")
    m4.metric("Hipótesis Rectora", "H1 · Actor", "Diagnóstica · Pendiente de contraste")

    t1, t2, t3 = st.tabs(["🧠 Simulador ilustrativo (datos sintéticos)", "💼 Negocio & Datos (Decisión y Costo de Error)", "📋 Producto & Gestión (Convergencia y Borrador)"])
    with t1:
        st.warning("Esta demo se entrena con datos sintéticos generados por el código, no con la muestra del proyecto. La estimación no es una predicción validada.")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            suenio_ord = st.select_slider("1. `mapeo_horas_suenio` (1: <5h | 2: 5-6h | 3: 6-8h | 4: >8h):", options=[1, 2, 3, 4], value=2)
            foco_ord = st.select_slider("2. `mapeo_facilidad_concentracion` (1: Difícil | 2: Moderado | 3: Fácil):", options=[1, 2, 3], value=1)
            pantallas_min = st.selectbox("3. `desconexion_pantallas_min` (Minutos):", options=[0, 20, 40], index=0)
            edad_val = st.slider("4. `mapeo_edad` (Edad):", 18, 75, 27)
        with col_f2:
            ocup_sel = st.selectbox("5–7. Ocupación (One-Hot):", [
                "Trabajo y estudio (H1 Rectora)",
                "Trabajo tiempo completo / Independiente",
                "Estudio unicamente",
                "Otra ocupación"
            ])
            dren_sel = st.selectbox("8–10. Drenador Principal (One-Hot):", [
                "La carga mental / Estres trabajo o estudio",
                "Las preocupaciones economicas o familiares",
                "El uso excesivo de pantallas / Las redes sociales",
                "Otro drenador"
            ])
            bateria_usr = st.select_slider("Autopercepción del usuario hoy (%):", options=[20, 40, 60, 80, 100], value=40)

        vec = pd.DataFrame([{
            'mapeo_horas_suenio': suenio_ord,
            'mapeo_facilidad_concentracion': foco_ord,
            'desconexion_pantallas_min': pantallas_min,
            'mapeo_edad': edad_val,
            'ocupacion_Trabajo tiempo completo / Independiente': 1 if ocup_sel == "Trabajo tiempo completo / Independiente" else 0,
            'ocupacion_Trabajo y estudio': 1 if ocup_sel == "Trabajo y estudio (H1 Rectora)" else 0,
            'ocupacion_Estudio unicamente': 1 if ocup_sel == "Estudio unicamente" else 0,
            'drenador_La carga mental / Estres trabajo o estudio': 1 if dren_sel == "La carga mental / Estres trabajo o estudio" else 0,
            'drenador_Las preocupaciones economicas o familiares': 1 if dren_sel == "Las preocupaciones economicas o familiares" else 0,
            'drenador_El uso excesivo de pantallas / Las redes sociales': 1 if dren_sel == "El uso excesivo de pantallas / Las redes sociales" else 0
        }])
        pred = int(modelo_demo.predict(vec)[0])
        st.success(f"### 🎯 Resultado ilustrativo: **{pred}%** | (Diferencia con usuario: {abs(pred - bateria_usr)} pp)")

    with t2:
        st.markdown("""
        - **Decisión de negocio:** La estimación se usa como **señal orientativa revisable por el usuario** para priorizar qué hábito o dimensión revisar en el profiling (no automatiza diagnósticos).
        - **Costo de Falso Positivo de Vitalidad (Mayor riesgo):** Estimar batería alta (80%–100%) cuando la persona está agotada (20%–40%), porque minimiza su cansancio real.
        - **Costo de Falso Negativo de Vitalidad:** Estimar batería baja cuando la persona se siente bien; costo moderado si se presenta como sugerencia preventiva.
        """)
    with t3:
        st.markdown("""
        - **Convergencia en H1 como Hipótesis Rectora:** Cumple el piso de madurez **Diagnóstico**; todas las hipótesis (H1–H7) permanecen **Pendientes de contraste inferencial definitivo**.
        - **Primer Borrador del Informe Técnico (Secciones 1 a 8):** 1. Comprensión del negocio · 2. Datos · 3. Análisis exploratorio · 4. Backlog de hipótesis · 5. Modelado · 6. Resultados · 7. Reproducibilidad · 8. Límites y próximos pasos hacia PP2.
        """)