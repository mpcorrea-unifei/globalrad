# app.py
import streamlit as st
import numpy as np
import pandas as pd
import math
import json
from datetime import datetime
import xgboost as xgb
from sklearn.linear_model import Lasso
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from pathlib import Path

# ------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------
st.set_page_config(
    page_title="GSR-ML - Estimativa de Radiação Solar",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f39c12;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 1rem;
        color: #576574;
    }
    .stButton>button {
        background-color: #f39c12;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e67e22;
        transform: scale(1.02);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffecd2 0%, #fcb69f 100%);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Funções auxiliares (cálculo de H0, carregamento de modelos)
# ------------------------------------------------------------
Gsc = 0.0820

def extraterrestrial_radiation(doy, lat_deg):
    phi = math.radians(lat_deg)
    delta = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    omega_s = math.acos(-math.tan(phi) * math.tan(delta))
    d_r = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    H0 = (24 * 60 / math.pi) * Gsc * d_r * (
        omega_s * math.sin(phi) * math.sin(delta) +
        math.cos(phi) * math.cos(delta) * math.sin(omega_s)
    )
    return H0

@st.cache_resource
def load_lasso_model(path='H_lasso.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    model = Lasso(alpha=data['alpha'])
    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])
    return model, np.array(data['scaler_mean']), np.array(data['scaler_scale']), data['feature_names']

@st.cache_resource
def load_xgboost_model(model_path='H_xgboost.json', scaler_path='scaler_H_xgboost.json'):
    model = xgb.Booster()
    model.load_model(model_path)
    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)
    return model, np.array(scaler_data['scaler_mean']), np.array(scaler_data['scaler_scale']), scaler_data['feature_names']

# ------------------------------------------------------------
# Sidebar – informações e configurações
# ------------------------------------------------------------
#with st.sidebar:
#    col1, col2, col3, col4 = st.columns(4)
#    col1.image("Fapemig.png", width=80)
#    col2.image("Unifei.png", width=80)
#    col3.image("logo_IRN_b.jpg", width=80)
#    col4.image("logo_CAT-removebg-preview.png", width=80)
# Função para converter imagens
def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

with st.sidebar:
    imagens = [
        "Fapemig.png",
        "Unifei.png",
        "logo_IRN_b.jpg",
        "logo_CAT-removebg-preview.png"
    ]

    # Converter cada imagem para base64 e construir o HTML
    imgs_html = '<div style="display: flex; justify-content: center; gap: 10px;">'
    for img in imagens:
        b64 = img_to_base64(img)
        # Preservar tipo de imagem (png ou jpg)
        ext = Path(img).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        imgs_html += f'<img src="data:{mime};base64,{b64}" width="65">'
    imgs_html += '</div>'

    st.markdown(imgs_html, unsafe_allow_html=True)   
    #st.image("Fapemig.png", width=80)
    #st.image("logo_IRN_b.jpg", width=80)
    st.markdown("## Sobre")
    st.markdown("""
    Este aplicativo estima a **radiação solar global diária** (MJ/m²/dia) 
    utilizando modelos de machine learning (Lasso e XGBoost) treinados 
    com dados coletados em Itajubá-MG entre julho/23 a setembro/25.
    
    **Modelos disponíveis:** Lasso (linear regularizado) e XGBoost (gradient boosting).
    
    **Projeto Fapemig APQ-00768-18**
    """)
    st.markdown("---")
    st.markdown("### ℹ️ Como usar")
    st.markdown("""
    1. Insira a data e os dados meteorológicos.
    2. Clique em **Estimar Radiação**.
    3. Compare os resultados no painel principal.
    """)
    st.markdown("---")
    st.caption(f"© {datetime.now().year} - RadSolar v1.0")

# ------------------------------------------------------------
# Cabeçalho principal
# ------------------------------------------------------------
st.markdown('<div class="main-header">☀️ GSR-ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Global Solar Radiation via Machine Learning</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Layout de entrada de dados (colunas)
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown("### 📅 Data & Local")
    data = st.date_input("Data", value=datetime(2025, 1, 1))
    latitude = st.number_input("Latitude (°)", value=-22.42, step=0.01, 
                               help="Padrão: Itajubá. Alterar apenas se o modelo for recalibrado.")
    
with col2:
    st.markdown("### 🌡️ Temperatura & Umidade")
    tmax = st.number_input("Temp. máxima (°C)", value=30.0, step=0.1)
    tmin = st.number_input("Temp. mínima (°C)", value=18.0, step=0.1)
    rh = st.number_input("Umidade relativa (%)", value=65.0, min_value=0.0, max_value=100.0, step=0.1)

with col3:
    st.markdown("### 💨 Vento & Pressão")
    wind = st.number_input("Vento (m/s)", value=2.0, step=0.1)
    pressure = st.number_input("Pressão (hPa)", value=920.0, step=0.1)

# Botão centralizado
_, center_col, _ = st.columns([0.3, 0.4, 0.3])
with center_col:
    estimar = st.button("🔆 Estimar Radiação Solar", use_container_width=True)

# ------------------------------------------------------------
# Processamento e exibição de resultados
# ------------------------------------------------------------
if estimar:
    doy = data.timetuple().tm_yday
    H0 = extraterrestrial_radiation(doy, latitude)
    sin_doy = math.sin(2 * math.pi * doy / 365)
    cos_doy = math.cos(2 * math.pi * doy / 365)
    
    # DataFrame de entrada
    feature_names = ['Tmax', 'Tmin', 'RHmean', 'Wind', 'Pressure', 'H0', 'sin_doy', 'cos_doy']
    entrada = pd.DataFrame([{
        'Tmax': tmax,
        'Tmin': tmin,
        'RHmean': rh,
        'Wind': wind,
        'Pressure': pressure,
        'H0': H0,
        'sin_doy': sin_doy,
        'cos_doy': cos_doy
    }])
    
    resultados = {}
    
    # Previsão Lasso
    try:
        lasso_model, lasso_mean, lasso_scale, lasso_feat = load_lasso_model()
        X_l = entrada[lasso_feat].values
        X_l_scaled = (X_l - lasso_mean) / lasso_scale
        pred_lasso = lasso_model.predict(X_l_scaled)[0]
        resultados['Lasso'] = pred_lasso
    except FileNotFoundError:
        st.warning("⚠️ Modelo Lasso (H_lasso.json) não encontrado.")
    except Exception as e:
        st.error(f"Erro no Lasso: {e}")
    
    # Previsão XGBoost
    try:
        xgb_model, xgb_mean, xgb_scale, xgb_feat = load_xgboost_model()
        X_x = entrada[xgb_feat].values
        X_x_scaled = (X_x - xgb_mean) / xgb_scale
        dmatrix = xgb.DMatrix(X_x_scaled, feature_names=xgb_feat)
        pred_xgb = xgb_model.predict(dmatrix)[0]
        resultados['XGBoost'] = pred_xgb
    except FileNotFoundError:
        st.warning("⚠️ Modelo XGBoost (H_xgboost.json) não encontrado.")
    except Exception as e:
        st.error(f"Erro no XGBoost: {e}")
    
    if resultados:
        # Exibição dos parâmetros calculados
        with st.expander("📊 Parâmetros calculados", expanded=True):
            cols_param = st.columns(4)
            cols_param[0].metric("Dia do ano", doy)
            cols_param[1].metric("Rad. Extraterrestre (H₀)", f"{H0:.2f} MJ/m²/dia")
            cols_param[2].metric("sin(doy)", f"{sin_doy:.3f}")
            cols_param[3].metric("cos(doy)", f"{cos_doy:.3f}")
        
        st.markdown("---")
        st.markdown("## 📈 Resultados das Estimativas")
        
        # Cards com valores estimados
        card_cols = st.columns(len(resultados))
        for idx, (modelo, valor) in enumerate(resultados.items()):
            with card_cols[idx]:
                st.markdown(f"""
                <div class="result-card">
                    <div class="metric-label">{modelo}</div>
                    <div class="metric-value">{valor:.2f}</div>
                    <div class="metric-label">MJ/m²/dia</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Gráfico comparativo com Plotly
        st.markdown("### 📊 Comparação Visual")
        fig = go.Figure()
        modelos_list = list(resultados.keys())
        valores_list = list(resultados.values())
        
        fig.add_trace(go.Bar(
            x=modelos_list,
            y=valores_list,
            marker_color=['#f39c12', '#2c3e50'],
            text=[f'{v:.2f}' for v in valores_list],
            textposition='outside',
            textfont=dict(size=16),
            width=0.5
        ))
        fig.update_layout(
            title=None,
            xaxis_title='Modelo',
            yaxis_title='Radiação (MJ/m²/dia)',
            template='plotly_white',
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        fig.update_yaxes(range=[0, max(valores_list)*1.2])
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretação
        st.markdown("---")
        st.info("""
        **Interpretação:** Os valores indicam a energia solar total (onda curta) recebida em uma superfície horizontal 
        ao longo do dia. Essa grandeza é fundamental para estimar a evapotranspiração de referência (ETo) pelo método 
        FAO-56 e alimentar modelos de produtividade agrícola.
        
        - **Lasso** é um modelo linear regularizado, interpretável e leve.
        - **XGBoost** captura relações não lineares, geralmente mais preciso.
        """)
    else:
        st.error("❌ Nenhum modelo pôde ser carregado. Verifique se os arquivos JSON estão na pasta do aplicativo.")
