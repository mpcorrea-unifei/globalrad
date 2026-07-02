# app.py
import streamlit as st
import numpy as np
import pandas as pd
import math
import json
from datetime import datetime
import xgboost as xgb
from sklearn.linear_model import Lasso

# ------------------------------------------------------------
# Funções auxiliares para cálculo de H0 e sazonalidade
# ------------------------------------------------------------
Gsc = 0.0820  # MJ m⁻² min⁻¹

def extraterrestrial_radiation(doy, lat_deg):
    """Calcula H0 (MJ/m²/dia) e duração do dia N (horas)."""
    phi = math.radians(lat_deg)
    delta = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    omega_s = math.acos(-math.tan(phi) * math.tan(delta))
    d_r = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    H0 = (24 * 60 / math.pi) * Gsc * d_r * (
        omega_s * math.sin(phi) * math.sin(delta) +
        math.cos(phi) * math.cos(delta) * math.sin(omega_s)
    )
    return H0

# ------------------------------------------------------------
# Carregar modelos salvos (com tratamento de erros)
# ------------------------------------------------------------
@st.cache_resource
def load_lasso_model(path='H_lasso.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    model = Lasso(alpha=data['alpha'])
    model.coef_ = np.array(data['coef_'])
    model.intercept_ = np.array(data['intercept_'])
    scaler_mean = np.array(data['scaler_mean'])
    scaler_scale = np.array(data['scaler_scale'])
    feature_names = data['feature_names']
    return model, scaler_mean, scaler_scale, feature_names

@st.cache_resource
def load_xgboost_model(model_path='H_xgboost.json', scaler_path='scaler_H_xgboost.json'):
    model = xgb.Booster()
    model.load_model(model_path)
    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)
    scaler_mean = np.array(scaler_data['scaler_mean'])
    scaler_scale = np.array(scaler_data['scaler_scale'])
    feature_names = scaler_data['feature_names']
    return model, scaler_mean, scaler_scale, feature_names

# ------------------------------------------------------------
# Interface do Streamlit
# ------------------------------------------------------------
st.set_page_config(page_title="Estimador de Radiação Solar", layout="centered")
st.title("☀️ Estimativa de Radiação Solar Global Diária")
st.markdown("""
Preencha os dados meteorológicos do dia para obter a radiação solar estimada pelos modelos 
**Lasso** e **XGBoost**, treinados para Itajubá - MG.
""")

# Entradas do usuário
col1, col2 = st.columns(2)
with col1:
    data = st.date_input("Data", value=datetime(2025, 1, 1))
    tmax = st.number_input("Temperatura máxima (°C)", value=30.0, step=0.1)
    tmin = st.number_input("Temperatura mínima (°C)", value=18.0, step=0.1)
    rh = st.number_input("Umidade relativa média (%)", value=65.0, min_value=0.0, max_value=100.0, step=0.1)
with col2:
    wind = st.number_input("Velocidade do vento (m/s)", value=2.0, step=0.1)
    pressure = st.number_input("Pressão atmosférica (hPa)", value=920.0, step=0.1)
    latitude = st.number_input("Latitude (°)", value=-22.42, step=0.01, 
                               help="Padrão: Itajubá. Altere somente se o modelo for recalibrado.")

# Botão de cálculo
if st.button("Estimar Radiação Solar"):
    # Calcular variáveis derivadas
    doy = data.timetuple().tm_yday
    H0 = extraterrestrial_radiation(doy, latitude)
    sin_doy = math.sin(2 * math.pi * doy / 365)
    cos_doy = math.cos(2 * math.pi * doy / 365)
    
    # Criar DataFrame com as features na ordem correta
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
    
    # 1. Previsão com Lasso
    try:
        lasso_model, lasso_mean, lasso_scale, lasso_features = load_lasso_model()
        X_lasso = entrada[lasso_features].values
        X_lasso_scaled = (X_lasso - lasso_mean) / lasso_scale
        pred_lasso = lasso_model.predict(X_lasso_scaled)[0]
        resultados['Lasso'] = pred_lasso
    except FileNotFoundError:
        st.warning("Modelo Lasso não encontrado (H_lasso.json).")
    except Exception as e:
        st.error(f"Erro ao carregar/prever com Lasso: {e}")
    
    # 2. Previsão com XGBoost
    try:
        xgb_model, xgb_mean, xgb_scale, xgb_features = load_xgboost_model()
        X_xgb = entrada[xgb_features].values
        X_xgb_scaled = (X_xgb - xgb_mean) / xgb_scale
        dmatrix = xgb.DMatrix(X_xgb_scaled, feature_names=xgb_features)
        pred_xgb = xgb_model.predict(dmatrix)[0]
        resultados['XGBoost'] = pred_xgb
    except FileNotFoundError:
        st.warning("Modelo XGBoost ou scaler não encontrado (H_xgboost.json / scaler_H_xgboost.json).")
    except Exception as e:
        st.error(f"Erro ao carregar/prever com XGBoost: {e}")
    
    # Exibir resultados
    if resultados:
        st.success("Estimativas geradas com sucesso!")
        # Tabela com os valores
        df_res = pd.DataFrame.from_dict(resultados, orient='index', columns=['Radiação (MJ/m²/dia)'])
        df_res.index.name = 'Modelo'
        st.table(df_res.style.format("{:.3f}"))
        
        # Explicação adicional
        st.markdown("""
        **Interpretação:**  
        Os valores correspondem à radiação solar global diária (onda curta) que atinge uma superfície horizontal, 
        em MJ/m² por dia. Esse é o principal insumo para modelos de produtividade agrícola (ex.: FAO-56).
        """)
    else:
        st.error("Nenhum modelo pôde ser carregado. Verifique se os arquivos JSON estão na pasta do aplicativo.")