"""
AGENTE DE INVESTIGACIÓN DE MERCADO 24/7 - v2 (sin IA, 100% gratis)
-------------------------------------------------------------------
Esto NO es asesoría financiera. Es una herramienta que te ayuda a
detectar posibles operaciones para que VOS decidas si las ejecutás
en XTB. Nunca ejecuta órdenes automáticamente.

Diferencia con la v1: acá no se usa ningún modelo de IA ni API paga.
Los setups se detectan con fórmulas de análisis técnico clásico:
medias móviles, RSI, y máximos/mínimos recientes. Todo se calcula
con datos gratuitos de Yahoo Finance.

Requisitos:
    pip install yfinance pandas numpy
"""

import os
import json
from datetime import datetime, timezone

import numpy as np
import yfinance as yf

# -----------------------------------------------------------------
# 1. CONFIGURACIÓN - editá esto a tu gusto
# -----------------------------------------------------------------

WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

# Ratio riesgo/beneficio mínimo para que una señal se considere válida
MIN_RATIO_RIESGO_BENEFICIO = 1.5

# Multiplicador de riesgo usado para fijar el objetivo (objetivo = entrada + N * riesgo)
MULTIPLICADOR_OBJETIVO = 2.0


# -----------------------------------------------------------------
# 2. TRAER DATOS Y CALCULAR INDICADORES
# -----------------------------------------------------------------

def calcular_rsi(cierres, periodo=14):
    """RSI clásico de Wilder."""
    delta = cierres.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    media_ganancia = ganancia.ewm(alpha=1 / periodo, min_periods=periodo).mean()
    media_perdida = perdida.ewm(alpha=1 / periodo, min_periods=periodo).mean()
    rs = media_ganancia / media_perdida.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def obtener_datos(ticker: str):
    """Trae 90 días de historial y calcula todos los indicadores necesarios."""
    t = yf.Ticker(ticker)
    hist = t.history(period="90d")
    if hist.empty or len(hist) < 55:
        return None

    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    hist["RSI14"] = calcular_rsi(hist["Close"])
    hist["VOL_PROM20"] = hist["Volume"].rolling(20).mean()
    hist["MAX20"] = hist["High"].rolling(20).max()
    hist["MIN20"] = hist["Low"].rolling(20).min()

    ultimo = hist.iloc[-1]
    anteultimo = hist.iloc[-2]

    return {
        "ticker": ticker,
        "precio_actual": round(float(ultimo["Close"]), 2),
        "variacion_pct": round(float((ultimo["Close"] - anteultimo["Close"]) / anteultimo["Close"] * 100), 2),
        "variacion_5d_pct": round(float((ultimo["Close"] - hist["Close"].iloc[-6]) / hist["Close"].iloc[-6] * 100), 2),
        "volumen_hoy": float(ultimo["Volume"]),
        "volumen_prom20": float(ultimo["VOL_PROM20"]),
        "sma20": round(float(ultimo["SMA20"]), 2),
        "sma50": round(float(ultimo["SMA50"]), 2),
        "rsi14": round(float(ultimo["RSI14"]), 1) if not np.isnan(ultimo["RSI14"]) else None,
        "max20": round(float(ultimo["MAX20"]), 2),
        "min20": round(float(ultimo["MIN20"]), 2),
        "high_hoy": round(float(ultimo["High"]), 2),
        "low_hoy": round(float(ultimo["Low"]), 2),
    }


# -----------------------------------------------------------------
# 3. REGLAS DE DETECCIÓN (reemplaza al modelo de IA)
# -----------------------------------------------------------------

def construir_plan(entrada, stop_loss, es_long=True):
    """Calcula objetivo y ratio riesgo/beneficio a partir de entrada y stop."""
    riesgo = abs(entrada - stop_loss)
    if riesgo == 0:
        return None, None
    if es_long:
        objetivo = entrada + riesgo * MULTIPLICADOR_OBJETIVO
    else:
        objetivo = entrada - riesgo * MULTIPLICADOR_OBJETIVO
    ratio = round((abs(objetivo - entrada)) / riesgo, 2)
    return round(objetivo, 2), ratio


def evaluar_setup(d: dict) -> dict:
    """Aplica las 5 reglas en orden de prioridad y devuelve la primera que aplique."""

    precio = d["precio_actual"]
    volumen_alto = d["volumen_hoy"] > d["volumen_prom20"] * 1.3
    tendencia_alcista = d["sma20"] > d["sma50"]
    tendencia_bajista = d["sma20"] < d["sma50"]

    # 1) RUPTURA: el precio de hoy superó el máximo de los últimos 20 días, con volumen alto
    if d["high_hoy"] >= d["max20"] * 0.999 and volumen_alto:
        stop = round(d["min20"], 2)
        objetivo, ratio = construir_plan(precio, stop, es_long=True)
        return {
            "hay_señal": True, "tipo_setup": "ruptura", "direccion": "long", "confianza": "alta",
            "entrada": precio, "stop_loss": stop, "objetivo": objetivo, "ratio_riesgo_beneficio": ratio,
            "razon": f"El precio rompió el máximo de los últimos 20 días con un volumen {round(d['volumen_hoy']/d['volumen_prom20'],1)}x el promedio, lo que sugiere que compradores fuertes están entrando.",
        }

    # 2) MOMENTUM: variación de 5 días fuerte + volumen creciente
    if abs(d["variacion_5d_pct"]) >= 6 and volumen_alto:
        es_long = d["variacion_5d_pct"] > 0
        stop = round(d["min20"], 2) if es_long else round(d["max20"], 2)
        objetivo, ratio = construir_plan(precio, stop, es_long=es_long)
        return {
            "hay_señal": True, "tipo_setup": "momentum", "direccion": "long" if es_long else "short", "confianza": "media",
            "entrada": precio, "stop_loss": stop, "objetivo": objetivo, "ratio_riesgo_beneficio": ratio,
            "razon": f"El precio se movió {d['variacion_5d_pct']}% en los últimos 5 días con volumen por encima del promedio, mostrando una aceleración fuerte en esa dirección.",
        }

    # 3) PULLBACK: tendencia alcista clara, precio retrocedió cerca de la SMA20 sin romperla
    if tendencia_alcista and precio > d["sma20"] * 0.98 and precio < d["sma20"] * 1.02 and d["variacion_pct"] < 0:
        stop = round(d["sma20"] * 0.97, 2)
        objetivo, ratio = construir_plan(precio, stop, es_long=True)
        return {
            "hay_señal": True, "tipo_setup": "pullback", "direccion": "long", "confianza": "media",
            "entrada": precio, "stop_loss": stop, "objetivo": objetivo, "ratio_riesgo_beneficio": ratio,
            "razon": "La tendencia de fondo es alcista (media de 20 días por encima de la de 50) y el precio retrocedió hasta esa media sin romper la estructura.",
        }

    # 4) REVERSIÓN: RSI en zona extrema
    if d["rsi14"] is not None and (d["rsi14"] >= 72 or d["rsi14"] <= 28):
        es_long = d["rsi14"] <= 28
        stop = round(d["min20"], 2) if es_long else round(d["max20"], 2)
        objetivo, ratio = construir_plan(precio, stop, es_long=es_long)
        return {
            "hay_señal": True, "tipo_setup": "reversion", "direccion": "long" if es_long else "short", "confianza": "media",
            "entrada": precio, "stop_loss": stop, "objetivo": objetivo, "ratio_riesgo_beneficio": ratio,
            "razon": f"El RSI está en {d['rsi14']}, una zona de {'sobreventa' if es_long else 'sobrecompra'} que históricamente anticipa un posible giro.",
        }

    # 5) CONTINUACIÓN / SIN SEÑAL
    if tendencia_alcista or tendencia_bajista:
        return {
            "hay_señal": False, "tipo_setup": None, "direccion": None, "confianza": None,
            "entrada": None, "stop_loss": None, "objetivo": None, "ratio_riesgo_beneficio": None,
            "razon": f"Tendencia {'alcista' if tendencia_alcista else 'bajista'} de fondo, pero sin un gatillo claro (ni ruptura, ni pullback a la media, ni RSI extremo) todavía.",
        }

    return {
        "hay_señal": False, "tipo_setup": None, "direccion": None, "confianza": None,
        "entrada": None, "stop_loss": None, "objetivo": None, "ratio_riesgo_beneficio": None,
        "razon": "Sin tendencia clara ni patrón reconocible por ahora.",
    }


def pasa_filtro_de_calidad(señal: dict) -> bool:
    if not señal.get("hay_señal"):
        return False
    ratio = señal.get("ratio_riesgo_beneficio")
    if ratio is None or ratio < MIN_RATIO_RIESGO_BENEFICIO:
        return False
    return True


# -----------------------------------------------------------------
# 4. GUARDAR RESULTADOS
# -----------------------------------------------------------------

def guardar_log(resultados: list):
    os.makedirs("logs", exist_ok=True)
    momento = datetime.now(timezone.utc)

    nombre_historico = f"logs/escaneo_{momento.strftime('%Y%m%d_%H%M')}.json"
    with open(nombre_historico, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    paquete = {"actualizado": momento.isoformat(), "resultados": resultados}
    with open("logs/latest.json", "w", encoding="utf-8") as f:
        json.dump(paquete, f, ensure_ascii=False, indent=2)

    print(f"\nLog guardado en: {nombre_historico}")
    print("Archivo logs/latest.json actualizado (esto lo lee el dashboard).")


# -----------------------------------------------------------------
# 5. PROGRAMA PRINCIPAL
# -----------------------------------------------------------------

def main():
    print(f"Escaneando {len(WATCHLIST)} acciones... ({datetime.now(timezone.utc).isoformat()})\n")

    resultados = []
    señales_validas = []

    for ticker in WATCHLIST:
        datos = obtener_datos(ticker)
        if datos is None:
            print(f"[{ticker}] No se pudieron obtener suficientes datos.")
            continue

        señal = evaluar_setup(datos)
        señal["ticker"] = ticker
        señal["precio_al_momento_del_analisis"] = datos["precio_actual"]
        resultados.append(señal)

        if pasa_filtro_de_calidad(señal):
            señales_validas.append(señal)
            print(f"[{ticker}] ✅ SEÑAL: {señal['tipo_setup']} ({señal['direccion']}) "
                  f"- Entrada: {señal['entrada']} / Stop: {señal['stop_loss']} / "
                  f"Objetivo: {señal['objetivo']} / R:B 1:{señal['ratio_riesgo_beneficio']}")
        else:
            print(f"[{ticker}] Sin señal de alta calidad por ahora.")

    guardar_log(resultados)

    print(f"\nTotal de señales de alta calidad encontradas: {len(señales_validas)}")
    return señales_validas


if __name__ == "__main__":
    main()
