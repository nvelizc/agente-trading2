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
import time
from datetime import datetime, timezone

import numpy as np
import requests
import yfinance as yf

# -----------------------------------------------------------------
# 1. CONFIGURACIÓN - editá esto a tu gusto
# -----------------------------------------------------------------

# Watchlist organizada por sector — editá libremente
WATCHLIST_POR_SECTOR = {
    "Tecnología": ["AAPL", "MSFT", "NVDA"],
    "Finanzas": ["JPM", "V"],
    "Energía": ["XOM", "CVX"],
    "Consumo": ["AMZN", "KO"],
    "Índices": ["SPY", "QQQ"],
}
WATCHLIST = [ticker for tickers in WATCHLIST_POR_SECTOR.values() for ticker in tickers]

# Cuántos tickers en tendencia sumar por corrida (gainers + losers + most actives)
CANTIDAD_TENDENCIAS = 8


def obtener_tickers_en_tendencia(cantidad: int = CANTIDAD_TENDENCIAS) -> list:
    """Trae acciones que más se mueven HOY (más suben, más bajan, más volumen),
    que suele reflejar dónde está la atención de las noticias financieras en ese momento."""
    encontrados = []
    for screener in ["day_gainers", "day_losers", "most_actives"]:
        try:
            resultado = yf.screen(screener, count=6)
            quotes = resultado.get("quotes", []) if isinstance(resultado, dict) else []
            for q in quotes:
                simbolo = q.get("symbol")
                if simbolo and simbolo not in encontrados:
                    encontrados.append(simbolo)
        except Exception as e:
            print(f"[DEBUG] No se pudo traer el screener '{screener}': {type(e).__name__} -> {e}")
    return encontrados[:cantidad]


def sector_de(ticker: str) -> str:
    for sector, tickers in WATCHLIST_POR_SECTOR.items():
        if ticker in tickers:
            return sector
    return "Tendencia del día"

# Ratio riesgo/beneficio mínimo para que una señal se considere válida
MIN_RATIO_RIESGO_BENEFICIO = 1.5

# Multiplicador de riesgo usado para fijar el objetivo (objetivo = entrada + N * riesgo)
MULTIPLICADOR_OBJETIVO = 2.0

# Tope máximo de distancia del stop respecto al precio de entrada (8% = riesgo razonable)
MAX_RIESGO_PORCENTAJE = 0.08

# Modelo de Gemini a usar para analizar noticias (gratis en Google AI Studio)
MODELO_GEMINI = "gemini-3.6-flash"
CANTIDAD_TITULARES = 5


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

    historial_precios = [
        {"fecha": idx.strftime("%Y-%m-%d"), "cierre": round(float(val), 2)}
        for idx, val in hist["Close"].tail(30).items()
    ]

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
        "historial_precios": historial_precios,
    }


# -----------------------------------------------------------------
# 3. REGLAS DE DETECCIÓN (reemplaza al modelo de IA)
# -----------------------------------------------------------------

def limitar_stop(entrada, stop_crudo, es_long=True):
    """Si el stop calculado queda demasiado lejos del precio (por una suba/baja muy fuerte
    reciente), lo acerca a un máximo razonable en vez de usar el mínimo/máximo de 20 días tal cual."""
    distancia_pct = abs(entrada - stop_crudo) / entrada
    if distancia_pct <= MAX_RIESGO_PORCENTAJE:
        return stop_crudo
    if es_long:
        return round(entrada * (1 - MAX_RIESGO_PORCENTAJE), 2)
    return round(entrada * (1 + MAX_RIESGO_PORCENTAJE), 2)


def construir_plan(entrada, stop_crudo, es_long=True):
    """Calcula objetivo y ratio riesgo/beneficio a partir de entrada y stop, limitando el riesgo."""
    stop_loss = limitar_stop(entrada, stop_crudo, es_long)
    riesgo = abs(entrada - stop_loss)
    if riesgo == 0:
        return None, None, None
    if es_long:
        objetivo = entrada + riesgo * MULTIPLICADOR_OBJETIVO
    else:
        objetivo = entrada - riesgo * MULTIPLICADOR_OBJETIVO
    ratio = round((abs(objetivo - entrada)) / riesgo, 2)
    return stop_loss, round(objetivo, 2), ratio


def evaluar_setup(d: dict) -> dict:
    """Aplica las 5 reglas en orden de prioridad y devuelve la primera que aplique."""

    precio = d["precio_actual"]
    volumen_alto = d["volumen_hoy"] > d["volumen_prom20"] * 1.3
    tendencia_alcista = d["sma20"] > d["sma50"]
    tendencia_bajista = d["sma20"] < d["sma50"]

    # 1) RUPTURA: el precio de hoy superó el máximo de los últimos 20 días, con volumen alto
    if d["high_hoy"] >= d["max20"] * 0.999 and volumen_alto:
        stop_crudo = round(d["min20"], 2)
        stop, objetivo, ratio = construir_plan(precio, stop_crudo, es_long=True)
        return {
            "hay_señal": True, "tipo_setup": "ruptura", "direccion": "long", "confianza": "alta",
            "entrada": precio, "stop_loss": stop, "objetivo": objetivo, "ratio_riesgo_beneficio": ratio,
            "razon": f"El precio rompió el máximo de los últimos 20 días con un volumen {round(d['volumen_hoy']/d['volumen_prom20'],1)}x el promedio, lo que sugiere que compradores fuertes están entrando.",
        }

    # 2) MOMENTUM: variación de 5 días fuerte + volumen creciente
    if abs(d["variacion_5d_pct"]) >= 6 and volumen_alto:
        es_long = d["variacion_5d_pct"] > 0
        stop_crudo = round(d["min20"], 2) if es_long else round(d["max20"], 2)
        stop, objetivo, ratio = construir_plan(precio, stop_crudo, es_long=es_long)
        return {
            "hay_señal": True, "tipo_setup": "momentum", "direccion": "long" if es_long else "short", "confianza": "media",
            "entrada": precio, "stop_loss": stop, "objetivo": objetivo, "ratio_riesgo_beneficio": ratio,
            "razon": f"El precio se movió {d['variacion_5d_pct']}% en los últimos 5 días con volumen por encima del promedio, mostrando una aceleración fuerte en esa dirección.",
        }

    # 3) PULLBACK: tendencia alcista clara, precio retrocedió cerca de la SMA20 sin romperla
    if tendencia_alcista and precio > d["sma20"] * 0.98 and precio < d["sma20"] * 1.02 and d["variacion_pct"] < 0:
        stop_crudo = round(d["sma20"] * 0.97, 2)
        stop, objetivo, ratio = construir_plan(precio, stop_crudo, es_long=True)
        return {
            "hay_señal": True, "tipo_setup": "pullback", "direccion": "long", "confianza": "media",
            "entrada": precio, "stop_loss": stop, "objetivo": objetivo, "ratio_riesgo_beneficio": ratio,
            "razon": "La tendencia de fondo es alcista (media de 20 días por encima de la de 50) y el precio retrocedió hasta esa media sin romper la estructura.",
        }

    # 4) REVERSIÓN: RSI en zona extrema
    if d["rsi14"] is not None and (d["rsi14"] >= 72 or d["rsi14"] <= 28):
        es_long = d["rsi14"] <= 28
        stop_crudo = round(d["min20"], 2) if es_long else round(d["max20"], 2)
        stop, objetivo, ratio = construir_plan(precio, stop_crudo, es_long=es_long)
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
# 3B. NOTICIAS + ANÁLISIS DE SENTIMIENTO CON GEMINI (capa de IA)
# -----------------------------------------------------------------

def obtener_noticias(ticker: str, cantidad: int = CANTIDAD_TITULARES) -> list:
    """Trae noticias recientes de un ticker con título, link y fuente (gratis, sin API key)."""
    try:
        t = yf.Ticker(ticker)
        crudo = t.news or []
        noticias = []
        for n in crudo[:cantidad]:
            contenido = n.get("content", n)  # yfinance a veces anida bajo "content"
            titulo = contenido.get("title") or n.get("title")
            if not titulo:
                continue

            link = None
            if isinstance(contenido.get("canonicalUrl"), dict):
                link = contenido["canonicalUrl"].get("url")
            elif isinstance(contenido.get("clickThroughUrl"), dict):
                link = contenido["clickThroughUrl"].get("url")

            fuente = None
            if isinstance(contenido.get("provider"), dict):
                fuente = contenido["provider"].get("displayName")

            noticias.append({"titulo": titulo, "link": link, "fuente": fuente})
        return noticias
    except Exception:
        return []


def analizar_sentimiento_noticias(api_key: str, ticker: str, noticias: list) -> dict:
    """Le pide a Gemini que evalúe el sentimiento de las noticias y dé una perspectiva de corto plazo."""
    if not noticias:
        return {
            "sentimiento": "neutral",
            "resumen": "No hay titulares recientes disponibles.",
            "perspectiva": "lateral",
            "prediccion": "Sin noticias recientes suficientes para armar una perspectiva.",
        }

    lista_titulares = "\n".join(f"- {n['titulo']}" for n in noticias)
    prompt = f"""Sos un analista financiero. Estos son los titulares de noticias más recientes sobre {ticker}:

{lista_titulares}

Con base SOLO en estos titulares (no inventes datos que no estén ahí), armá:
1. El sentimiento general para alguien pensando en operar esta acción en el corto plazo.
2. Una perspectiva de hacia dónde podría inclinarse el precio en los próximos días: alcista, bajista, o lateral.
3. Una predicción breve (2-3 oraciones, en español simple, para alguien que no sabe de trading) explicando el porqué,
   dejando claro que es una lectura de noticias y NO una certeza.

Respondé SOLO con un JSON, sin texto adicional ni markdown, con esta forma exacta:

{{
  "sentimiento": "positivo" o "negativo" o "neutral",
  "resumen": "una oración breve explicando el sentimiento",
  "perspectiva": "alcista" o "bajista" o "lateral",
  "prediccion": "2-3 oraciones explicando la perspectiva de corto plazo basada en estas noticias"
}}
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO_GEMINI}:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    for intento in range(2):  # probamos hasta 2 veces antes de rendirnos
        try:
            resp = requests.post(url, json=body, timeout=45)
            resp.raise_for_status()
            data = resp.json()
            texto = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            texto = texto.replace("```json", "").replace("```", "").strip()
            return json.loads(texto)
        except requests.exceptions.HTTPError as e:
            print(f"[DEBUG Gemini] {ticker} (intento {intento+1}): HTTP {e.response.status_code} -> {e.response.text[:300]}")
        except Exception as e:
            print(f"[DEBUG Gemini] {ticker} (intento {intento+1}): {type(e).__name__} -> {e}")
        time.sleep(8)  # esperamos antes de reintentar

    return {
        "sentimiento": "neutral",
        "resumen": "No se pudo analizar el sentimiento después de reintentar.",
        "perspectiva": "lateral",
        "prediccion": "No se pudo generar una perspectiva por un problema técnico al consultar el modelo.",
    }


def combinar_señal_con_noticias(señal: dict, análisis: dict) -> dict:
    """Agrega la predicción de noticias como campos propios y ajusta la confianza de la señal técnica."""
    señal["noticias_sentimiento"] = análisis.get("sentimiento")
    señal["noticias_resumen"] = análisis.get("resumen")
    señal["noticias_perspectiva"] = análisis.get("perspectiva")
    señal["noticias_prediccion"] = análisis.get("prediccion")

    if not señal.get("hay_señal"):
        return señal

    perspectiva = análisis.get("perspectiva")
    alineado = (
        (señal["direccion"] == "long" and perspectiva == "alcista")
        or (señal["direccion"] == "short" and perspectiva == "bajista")
    )
    contradice = (
        (señal["direccion"] == "long" and perspectiva == "bajista")
        or (señal["direccion"] == "short" and perspectiva == "alcista")
    )

    if alineado:
        señal["confianza"] = "alta"
    elif contradice:
        señal["confianza"] = "baja"
    # si es lateral, se deja la confianza que ya tenía el análisis técnico

    return señal


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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Aviso: no encontré GEMINI_API_KEY. El análisis técnico va a correr igual, "
              "pero sin la capa de noticias.\n")

    tickers_tendencia = obtener_tickers_en_tendencia()
    tickers_a_escanear = WATCHLIST + [t for t in tickers_tendencia if t not in WATCHLIST]

    print(f"Escaneando {len(tickers_a_escanear)} acciones "
          f"({len(WATCHLIST)} fijas + {len(tickers_tendencia)} en tendencia hoy)... "
          f"({datetime.now(timezone.utc).isoformat()})\n")

    resultados = []
    señales_validas = []

    for ticker in tickers_a_escanear:
        datos = obtener_datos(ticker)
        if datos is None:
            print(f"[{ticker}] No se pudieron obtener suficientes datos.")
            continue

        señal = evaluar_setup(datos)
        señal["ticker"] = ticker
        señal["sector"] = sector_de(ticker)
        señal["precio_al_momento_del_analisis"] = datos["precio_actual"]
        señal["historial_precios"] = datos["historial_precios"]

        noticias = obtener_noticias(ticker)
        señal["noticias"] = noticias

        if api_key:
            sentimiento = analizar_sentimiento_noticias(api_key, ticker, noticias)
            señal = combinar_señal_con_noticias(señal, sentimiento)
            time.sleep(3)

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
