from prophet import Prophet
import time
import threading
import pandas as pd
import cv2
import datetime
import telebot
import json
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CAMERA_URL = os.getenv("CAMERA_URL", "http://192.168.100.4:8080/video")

if not TOKEN or not CHAT_ID:
    raise RuntimeError("Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID. Configura tu archivo .env (ver .env.example).")

bot = telebot.TeleBot(TOKEN)
model_yolo = YOLO("runs/detect/camion_tp/weights/best.pt")

ARCHIVO_TIEMPOS = "tiempos.json"

# Variables globales para compartir el frame entre hilos de forma segura
lock = threading.Lock()
ultimo_frame = None
corriendo = True

def cargar_tiempos():
    if os.path.exists(ARCHIVO_TIEMPOS):
        with open(ARCHIVO_TIEMPOS, 'r') as f:
            datos = json.load(f)
            return [datetime.datetime.fromisoformat(t) for t in datos]
    return []

def guardar_tiempos():
    with open(ARCHIVO_TIEMPOS, 'w') as f:
        json.dump([t.isoformat() for t in tiempos], f)

tiempos = cargar_tiempos()
errores = []

def camion_detectado():
    ahora = datetime.datetime.now()
    tiempos.append(ahora)
    guardar_tiempos()

    mensaje = f"🚌 Camión R27/C-64 detectado\n🕐 {ahora.strftime('%H:%M:%S')}"

    if len(tiempos) > 1:
        intervalo = (tiempos[-1] - tiempos[-2]).seconds // 60
        promedio = sum([(tiempos[i+1]-tiempos[i]).seconds//60
                       for i in range(len(tiempos)-1)]) / (len(tiempos)-1)
        mensaje += f"\n⏱ Último intervalo: {intervalo} min"
        mensaje += f"\n📊 Promedio de pasada: {promedio:.1f} min"
    try:
        bot.send_message(CHAT_ID, mensaje)
    except Exception as e:
        errores.append(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {str(e)}")

@bot.message_handler(func=lambda msg: True)
def responder(message):
    print(f"Mensaje recibido: {message.text}")
    if "cuando" in message.text.lower() or "hora" in message.text.lower():
        if len(tiempos) < 5:
            bot.reply_to(message, f"Aún no tengo suficientes datos ({len(tiempos)}/5), espera que pasen más camiones.")
            return

        df = pd.DataFrame({
            'ds': tiempos,
            'y': range(len(tiempos))
        })

        m = Prophet()
        m.fit(df)

        ahora = pd.Timestamp.now()
        ultimo_dato = df['ds'].max()
        minutos_diferencia = int((ahora - ultimo_dato).total_seconds() / 60) + 30

        futuro = m.make_future_dataframe(periods=minutos_diferencia, freq='min')
        prediccion = m.predict(futuro)
        proximas = prediccion[prediccion['ds'] > ahora]['ds'].head(5)

        mensaje = "🚌 Próximas pasadas estimadas:\n"
        for hora in proximas:
            mensaje += f"• {hora.strftime('%H:%M')}\n"

        bot.reply_to(message, mensaje)
    else:
        bot.reply_to(message, "Pregúntame: ¿a qué hora pasa el siguiente?")

def hilo_bot():
    bot.polling(none_stop=True)

# 🧵 HILO NUEVO: Solo se dedica a leer la cámara IP para que el buffer nunca se acumule
def hilo_captura_video():
    global ultimo_frame, corriendo
    cap = cv2.VideoCapture(CAMERA_URL)
    
    while corriendo:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue
        
        # Guardamos el frame más nuevo bloqueando el hilo un milisegundo
        with lock:
            ultimo_frame = frame.copy()
            
    cap.release()

# 🧵 HILO PROCESAMIENTO: Consume el frame más nuevo y corre YOLO sin retrasos de red
def hilo_yolo_procesamiento():
    global ultimo_frame, corriendo
    
    COOLDOWN = 90             # 1 minuto y medio entre camiones reales
    FRAMES_REQUERIDOS = 3     # Debe aparecer en 3 frames consecutivos
    ultimo_aviso = 0
    conteo_frames_camion = 0

    while corriendo:
        frame_actual = None
        
        # Obtenemos el último frame disponible
        with lock:
            if ultimo_frame is not None:
                frame_actual = ultimo_frame.copy()
        
        # Si aún no hay frame disponible, esperamos un poco
        if frame_actual is None:
            time.sleep(0.01)
            continue

        # Pasar el frame por el modelo de IA
        results = model_yolo(frame_actual, conf=0.8, iou=0.5)
        detecciones = results[0].boxes
        ahora = time.time()

        # Lógica de histéresis anti-falsos positivos
        if len(detecciones) > 0:
            conteo_frames_camion += 1
        else:
            conteo_frames_camion = 0

        if conteo_frames_camion >= FRAMES_REQUERIDOS and (ahora - ultimo_aviso) > COOLDOWN:
            ultimo_aviso = ahora
            try:
                camion_detectado()
            except Exception as e:
                errores.append(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {str(e)}")
            conteo_frames_camion = 0

        # Mostrar visualización
        annotated = results[0].plot()
        cv2.imshow("Deteccion camion_tp", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            corriendo = False
            break

    cv2.destroyAllWindows()


# --- INICIALIZACIÓN DE LOS 3 HILOS ---
t_bot = threading.Thread(target=hilo_bot, daemon=True)
t_captura = threading.Thread(target=hilo_captura_video, daemon=True)
t_yolo = threading.Thread(target=hilo_yolo_procesamiento, daemon=True)

t_bot.start()
t_captura.start()
t_yolo.start()

try:
    while corriendo:
        time.sleep(1)
except KeyboardInterrupt:
    print("Cerrando...")
    corriendo = False
    if errores:
        resumen = "Errores durante la sesion:\n" + "\n".join(errores)
        try:
            bot.send_message(CHAT_ID, resumen)
        except:
            print(resumen)
