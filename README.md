# 🚌 Detección de Camión Urbano R27/C-64 con YOLOv8 + Predicción de Horarios

Sistema de Visión Artificial que detecta en tiempo real el paso del camión urbano de la ruta **R27/C-64** en Guadalajara, envía alertas automáticas por Telegram y estima la hora de las próximas pasadas mediante un modelo de series de tiempo (Prophet).

**Integrante:** Pablo
**Materia:** 6E261 - Visión Artificial
**Institución:** CETI Plantel Colomos

---

## 🎯 Objetivo del proyecto

Aplicar los conceptos de Visión Artificial entrenando un modelo de la familia YOLO (YOLOv8) para reconocer un objeto específico —en este caso, el camión de una ruta urbana particular— y resolver un problema real: la incertidumbre sobre el horario de paso del transporte público. El sistema no solo detecta, sino que **notifica y predice**, integrando visión por computadora con análisis de series de tiempo.

---

## 🧠 ¿Cómo funciona?

1. Una cámara IP (celular Android con la app **IP Webcam**) transmite video en vivo enfocando el punto de la calle por donde pasa el camión.
2. Un modelo **YOLOv8** entrenado específicamente para reconocer el camión de la ruta R27/C-64 procesa el video frame por frame.
3. Cuando el modelo detecta el camión de forma consistente (varios frames seguidos, para evitar falsos positivos), el sistema:
   - Registra el timestamp de la detección.
   - Envía una alerta a Telegram con la hora exacta y el intervalo desde la última pasada.
4. Un modelo de **Prophet** (Meta/Facebook) usa el historial de pasadas para predecir los próximos horarios estimados.
5. El usuario puede preguntarle al bot de Telegram "¿a qué hora pasa?" y recibe una predicción basada en el patrón histórico de pasadas.

---

## 🛠️ Hardware y software utilizado

| Componente | Detalle |
|---|---|
| Cámara | Celular Xiaomi con app **IP Webcam** (streaming MJPEG por red local) |
| Modelo de detección | YOLOv8 (Ultralytics), entrenado con dataset propio |
| Notificaciones | Bot de Telegram (pyTelegramBotAPI) |
| Predicción de horarios | Prophet (modelo de series de tiempo) |
| Procesamiento de video | OpenCV |
| Lenguaje | Python 3 |

---

## 📂 Estructura del repositorio

```
testYolo/
├── train.py              # Script de entrenamiento del modelo YOLOv8
├── testYolo.py            # Script de prueba de detección sobre una imagen
├── main3.py                # Programa principal: captura de video + detección + bot + predicción
├── telegram_bot.py        # Script de prueba para verificar la conexión con el bot
├── requirements.txt        # Dependencias del proyecto
├── librerias.txt           # Lista de dependencias (formato alterno)
├── .env.example             # Plantilla de variables de entorno (sin credenciales reales)
├── .gitignore
└── README.md
```

> **Nota:** las carpetas `datasets_bus27/` (dataset de imágenes), `runs/` (resultados de entrenamiento) y los pesos `*.pt` no se incluyen en el repositorio por su peso. Ver sección de Evidencias.

---

## ⚙️ Instrucciones para correr el proyecto

### 1. Clonar el repositorio

```bash
git clone <URL_DE_TU_REPOSITORIO>
cd testYolo
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv yolo-env
source yolo-env/bin/activate      # En Windows: yolo-env\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Copia el archivo de ejemplo y agrega tus propias credenciales (token de bot de Telegram, chat ID y URL de tu cámara):

```bash
cp .env.example .env
```

Edita `.env` con tus datos:

```
TELEGRAM_TOKEN=tu_token_de_botfather
TELEGRAM_CHAT_ID=tu_chat_id
CAMERA_URL=http://IP_DE_TU_CAMARA:8080/video
```

> ⚠️ El archivo `.env` nunca debe subirse al repositorio (ya está incluido en `.gitignore`). Las credenciales son personales e intransferibles.

### 4. Entrenar el modelo (opcional, si quieres reentrenar)

```bash
python train.py
```

Esto genera los pesos entrenados en `runs/detect/camion_tp/weights/best.pt`.

### 5. Probar la detección sobre una imagen

```bash
python testYolo.py
```

### 6. Ejecutar el sistema completo (detección en vivo + bot + predicción)

```bash
python main3.py
```

Esto inicia tres hilos en paralelo:
- Captura continua del stream de la cámara IP.
- Inferencia de YOLOv8 sobre cada frame.
- Bot de Telegram escuchando mensajes y respondiendo predicciones.

---

## 🏭 Caso de Estudio: Aplicación en un entorno real

### Problema a resolver

En muchas rutas de transporte público urbano (como la R27/C-64 en Guadalajara) no existen horarios publicados ni aplicaciones oficiales de rastreo en tiempo real. Los usuarios deben esperar en la parada sin saber cuánto tiempo falta para que pase el siguiente camión, lo cual genera incertidumbre y pérdida de tiempo, especialmente en rutas con frecuencia irregular.

### Propuesta de solución

Se propone instalar un sistema de monitoreo fijo en un punto estratégico de la ruta (por ejemplo, una parada con buena visibilidad de la calle) que detecte automáticamente el paso del camión e informe a los usuarios registrados, además de aprender el patrón de frecuencia para predecir las próximas pasadas.

### Hardware propuesto

- **Cámara fija de vigilancia tipo IP** (en vez de un celular, para una instalación permanente), montada en un poste o fachada con vista directa al carril por donde circula el camión.
- **Mini-PC o Raspberry Pi 4/5** como unidad de procesamiento local, ejecutando el modelo YOLOv8 ya entrenado (cuantizado a formato ONNX o TensorRT para mejorar el rendimiento en hardware limitado).
- **Conexión a internet** (Wi-Fi o datos móviles vía módem 4G) para el envío de alertas a través del bot de Telegram.
- Opcionalmente, una **pantalla LED o letrero digital** en la parada del camión, conectado a la misma red, que muestre en vivo "Próxima pasada estimada: X min" sin que el usuario necesite el celular.

### Flujo de funcionamiento

1. La cámara IP transmite el video en vivo a la Raspberry Pi.
2. El modelo YOLOv8 corre inferencia continua sobre el stream, igual que en `main3.py`.
3. Al detectar el camión de forma consistente, el sistema:
   - Registra el timestamp en una base de datos (en producción, se reemplazaría el archivo `tiempos.json` local por una base de datos en la nube, como Firebase o PostgreSQL, para que múltiples puntos de monitoreo alimenten el mismo historial).
   - Envía la alerta a un canal o grupo de Telegram al que los usuarios de la ruta se suscriben.
4. El modelo Prophet, alimentado con el historial acumulado de pasadas (de días, semanas o meses), genera una predicción cada vez más precisa de los horarios típicos por franja horaria (hora pico vs. valle).
5. Los usuarios consultan la próxima hora estimada directamente desde el bot, sin necesidad de instalar una app adicional.

### Beneficio esperado

Reducir la incertidumbre del usuario de transporte público con una solución de bajo costo, reutilizando infraestructura existente (postes, cámaras de vigilancia urbana) y tecnología accesible (Raspberry Pi, modelos de detección ligeros), sin depender de que la propia ruta de camiones tenga GPS o telemetría instalada.

---

## 🖼️ Evidencias

Las imágenes y videos de prueba con las detecciones (bounding boxes) generadas por el modelo se encuentran en la carpeta `evidencias/` del repositorio.

---

## 📌 Notas técnicas

- El modelo usa un umbral de confianza alto (`conf=0.8`) y una lógica de histéresis (requiere detección en 3 frames consecutivos) para evitar falsos positivos por reflejos, otros vehículos similares o ruido visual.
- Se implementa un *cooldown* de 90 segundos entre notificaciones para no duplicar alertas del mismo paso del camión.
- Las credenciales (token de Telegram, chat ID, URL de cámara) se manejan mediante variables de entorno (`.env`), nunca hardcodeadas en el código fuente, siguiendo buenas prácticas de seguridad para repositorios públicos.
