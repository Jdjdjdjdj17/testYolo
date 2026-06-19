import telebot
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError("Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID. Configura tu archivo .env (ver .env.example).")

bot = telebot.TeleBot(TOKEN)

def enviar_alerta(mensaje):
    bot.send_message(CHAT_ID, mensaje)

# Prueba
enviar_alerta("🚌 Bot funcionando correctamente")
