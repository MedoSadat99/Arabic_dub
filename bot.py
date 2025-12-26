# bot.py
import os
import sys
import re
import logging
import tempfile
import textwrap
import subprocess
import asyncio
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)
import deepl
from langdetect import detect
from pydub import AudioSegment
import soundfile as sf
import whisper
import youtube_dl
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import torch

# --- تثبيت ffmpeg تلقائيًا (مهم لـ Render) ---
def install_ffmpeg():
    if not os.path.exists("/usr/bin/ffmpeg"):
        print("🔧 جاري تثبيت ffmpeg...")
        subprocess.run(["apt-get", "update"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ تم تثبيت ffmpeg.")

install_ffmpeg()

# --- تحميل Coqui TTS (مع معالجة الأخطاء) ---
print("🔄 جاري تحميل نماذج الصوت والنص... (قد يستغرق 5-15 دقيقة في المرة الأولى)")

# تحميل Whisper
whisper_model = whisper.load_model("base.en")
print("✅ Whisper جاهز.")

# تحميل Coqui TTS
try:
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)  # بدون GPU
    print("✅ Coqui TTS جاهز!")
except Exception as e:
    print(f"❌ فشل تحميل Coqui TTS: {e}")
    sys.exit(1)

# --- الإعدادات ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

if not TELEGRAM_BOT_TOKEN or not DEEPL_API_KEY:
    raise ValueError("❌ خطأ: يجب تعيين TELEGRAM_BOT_TOKEN و DEEPL_API_KEY في Render Environment Variables!")

translator = deepl.Translator(DEEPL_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- دوال البوت ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "مرحباً! 🎧\n"
        "أرسل لي:\n"
        "• ملف (PDF, DOCX, TXT, MP3, WAV)\n"
        "• أو رابط يوتيوب\n\n"
        "وسأقوم بدبلجته إلى صوت عربي بشري احترافي!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if not text:
        return
    if re.search(r"(youtube\.com|youtu\.be)", text):
        await update.message.reply_text("🎥 جاري معالجة رابط يوتيوب...")
        await process_youtube(update, text)
    else:
        await update.message.reply_text("📩 يُرجى إرسال ملف أو رابط يوتيوب.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📥 جاري تنزيل الملف...")
    file = await update.message.document.get_file()
    await process_file(update, file, update.message.document.file_name)

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔊 جاري معالجة الملف الصوتي...")
    if update.message.voice:
        file = await update.message.voice.get_file()
        filename = "voice.ogg"
    else:
        file = await update.message.audio.get_file()
        filename = update.message.audio.file_name or "audio.mp3"
    await process_file(update, file, filename)

# --- معالجة يوتيوب ---
async def process_youtube(update: Update, url: str):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(tmpdir, 'audio'),
                'quiet': True,
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # البحث عن ملف WAV
            wav_files = [f for f in os.listdir(tmpdir) if f.endswith('.wav')]
            if not wav_files:
                raise FileNotFoundError("لم يتم إنشاء ملف صوتي من يوتيوب.")
            wav_path = os.path.join(tmpdir, wav_files[0])
            
            # تحويل الصوت إلى نص
            result = whisper_model.transcribe(wav_path, language="en")
            await generate_and_send_output(update, result["text"])
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ في يوتيوب: {str(e)}")

# --- معالجة الملفات ---
async def process_file(update: Update, file, filename):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, filename or "file")
            await file.download_to_drive(file_path)

            # إذا كان ملف صوت وليس WAV، نحوله
            if filename and not filename.lower().endswith('.wav'):
                if any(ext in filename.lower() for ext in ['.mp3', '.m4a', '.ogg', '.oga']):
                    new_path = os.path.join(tmpdir, "audio.wav")
                    subprocess.run(['ffmpeg', '-i', file_path, new_path], 
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    file_path = new_path

            # استخراج النص
            if filename and any(ext in filename.lower() for ext in ['.mp3', '.wav', '.m4a', '.ogg']):
                result = whisper_model.transcribe(file_path, language="en")
                original_text = result["text"]
            else:
                # ملف نصي
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.pdf':
                    text = ""
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text += page.extract_text() or ""
                    original_text = text
                elif ext == '.docx':
                    doc = Document(file_path)
                    original_text = '\n'.join([p.text for p in doc.paragraphs])
                elif ext in ['.txt', '.md', '.rtf', '.html', '.htm']:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if ext in ['.html', '.htm']:
                            soup = BeautifulSoup(content, 'html.parser')
                            original_text = soup.get_text()
                        else:
                            original_text = content
                else:
                    original_text = ""

            await generate_and_send_output(update, original_text)
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ في المعالجة: {str(e)}")

# --- توليد الصوت باستخدام Coqui (الصوت البشري!) ---
async def generate_and_send_output(update: Update, original_text: str):
    if not original_text or not original_text.strip():
        await update.message.reply_text("❌ لم يتم استخراج أي نص.")
        return

    # كشف اللغة
    try:
        lang = detect(original_text[:3000])
    except:
        lang = "en"

    # ترجمة إذا كان إنجليزيًا
    if lang == "en":
        await update.message.reply_text("🔄 جاري الترجمة إلى العربية...")
        chunks = textwrap.wrap(original_text, width=10000)
        arabic_text = ''.join([
            translator.translate_text(chunk, source_lang="EN", target_lang="AR").text
            for chunk in chunks
        ])
    else:
        arabic_text = original_text

    # حفظ النص
    txt_path = "/tmp/النص_العربي.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(arabic_text)

    # توليد الصوت
    await update.message.reply_text("🎙️ جاري إنشاء الصوت البشري الاحترافي... (قد يستغرق 30-90 ثانية)")

    # تقسيم النص إلى جمل
    sentences = re.split(r'(?<=[.،!؟])\s+', arabic_text.strip())
    full_audio = []

    for i, sent in enumerate(sentences):
        if not sent.strip():
            continue
        temp_wav = f"/tmp/part_{i}.wav"
        try:
            tts.tts_to_file(
                text=sent,
                file_path=temp_wav,
                speaker="Ana Florence",  # متحدث يدعم العربية
                language="ar",
                split_sentences=False
            )
            full_audio.append(AudioSegment.from_wav(temp_wav))
            full_audio.append(AudioSegment.silent(duration=400))  # توقف طبيعي
            os.remove(temp_wav)
        except Exception as e:
            logger.warning(f"تخطي جملة بسبب خطأ: {e}")
            continue

    if not full_audio:
        await update.message.reply_text("❌ فشل توليد الصوت.")
        return

    # دمج الأجزاء
    final_audio = sum(full_audio[:-1]) if len(full_audio) > 1 else full_audio[0]
    mp3_path = "/tmp/الدبلجة_البشرية.mp3"
    final_audio.export(mp3_path, format="mp3", bitrate="192k")

    # إرسال النتائج
    await update.message.reply_document(
        document=open(txt_path, 'rb'),
        caption="📄 النص العربي المترجم"
    )
    await update.message.reply_audio(
        audio=open(mp3_path, 'rb'),
        caption="🎧 الصوت البشري الاحترافي (Coqui TTS)"
    )

    # تنظيف
    for path in [txt_path, mp3_path]:
        if os.path.exists(path):
            os.remove(path)

# --- التشغيل الرئيسي ---
if __name__ == "__main__":
    print("🚀 جاري تشغيل بوت الدبلجة على Render...")
    print("✅ تأكد من تعيين المتغيرات البيئية في Render!")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))

    print("✅ البوت قيد التشغيل!")
    app.run_polling()
