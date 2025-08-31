import feedparser
import aiohttp
import asyncio
import logging
import os
import re
import json
import torch
import difflib
from sentence_transformers import SentenceTransformer, util
from aiogram import Bot
from dotenv import load_dotenv
from aiogram.types import InputMediaPhoto, InputMediaVideo
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
DELAY_SECONDS = int(os.getenv("DELAY_SECONDS"))
RSS_FEED_URLS = os.getenv("RSS_FEED_URLS", "").split("|")
SENT_POSTS_FILE = "sent_posts.json"

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

bot = Bot(token=BOT_TOKEN)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

sent_posts = []

MAX_HISTORY_HOURS = 24
DUPLICATE_THRESHOLD = 0.60  


def extract_media(entry):
    photos = []
    video = None
    if 'summary' in entry:
        soup = BeautifulSoup(entry.summary, 'html.parser')
        for img in soup.find_all('img'):
            url = img.get('src')
            if url:
                photos.append(url)
    if 'enclosures' in entry:
        for enclosure in entry.enclosures:
            if enclosure.get('type', '').startswith('video'):
                video = enclosure.get('url')
    return photos, video


def clean_html(html):
    return BeautifulSoup(html, 'html.parser').get_text()


def is_duplicate(text):
    now = datetime.utcnow()
    text_emb = model.encode(text, convert_to_tensor=True)

    for record in sent_posts:
        if (now - record['timestamp']) > timedelta(hours=MAX_HISTORY_HOURS):
            continue
        record_emb = model.encode(record['text'], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_emb, record_emb).item()
        logging.info(f"🔍 Сравнение:\n→ Новый: {text[:100]}...\n→ Старый: {record['text'][:100]}...\n→ Сходство: {similarity:.4f}")
        if similarity > DUPLICATE_THRESHOLD:
            return True
    return False



def save_post_record(text):
    now = datetime.utcnow()
    record = {'text': text, 'timestamp': now}
    sent_posts.append(record)

    # Удаляем устаревшие записи
    cutoff = now - timedelta(hours=MAX_HISTORY_HOURS)
    fresh_posts = [p for p in sent_posts if p['timestamp'] > cutoff]
    sent_posts.clear()
    sent_posts.extend(fresh_posts)

    try:
        with open(SENT_POSTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(
                [{"text": p["text"], "timestamp": p["timestamp"].isoformat()} for p in sent_posts],
                f,
                ensure_ascii=False,
                indent=2
            )
    except Exception as e:
        logging.error(f"❌ Ошибка сохранения истории: {e}")


async def rewrite_with_gemini(title, text):
    prompt = f"""Перепиши следующий новостной текст, сделай его уникальным, простым к восприятию, коротким, информативным и затягивающим, разделяй его на абзацы, если текст большой. Сначала придумай новый заголовок (короткий, простой, цепляющий, без * (звездочек)), затем сам текст (сжатый, информативный и читабельный, короткий, с небольшим количеством смайликов, построение текста должно быть простым, текст должен выглядеть просто и сжато)...\n\nЗаголовок: {title}\nТекст: {text}\n\nФормат ответа:\nЗаголовок: <новый заголовок>\nТекст: <уникализированный текст>"""
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    json_data = {"contents": [{"parts": [{"text": prompt}]}]}
    async with aiohttp.ClientSession() as session:
        async with session.post(GEMINI_API_URL, headers=headers, json=json_data) as response:
            if response.status == 200:
                result = await response.json()
                try:
                    text_block = result['candidates'][0]['content']['parts'][0]['text']
                    match = re.search(r"Заголовок:\s*(.+?)\s*Текст:\s*(.+)", text_block, re.DOTALL)
                    if match:
                        return match.group(1).strip(), match.group(2).strip()
                except Exception as e:
                    logging.error(f"Ошибка разбора Gemini: {e}")
            else:
                logging.error(f"Gemini API статус: {response.status}")
    return title, text


async def send_post(entry):
    title = entry.get('title', '')
    raw_text = clean_html(entry.summary) if 'summary' in entry else ''
    uniq_title, uniq_text = await rewrite_with_gemini(title, raw_text)
    full_text = f"<b>{uniq_title}</b>\n\n{uniq_text}"

    if is_duplicate(uniq_text):
        logging.info("🚫 Дубликат найден, пропускаем.")
        return False

    photos, video = extract_media(entry)
    try:
        if video:
            await bot.send_video(CHANNEL_ID, video=video, caption=full_text[:1024], parse_mode='HTML')
        elif photos:
            media_group = [InputMediaPhoto(media=url) for url in photos[:10]]
            media_group[0].caption = full_text[:1024]
            media_group[0].parse_mode = 'HTML'
            await bot.send_media_group(chat_id=CHANNEL_ID, media=media_group)
        else:
            await bot.send_message(chat_id=CHANNEL_ID, text=full_text[:4096], parse_mode='HTML')

        save_post_record(uniq_text)
        logging.info(f"✅ Отправлено: {uniq_title}")
        return True

    except Exception as e:
        logging.error(f"❌ Ошибка при отправке: {e}")
        return False

def load_sent_posts():
    global sent_posts
    if os.path.exists(SENT_POSTS_FILE):
        with open(SENT_POSTS_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                sent_posts = [
                    {"text": p["text"], "timestamp": datetime.fromisoformat(p["timestamp"])}
                    for p in data
                ]
                logging.info(f"✅ Загружено {len(sent_posts)} старых постов из истории.")
            except Exception as e:
                logging.error(f"❌ Ошибка загрузки истории постов: {e}")
                sent_posts = []
    else:
        logging.info("📁 Файл истории не найден, начинаем с пустой базы.")


async def main_loop():
    while True:
        logging.info(f"🔁 Новый цикл. RSS-источников: {len(RSS_FEED_URLS)}")

        all_entries = []
        for url in RSS_FEED_URLS:
            logging.info(f"📡 Загружаю RSS: {url}")
            feed = feedparser.parse(url)
            entries = sorted(feed.entries, key=lambda e: e.get('published_parsed', None) or datetime.utcnow())
            all_entries.append(entries)

        max_len = max(len(feed) for feed in all_entries)
        sent_in_this_round = 0

        for i in range(max_len):
            for feed_idx, feed in enumerate(all_entries):
                if i < len(feed):
                    entry = feed[i]
                    title = entry.get('title', '').strip()
                    logging.info(f"📰 [{feed_idx + 1}] Пост: {title}")
                    success = await send_post(entry)

                    if success:
                        sent_in_this_round += 1
                        if sent_in_this_round >= 3:
                            logging.info(f"⏸ Лимит 3 поста достигнут. Пауза {DELAY_SECONDS} секунд.")
                            await asyncio.sleep(DELAY_SECONDS)
                            break

                    await asyncio.sleep(30)

            else:
                continue
            break


if __name__ == '__main__':
    load_sent_posts()
    asyncio.run(main_loop())
