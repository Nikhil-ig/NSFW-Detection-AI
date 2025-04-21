import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters,
    CallbackContext
)
from detection_engine import DetectionEngine
from user_manager import UserManager
from group_config import GroupConfigManager

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Init core managers
detector = DetectionEngine()
user_manager = UserManager()
group_config = GroupConfigManager()

async def start(update: Update, context: CallbackContext):
    """Start command handler"""
    await update.message.reply_text("👋 Hello! I'm an NSFW detection bot. I scan images, GIFs, and videos!")

async def toggle_group(update: Update, context: CallbackContext):
    """Command to toggle NSFW detection on/off in a group"""
    if update.effective_chat.type not in ["group", "supergroup"]:
        return await update.message.reply_text("⚠️ This command only works in groups.")
    
    group_id = update.effective_chat.id
    is_admin = await is_user_admin(update, context)
    if not is_admin:
        return await update.message.reply_text("🚫 Only admins can use this.")
    
    state = group_config.toggle_enabled(group_id)
    await update.message.reply_text(f"✅ NSFW bot is now {'enabled' if state else 'disabled'} in this group.")

async def is_user_admin(update, context):
    """Check if the user is an admin"""
    user_id = update.effective_user.id
    chat = update.effective_chat
    member = await context.bot.get_chat_member(chat.id, user_id)
    return member.status in ("administrator", "creator")

async def handle_media(update: Update, context: CallbackContext):
    """Handle incoming media (stickers, images, GIFs, videos)"""
    if update.effective_chat.type not in ["group", "supergroup"]:
        return

    group_id = update.effective_chat.id
    if not group_config.is_enabled(group_id):
        return

    file_id = None
    media_type = None

    # Determine the media type and file_id
    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        media_type = 'photo'
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        file_id = update.message.document.file_id
        media_type = 'document'
    elif update.message.animation:
        file_id = update.message.animation.file_id
        media_type = 'gif'
    elif update.message.video:
        file_id = update.message.video.file_id
        media_type = 'video'
    elif update.message.sticker and update.message.sticker.is_animated is False:
        file_id = update.message.sticker.file_id
        media_type = 'sticker'

    if not file_id:
        return

    # Download the file
    new_file = await context.bot.get_file(file_id)
    file_path = f"downloads/{file_id}"
    await new_file.download_to_drive(file_path)

    # Run the NSFW detection
    result = detector.analyze(file_path)
    os.remove(file_path)

    # If NSFW content is detected
    if result["is_nsfw"]:
        try:
            await update.message.delete()
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❌ NSFW {media_type} detected and deleted. [{result['reason']}]"
            )
        except Exception as e:
            logging.error(f"Failed to delete message: {e}")

        # Increment the user's warning
        user_id = update.effective_user.id
        warnings = user_manager.increment_warning(group_id, user_id)

        # Auto-ban the user if the group is configured for it
        if group_config.auto_ban_enabled(group_id) and warnings >= 3:
            try:
                await context.bot.ban_chat_member(group_id, user_id)
                await context.bot.send_message(
                    chat_id=group_id,
                    text=f"🚫 User banned after 3 violations."
                )
            except Exception as e:
                logging.error(f"Failed to ban user: {e}")

def main():
    """Main entry point for the bot"""
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN missing in .env")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Command Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("toggle_nsfw", toggle_group))

    # Media Handler (Handles all types of media)
    app.add_handler(MessageHandler(filters.ALL, handle_media))

    logging.info("🤖 NSFW Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
