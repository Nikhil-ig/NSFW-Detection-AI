import logging
from telegram.ext import ContextTypes
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MessageManager:
    @staticmethod
    async def delete_messages(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_ids: List[int]) -> None:
        """Delete multiple messages with error handling"""
        for msg_id in message_ids:
            await MessageManager.safe_delete_message(context, chat_id, msg_id)

    @staticmethod
    async def safe_delete_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int) -> None:
        """Safely delete a single message"""
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception as e:
            if "message to delete not found" not in str(e):
                logger.warning(f"Failed to delete message {message_id}: {e}")

    @staticmethod
    def track_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int) -> None:
        """Track a message for future cleanup"""
        if 'message_tracker' not in context.chat_data:
            context.chat_data['message_tracker'] = {}
        
        if chat_id not in context.chat_data['message_tracker']:
            context.chat_data['message_tracker'][chat_id] = []
            
        context.chat_data['message_tracker'][chat_id].append(message_id)

    @staticmethod
    async def cleanup_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
        """Cleanup all tracked messages in a chat"""
        if 'message_tracker' in context.chat_data and chat_id in context.chat_data['message_tracker']:
            await MessageManager.delete_messages(
                context,
                chat_id,
                context.chat_data['message_tracker'][chat_id]
            )
            context.chat_data['message_tracker'][chat_id] = []

    @staticmethod
    def schedule_cleanup(context: ContextTypes.DEFAULT_TYPE, chat_id: int, delay: int = 120) -> None:
        """Schedule automatic cleanup"""
        context.job_queue.run_once(
            callback=lambda ctx: MessageManager.cleanup_chat(ctx, chat_id),
            when=delay,
            chat_id=chat_id,
            name=f"cleanup_{chat_id}"
        )