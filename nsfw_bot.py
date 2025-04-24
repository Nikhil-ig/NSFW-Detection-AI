# import os
# import logging
# from dotenv import load_dotenv
# from telegram import Update
# from telegram.ext import (
#     ApplicationBuilder, CommandHandler, MessageHandler, filters,
#     CallbackContext
# )
# from detection_engine import DetectionEngine
# from user_manager import UserManager
# from group_config import GroupConfigManager

# load_dotenv()
# BOT_TOKEN = os.getenv("BOT_TOKEN")

# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )

# # Init core managers
# detector = DetectionEngine()
# user_manager = UserManager()
# group_config = GroupConfigManager()

# async def start(update: Update, context: CallbackContext):
#     """Start command handler"""
#     await update.message.reply_text("👋 Hello! I'm an NSFW detection bot. I scan images, GIFs, and videos!")

# async def toggle_group(update: Update, context: CallbackContext):
#     """Command to toggle NSFW detection on/off in a group"""
#     if update.effective_chat.type not in ["group", "supergroup"]:
#         return await update.message.reply_text("⚠️ This command only works in groups.")
    
#     group_id = update.effective_chat.id
#     is_admin = await is_user_admin(update, context)
#     if not is_admin:
#         return await update.message.reply_text("🚫 Only admins can use this.")
    
#     state = group_config.toggle_enabled(group_id)
#     await update.message.reply_text(f"✅ NSFW bot is now {'enabled' if state else 'disabled'} in this group.")

# async def is_user_admin(update, context):
#     """Check if the user is an admin"""
#     user_id = update.effective_user.id
#     chat = update.effective_chat
#     member = await context.bot.get_chat_member(chat.id, user_id)
#     return member.status in ("administrator", "creator")

# async def handle_media(update: Update, context: CallbackContext):
#     """Handle incoming media (stickers, images, GIFs, videos)"""
#     if update.effective_chat.type not in ["group", "supergroup"]:
#         return

#     group_id = update.effective_chat.id
#     if not group_config.is_enabled(group_id):
#         return

#     file_id = None
#     media_type = None

#     # Determine the media type and file_id
#     if update.message.photo:
#         file_id = update.message.photo[-1].file_id
#         media_type = 'photo'
#     elif update.message.document and update.message.document.mime_type.startswith("image/"):
#         file_id = update.message.document.file_id
#         media_type = 'document'
#     elif update.message.animation:
#         file_id = update.message.animation.file_id
#         media_type = 'gif'
#     elif update.message.video:
#         file_id = update.message.video.file_id
#         media_type = 'video'
#     elif update.message.sticker and update.message.sticker.is_animated is False:
#         file_id = update.message.sticker.file_id
#         media_type = 'sticker'

#     if not file_id:
#         return

#     # Download the file
#     new_file = await context.bot.get_file(file_id)
#     file_path = f"downloads/{file_id}"
#     await new_file.download_to_drive(file_path)

#     # Run the NSFW detection
#     result = detector.analyze(file_path)
#     os.remove(file_path)

#     # If NSFW content is detected
#     if result["is_nsfw"]:
#         try:
#             await update.message.delete()
#             await context.bot.send_message(
#                 chat_id=update.effective_chat.id,
#                 text=f"❌ NSFW {media_type} detected and deleted. [{result['reason']}]"
#             )
#         except Exception as e:
#             logging.error(f"Failed to delete message: {e}")

#         # Increment the user's warning
#         user_id = update.effective_user.id
#         warnings = user_manager.increment_warning(group_id, user_id)

#         # Auto-ban the user if the group is configured for it
#         if group_config.auto_ban_enabled(group_id) and warnings >= 3:
#             try:
#                 await context.bot.ban_chat_member(group_id, user_id)
#                 await context.bot.send_message(
#                     chat_id=group_id,
#                     text=f"🚫 User banned after 3 violations."
#                 )
#             except Exception as e:
#                 logging.error(f"Failed to ban user: {e}")

# def main():
#     """Main entry point for the bot"""
#     if not BOT_TOKEN:
#         raise RuntimeError("BOT_TOKEN missing in .env")

#     app = ApplicationBuilder().token(BOT_TOKEN).build()

#     # Command Handlers
#     app.add_handler(CommandHandler("start", start))
#     app.add_handler(CommandHandler("toggle_nsfw", toggle_group))

#     # Media Handler (Handles all types of media)
#     app.add_handler(MessageHandler(filters.ALL, handle_media))

#     logging.info("🤖 NSFW Bot is running...")
#     app.run_polling()

# if __name__ == "__main__":
#     main()



import os
import logging
import uuid
import json
import cv2
import numpy as np
from PIL import Image, ImageSequence
from nudenet import NudeDetector
import tensorflow as tf
from tensorflow.keras.models import load_model
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
    CommandHandler,
    CallbackQueryHandler
)
import asyncio
from typing import Dict, List, Union

# Replit-specific modifications
from flask import Flask, Response
import threading

# Initialize Flask server for uptime monitoring
server = Flask(__name__)

@server.route('/')
def home():
    return Response("🤖 NSFW Detection Bot is running", status=200)

def run_flask():
    server.run(host='0.0.0.0', port=8080)


MODEL_SELECTION = "both"  # Using both models for maximum accuracy
# Enhanced logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("nsfw_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DOWNLOAD_DIR = "downloads"
CONFIG_FILE = "groups_configs.json"
# BLOCKLIST_FILE = "group_blocklists.json"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "NSFW_THRESHOLD": 0.45,
    "SAFE_THRESHOLD": 0.25,
    "FRAME_ANALYSIS_COUNT": 3,
    "MIN_DETECTION_CONFIDENCE": 0.25,
    "IGNORE_ADMINS": True,
    "MODEL_SELECTION": "both"
}

# Model configuration
TF_MODEL_PATH = 'nsfw.299x299.h5'
TF_INPUT_SIZE = (299, 299)

# NSFW classes with strict filtering
STRICT_NSFW_CLASSES = {
    # Explicit Nudity
    'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED', 'ANUS_EXPOSED',
    'PUBIC_HAIR_EXPOSED', 'AREOLA_EXPOSED', 'NIPPLE_EXPOSED', 'BUTTOCKS_EXPOSED',
    'FEMALE_BREAST_EXPOSED', 'UNDERWEAR_VISIBLE_EXPLICIT', 'CLOTHING_SEE_THROUGH',
    'WET_CLOTHING_EXPLICIT', 'EXPOSED_CHEST_MALE', 'EXPOSED_CHEST_FEMALE',
    'EXPOSED_THIGHS_EXPLICIT', 'EXPOSED_BACK_EXPLICIT', 'EXPOSED_STOMACH_EXPLICIT',
    'EXPOSED_LEGS_EXPLICIT', 'NAKED_BODY_EXPLICIT', 'EXPOSED_ARMPIT_EXPLICIT',
    'EXPOSED_FEET_FETISH', 'CLOTHING_RIPPED_EXPLICIT', 'TOPLESS_EXPLICIT',
    'BOTTOMLESS_EXPLICIT', 'EXPOSED_BUTTOCKS_CLEAVAGE', 'EXPOSED_BREAST_CLEAVAGE',
    'EXPOSED_UPPER_BODY_EXPLICIT', 'EXPOSED_LOWER_BODY_EXPLICIT', 'PARTIAL_NUDITY_EXPLICIT',

    # Sexual Activities
    'SEXUAL_INTERCOURSE', 'ORAL_SEX_ACTIVE', 'ANAL_SEX_ACTIVE', 'MASTURBATION_EXPLICIT',
    'SEX_TOYS_VISIBLE', 'PORNOGRAPHIC_POSES', 'GROUP_SEX_ACTIVITY', 'BDSM_EQUIPMENT',
    'FETISH_ACTIVITY_EXPLICIT', 'DOMINATION_ACT', 'SUBMISSION_ACT', 'BONDAGE_ACT',
    'HUMAN_TRAFFICKING_INDICATORS', 'CHILD_EXPLOITATION', 'REVENGE_PORN_INDICATORS',
    'UPSKIRT_SHOTS', 'DOWNBLOUSE_SHOTS', 'NON_CONSENSUAL_ACT', 'EXHIBITIONISM_ACT',
    'VOYEURISM_ACT', 'SEXUAL_HARASSMENT_ACT', 'CYBERSEX_ACTIVITY', 'SEXTING_INDICATORS',
    'SEXUAL_SIMULATION_ACT', 'GRINDING_EXPLICIT', 'FROTTAGE_ACT', 'INCEST_THEME',
    'BESTIALITY_INDICATORS', 'NECROPHILIA_INDICATORS', 'SCAT_FETISH_ACT', 'WATERSPORTS_ACT',
    'SADOMASOCHISM_ACT',

    # Exploitation & Illegal
    'MINOR_IN_EXPLICIT_CONTENT', 'CHILD_ABUSE_MATERIAL', 'SEXUALIZED_MINORS',
    'TEEN_EXPLOITATION', 'REVENGE_PORN_CONTENT', 'BLACKMAIL_CONTENT',
    'HUMAN_TRAFFICKING_SIGNS', 'DRUG_FACILITATED_ASSAULT', 'DATE_RAPE_INDICATORS',
    'ALCOHOL_INTOXICATION_EXPLOITATION', 'UNCONSCIOUS_PERSON_EXPLOITATION',
    'SLEEPING_PERSON_EXPLOITATION', 'DRUGGED_PERSON_EXPLOITATION', 'NON_CONSENSUAL_SHARING',
    'DEEPFAKE_PORN', 'AI_GENERATED_EXPLICIT', 'MORPHED_EXPLICIT_CONTENT', 'PRIVATE_PARTS_ZOOM',

    # Extreme Content
    'VIOLENT_SEX_ACT', 'BLOOD_IN_SEXUAL_CONTEXT', 'GORE_IN_SEXUAL_CONTEXT',
    'WEAPONS_IN_SEXUAL_CONTEXT', 'HATE_SYMBOLS_EXPLICIT', 'RACIST_SEXUAL_CONTENT',
    'HOMOPHOBIC_SEXUAL_CONTENT', 'TRANSPHOBIC_SEXUAL_CONTENT', 'SELF_HARM_EXPLICIT',
    'SUICIDE_IN_SEXUAL_CONTEXT', 'ANIMAL_CRUELTY_SEXUAL', 'DRUG_USE_EXPLICIT',
    'HARD_DRUG_USE_SEXUAL', 'INCEST_EXPLICIT', 'RAPE_DEPICTION', 'ABUSE_DEPICTION',
    'TORTURE_IN_SEXUAL_CONTEXT', 'SNUFF_FILM_INDICATORS', 'EXTREME_FETISH_ACT',
    'DANGEROUS_ACTS_SEXUAL', 'CHOKING_ACT_EXPLICIT', 'AUTOEROTIC_ASPHYXIATION',

    # Fetish/BDSM
    'BDSM_HARDCORE', 'SHIBARI_EXPLICIT', 'GOLDEN_SHOWER_ACT', 'SCATOLOGY_ACT',
    'NECROPHILIA_ACT', 'BESTIALITY_ACT', 'FISTING_ACT', 'FEMDOM_EXTREME', 'CBT_EXTREME',
    'FOOT_FETISH_EXPLICIT', 'UNDERAGE_FETISH', 'INCEST_FETISH', 'RAPE_FETISH',
    'FORCED_ACT_FETISH', 'HYDROPHILIA_EXTREME'
}

# Safe content indicators
SAFE_CONTENT_INDICATORS = {
    # Activities & Settings
    'FAMILY_GATHERING', 'CHILDREN_PLAYING', 'BABY_CARE', 'SCHOOL_ACTIVITY',
    'DANCE_PERFORMANCE', 'SPORTS_EVENT', 'GYM_WORKOUT', 'YOGA_PRACTICE',
    'SWIMMING_COMPETITION', 'BEACH_ACTIVITIES', 'MEDICAL_EXAM', 'BREASTFEEDING',
    'CHILDBIRTH_EDUCATION', 'ART_MODEL_SESSION', 'NATURE_PHOTOGRAPHY',
    'WILDLIFE_DOCUMENTARY', 'CULTURAL_EVENT', 'TRADITIONAL_CLOTHING',
    'RELIGIOUS_CEREMONY', 'HISTORICAL_ART', 'ANATOMY_EDUCATION', 'SEX_EDUCATION',
    'MEDICAL_ILLUSTRATION', 'FASHION_SHOW', 'THEATRE_PERFORMANCE', 'CIRCUS_ACT',
    'ACROBATICS', 'MARTIAL_ARTS', 'WRESTLING_MATCH', 'BALLET_PRACTICE',
    'CHEERLEADING', 'AEROBICS_CLASS', 'PHYSICAL_THERAPY', 'SURGICAL_PROCEDURE',
    'FIREFIGHTER_TRAINING',

    # Clothing & Appearance
    'FULLY_COVERED_BODY', 'PROFESSIONAL_UNIFORM', 'SPORTS_UNIFORM',
    'SWIMSUIT_NON_SEXUAL', 'UNDERWEAR_NON_EXPLICIT', 'CULTURAL_ATTIRE',
    'HISTORICAL_COSTUME', 'PROTECTIVE_GEAR', 'MEDICAL_SCRUBS', 'MILITARY_UNIFORM',
    'RELIGIOUS_GARB', 'TRADITIONAL_DRESS', 'ARTISTIC_NUDITY', 'BODY_PAINT_ART',
    'SCARIFICATION_ART', 'TATTOO_ART', 'PIERCING_ART', 'HAIRSTYLING_DEMO',
    'MAKEUP_TUTORIAL', 'FASHION_DESIGN', 'COSTUME_DESIGN', 'THEATRICAL_MAKEUP',
    'AGE_APPROPRIATE_CLOTHING', 'MODEST_DRESS', 'PROFESSIONAL_ATTIRE',

    # Nature & Animals
    'ANIMAL_CARE', 'VETERINARY_PROCEDURE', 'WILDLIFE_CONSERVATION', 'MARINE_LIFE',
    'BIRDWATCHING', 'SAFARI_TOUR', 'PET_GROOMING', 'ANIMAL_TRAINING',
    'FARM_ACTIVITIES', 'HORSE_RIDING', 'DOG_SHOW', 'CAT_SHOW', 'AQUARIUM_VISIT',
    'ZOO_ENVIRONMENT', 'NATIONAL_PARK', 'BOTANICAL_GARDEN', 'HIKING_ACTIVITY',
    'CAMPING_SCENE', 'BEACH_CLEANUP', 'ECOLOGY_STUDY',

    # Medical & Educational
    'MEDICAL_TRAINING', 'NURSING_PRACTICE', 'FIRST_AID_DEMO', 'ANATOMY_CLASS',
    'BIOLOGY_LAB', 'PSYCHOLOGY_STUDY', 'SEX_ED_CLASS', 'BIRTH_CONTROL_ED',
    'GYNECOLOGY_ED', 'URBAN_HEALTH_ED', 'NUTRITION_CLASS', 'FITNESS_EDUCATION',
    'SURGICAL_TRAINING', 'PHYSIOTHERAPY', 'RADIOLOGY_IMAGING', 'MEDICAL_ULTRASOUND',
    'DERMATOLOGY_STUDY', 'BURN_VICTIM_CARE',

    # Cultural & Historical
    'MUSEUM_EXHIBIT', 'ARCHAEOLOGICAL_FIND', 'HISTORICAL_ARTIFACT', 'CAVE_PAINTINGS',
    'TEMPLE_ART', 'RELIGIOUS_ART', 'FOLK_DANCE', 'TRADITIONAL_CEREMONY',
    'CULTURAL_FESTIVAL', 'HISTORICAL_REENACTMENT', 'ANCIENT_SCULPTURE',
    'ETHNOGRAPHIC_STUDY', 'INDIGENOUS_PRACTICES', 'TRIBAL_ART', 'ANCESTRAL_WISDOM'
}

# Initialize models
logger.info("⚙️ Loading advanced detection models...")
nude_detector = NudeDetector()

# Load TensorFlow model
tf_model = None
try:
    tf_model = load_model(
        TF_MODEL_PATH,
        custom_objects={'KerasLayer': tf.keras.layers.Layer},
        compile=False
    )
    logger.info(f"✅ TensorFlow model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading TensorFlow model: {str(e)}")
    if "CUDA" in str(e):
        logger.warning("⚠️ Trying to load with CPU only...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        try:
            tf_model = load_model(TF_MODEL_PATH, compile=False)
        except Exception as e:
            logger.error(f"❌ CPU load failed: {str(e)}")

# Configuration management
def load_all_configs() -> List[Dict]:
    """Load all group configurations from file"""
    try:
        with open('group_configs.json', 'r') as f:
            configs = json.load(f)
            # Ensure we have a list of dictionaries
            if isinstance(configs, dict):
                # Handle case where file contains a single group config as object
                return [configs]
            return configs
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def load_group_config(chat_id: Union[str, int]) -> Dict:
    """Load configuration for a specific group"""
    chat_id = str(chat_id)
    configs = load_all_configs()
    
    # Find existing config
    for config in configs:
        if str(config.get('chat_id')) == chat_id:
            # Ensure all required keys exist
            return {
                "chat_id": chat_id,
                "NSFW_THRESHOLD": config.get("NSFW_THRESHOLD", 0.85),
                "SAFE_THRESHOLD": config.get("SAFE_THRESHOLD", 0.25),
                "FRAME_ANALYSIS_COUNT": config.get("FRAME_ANALYSIS_COUNT", 3),
                "MIN_DETECTION_CONFIDENCE": config.get("MIN_DETECTION_CONFIDENCE", 0.25),
                "IGNORE_ADMINS": config.get("IGNORE_ADMINS", False),
                "MODEL_SELECTION": config.get("MODEL_SELECTION", "both")
            }
    
    # Create new default config if not found
    default_config = {
        "chat_id": chat_id,
        "NSFW_THRESHOLD": 0.85,
        "SAFE_THRESHOLD": 0.25,
        "FRAME_ANALYSIS_COUNT": 3,
        "MIN_DETECTION_CONFIDENCE": 0.25,
        "IGNORE_ADMINS": False,
        "MODEL_SELECTION": "both"
    }
    
    configs.append(default_config)
    save_all_configs(configs)
    return default_config

def save_all_configs(configs: list):
    with open('group_configs.json', 'w') as f:
        json.dump(configs, f, indent=4)

def save_group_config(chat_id: Union[str, int], new_config: Dict) -> None:
    """Save configuration for a specific group"""
    chat_id = str(chat_id)
    configs = load_all_configs()
    
    # Update existing config or add new one
    updated = False
    for i, config in enumerate(configs):
        if str(config.get('chat_id')) == chat_id:
            configs[i] = new_config
            updated = True
            break
    
    if not updated:
        configs.append(new_config)
    
    save_all_configs(configs)

def load_group_blocklist(chat_id: int):
    try:
        with open(BLOCKLIST_FILE) as f:
            blocklists = json.load(f)
            return blocklists.get(str(chat_id), {"stickers": [], "packs": [], "gifs": []})
    except:
        return {"stickers": [], "packs": [], "gifs": []}

def save_group_blocklist(chat_id: int, blocklist: dict):
    try:
        with open(BLOCKLIST_FILE) as f:
            blocklists = json.load(f)
    except:
        blocklists = {}
    blocklists[str(chat_id)] = blocklist
    with open(BLOCKLIST_FILE, 'w') as f:
        json.dump(blocklists, f)

# Admin verification
async def is_group_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat = update.effective_chat
    if chat.type == 'private':
        return False
    try:
        admins = await context.bot.get_chat_administrators(chat.id)
        return any(admin.user.id == user.id for admin in admins)
    except:
        return False

# Image processing
def preprocess_image_for_tf(image_path: str, target_size=TF_INPUT_SIZE) -> np.ndarray:
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=target_size,
            color_mode='rgb',
            interpolation='bilinear'
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array / 255.0
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        return np.zeros((1, *target_size, 3), dtype=np.float32)

# Detection functions
def detect_nsfw_tensorflow(image_path: str, chat_id: int) -> tuple[bool, float]:
    if tf_model is None:
        return False, 0.0
    
    config = load_group_config(chat_id)
    try:
        img_array = preprocess_image_for_tf(image_path)
        predictions = tf_model.predict(img_array)
        
        if predictions.shape[-1] == 5:  # Common NSFW model format
            nsfw_score = predictions[0][1] + predictions[0][3] * 0.8
        elif predictions.shape[-1] == 2:  # Binary classification
            nsfw_score = predictions[0][1]
        else:
            nsfw_score = predictions[0][0]
            
        confidence = float(nsfw_score)
        return confidence > config['NSFW_THRESHOLD'], confidence
    except Exception as e:
        logger.error(f"🔞 TensorFlow detection failed: {str(e)}")
        return False, 0.0

def detect_nsfw_nudenet(image_path: str, chat_id: int) -> tuple[bool, float]:
    config = load_group_config(chat_id)
    try:
        detections = nude_detector.detect(image_path)
        valid_detections = [
            det for det in detections 
            if det['class'] in STRICT_NSFW_CLASSES 
            and det['score'] >= config['MIN_DETECTION_CONFIDENCE']
        ]
        
        if not valid_detections:
            return False, 0.0
            
        max_detection = max(valid_detections, key=lambda x: x['score'])
        confidence = max_detection['score']
        
        if max_detection['class'] in {'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED'}:
            if confidence < 0.65:
                return False, confidence * 0.5
                
        return confidence > config['NSFW_THRESHOLD'], confidence
    except Exception as e:
        logger.error(f"🔞 NudeNet detection failed: {str(e)}")
        return False, 0.0

def detect_safe_content(image_path: str) -> bool:
    try:
        detections = nude_detector.detect(image_path)
        safe_detections = [
            det for det in detections 
            if det['class'] in SAFE_CONTENT_INDICATORS 
            and det['score'] >= 0.4
        ]
        return len(safe_detections) > 0
    except Exception as e:
        logger.error(f"🌿 Safe content detection failed: {str(e)}")
        return False

def detect_nsfw_combined(image_path: str, chat_id: int) -> tuple[bool, float]:
    config = load_group_config(chat_id)
    try:
        if detect_safe_content(image_path):
            return False, 0.0
            
        n_result, n_conf = detect_nsfw_nudenet(image_path, chat_id)
        t_result, t_conf = detect_nsfw_tensorflow(image_path, chat_id)
        
        if n_conf < config['SAFE_THRESHOLD'] and t_conf < config['SAFE_THRESHOLD']:
            return False, max(n_conf, t_conf)
            
        if max(n_conf, t_conf) > 0.9:
            confidence = max(n_conf, t_conf)
        else:
            confidence = (n_conf * 0.6 + t_conf * 0.4)
        
        if n_result and t_result:
            confidence = min(1.0, confidence * 1.1)
            
        logger.info(f"🔍 Combined confidence: {confidence:.2f}")
        return confidence > config['NSFW_THRESHOLD'], confidence
    except Exception as e:
        logger.error(f"🔞 Combined detection failed: {str(e)}")
        return False, 0.0

def detect_nsfw(image_path: str, chat_id: int) -> tuple[bool, float]:
    config = load_group_config(chat_id)
    if config['MODEL_SELECTION'] == "nudenet":
        return detect_nsfw_nudenet(image_path, chat_id)
    elif config['MODEL_SELECTION'] == "tensorflow":
        return detect_nsfw_tensorflow(image_path, chat_id)
    else:
        return detect_nsfw_combined(image_path, chat_id)

def analyze_frames(file_path: str, chat_id: int) -> float:
    config = load_group_config(chat_id)
    max_confidence = 0.0
    frame_count = 0
    
    try:
        if file_path.endswith(('.webm', '.mp4')):
            vid = cv2.VideoCapture(file_path)
            total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = max(1, total_frames // config['FRAME_ANALYSIS_COUNT'])
            
            for i in range(config['FRAME_ANALYSIS_COUNT']):
                vid.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
                success, frame = vid.read()
                if success:
                    frame_path = f"{file_path}_frame_{i}.jpg"
                    cv2.imwrite(frame_path, frame)
                    
                    if detect_safe_content(frame_path):
                        os.remove(frame_path)
                        continue
                        
                    _, confidence = detect_nsfw(frame_path, chat_id)
                    max_confidence = max(max_confidence, confidence)
                    os.remove(frame_path)
                    frame_count += 1
                    
                    if max_confidence > 0.95:
                        break
            vid.release()
            
        elif file_path.endswith('.gif'):
            with Image.open(file_path) as img:
                total_frames = img.n_frames
                frame_step = max(1, total_frames // config['FRAME_ANALYSIS_COUNT'])
                
                for i in range(config['FRAME_ANALYSIS_COUNT']):
                    img.seek(i * frame_step)
                    frame_path = f"{file_path}_frame_{i}.jpg"
                    img.convert('RGB').save(frame_path)
                    
                    if detect_safe_content(frame_path):
                        os.remove(frame_path)
                        continue
                        
                    _, confidence = detect_nsfw(frame_path, chat_id)
                    max_confidence = max(max_confidence, confidence)
                    os.remove(frame_path)
                    frame_count += 1
                    
                    if max_confidence > 0.95:
                        break
                    
        if frame_count > 0 and max_confidence < config['NSFW_THRESHOLD']:
            safe_frame_ratio = (config['FRAME_ANALYSIS_COUNT'] - frame_count) / config['FRAME_ANALYSIS_COUNT']
            if safe_frame_ratio > 0.7:
                max_confidence *= 0.7
                
        return max_confidence
    except Exception as e:
        logger.error(f"🎞 Frame analysis failed: {str(e)}")
        return 0.0

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 NSFW Detection Bot is running!\n"
        "I automatically analyze photos, videos, GIFs and stickers for explicit content.\n\n"
        "Commands:\n"
        "/settings - Configure group settings (admins only)\n"
        "/help - Show help information"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ NSFW Detection Bot Help\n\n"
        "I analyze these media types:\n"
        "- Photos (sent as photo or document)\n"
        "- Videos (up to 20MB)\n"
        "- GIFs\n"
        "- Static and animated stickers\n\n"
        "Admin commands:\n"
        "/settings - Configure detection parameters\n"
        "\n"
        "This bot automatically detects and removes NSFW content based on group settings."
    )

async def group_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_group_admin(update, context):
        return
        
    chat_id = update.effective_chat.id
    config = load_group_config(chat_id)
    
    keyboard = [
        [InlineKeyboardButton(f"🔞 NSFW Threshold ({config['NSFW_THRESHOLD']})", callback_data='set_nsfw')],
        [InlineKeyboardButton(f"✅ Safe Threshold ({config['SAFE_THRESHOLD']})", callback_data='set_safe')],
        [InlineKeyboardButton(f"🎞 Frames Analyzed ({config['FRAME_ANALYSIS_COUNT']})", callback_data='set_frames')],
        [InlineKeyboardButton(f"🎯 Min Confidence ({config['MIN_DETECTION_CONFIDENCE']})", callback_data='set_confidence')],
        [InlineKeyboardButton(f"👑 Ignore Admins ({'ON' if config['IGNORE_ADMINS'] else 'OFF'})", callback_data='toggle_ignore')],
        [InlineKeyboardButton(f"🤖 Model ({config['MODEL_SELECTION']})", callback_data='set_model')],
        [InlineKeyboardButton("📦 Blocklist Management", callback_data='blocklist_menu')],
        [InlineKeyboardButton("❌ Close Menu", callback_data='close_menu')]
    ]
    
    await update.message.reply_text(
        "⚙️ Group Settings Panel ⚙️\nChoose a setting to modify:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# Blocklist Commands
async def block_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the block command for stickers, packs, and GIFs"""
    if not await is_group_admin(update, context):
        await update.message.reply_text("❌ Only admins can manage blocklists")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("ℹ️ Please reply to a sticker or GIF with /block")
        return

    replied = update.message.reply_to_message
    chat_id = update.effective_chat.id
    blocklist = load_group_blocklist(chat_id)

    if replied.sticker:
        sticker = replied.sticker
        if sticker.set_name:
            # Block both sticker and its pack
            if sticker.file_unique_id not in blocklist['stickers']:
                blocklist['stickers'].append(sticker.file_unique_id)
            if sticker.set_name not in blocklist['packs']:
                blocklist['packs'].append(sticker.set_name)
            save_group_blocklist(chat_id, blocklist)
            await update.message.reply_text(
                f"✅ Blocked sticker and its pack:\n"
                f"Sticker ID: {sticker.file_unique_id[:8]}...\n"
                f"Pack: {sticker.set_name}"
            )
        else:
            # Block only the sticker
            if sticker.file_unique_id not in blocklist['stickers']:
                blocklist['stickers'].append(sticker.file_unique_id)
                save_group_blocklist(chat_id, blocklist)
                await update.message.reply_text(
                    f"✅ Blocked sticker:\n"
                    f"ID: {sticker.file_unique_id[:8]}..."
                )
            else:
                await update.message.reply_text("ℹ️ This sticker is already blocked")

    elif replied.animation:
        gif = replied.animation
        if gif.file_unique_id not in blocklist['gifs']:
            blocklist['gifs'].append(gif.file_unique_id)
            save_group_blocklist(chat_id, blocklist)
            await update.message.reply_text(
                f"✅ Blocked GIF:\n"
                f"ID: {gif.file_unique_id[:8]}..."
            )
        else:
            await update.message.reply_text("ℹ️ This GIF is already blocked")

    else:
        await update.message.reply_text("ℹ️ Please reply to a sticker or GIF to block it")


def load_group_blocklist(chat_id: int) -> dict:
    """Load blocklist for a specific group"""
    try:
        with open(BLOCKLIST_FILE, 'r') as f:
            blocklists = json.load(f)
            return blocklists.get(str(chat_id), {
                "stickers": [],
                "packs": [],
                "gifs": []
            })
    except (FileNotFoundError, json.JSONDecodeError):
        return {"stickers": [], "packs": [], "gifs": []}



def save_group_blocklist(chat_id: int, blocklist: dict):
    """Save blocklist for a specific group"""
    try:
        with open(BLOCKLIST_FILE, 'r') as f:
            blocklists = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        blocklists = {}

    blocklists[str(chat_id)] = blocklist

    with open(BLOCKLIST_FILE, 'w') as f:
        json.dump(blocklists, f, indent=4)


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat.id
    user_id = query.from_user.id
    
    if not await is_group_admin(update, context):
        await query.edit_message_text("❌ Only admins can modify settings")
        return

    config = load_group_config(chat_id)  # Always load fresh config
    
    if query.data == 'close_menu':
        await query.message.delete()
        return
        
    elif query.data == 'blocklist_menu':
        keyboard = [
            [InlineKeyboardButton("⛔ Block Sticker", callback_data='block_sticker'),
             InlineKeyboardButton("📦 Block Pack", callback_data='block_pack')],
            [InlineKeyboardButton("🎥 Block GIF", callback_data='block_gif'),
             InlineKeyboardButton("🔙 Back", callback_data='settings_main')]
        ]
        await query.edit_message_text(
            "🔒 Blocklist Management\nChoose an option:",
            reply_markup=InlineKeyboardMarkup(keyboard))
        return

    elif query.data == 'settings_main':
        config = load_group_config(chat_id)  # Reload config before showing menu
        keyboard = [
            [InlineKeyboardButton(f"🔞 NSFW Threshold ({config['NSFW_THRESHOLD']:.2f})", callback_data='set_NSFW_THRESHOLD')],
            [InlineKeyboardButton(f"✅ Safe Threshold ({config['SAFE_THRESHOLD']:.2f})", callback_data='set_SAFE_THRESHOLD')],
            [InlineKeyboardButton(f"🎞 Frames Analyzed ({config['FRAME_ANALYSIS_COUNT']})", callback_data='set_FRAME_ANALYSIS_COUNT')],
            [InlineKeyboardButton(f"🎯 Min Confidence ({config['MIN_DETECTION_CONFIDENCE']:.2f})", callback_data='set_MIN_DETECTION_CONFIDENCE')],
            [InlineKeyboardButton(f"👑 Ignore Admins ({'ON' if config['IGNORE_ADMINS'] else 'OFF'})", callback_data='toggle_ignore')],
            [InlineKeyboardButton(f"🤖 Model ({config['MODEL_SELECTION']})", callback_data='set_model')],
            [InlineKeyboardButton("📦 Blocklist Management", callback_data='blocklist_menu')],
            [InlineKeyboardButton("❌ Close Menu", callback_data='close_menu')]
        ]
        await query.edit_message_text(
            "⚙️ Group Settings Panel ⚙️\nChoose a setting to modify:",
            reply_markup=InlineKeyboardMarkup(keyboard))
        return

    elif query.data == 'toggle_ignore':
        config['IGNORE_ADMINS'] = not config['IGNORE_ADMINS']
        save_group_config(chat_id, config)
        # Return to settings menu to show updated value
        await query.edit_message_text(
            f"✅ Admin ignoring {'enabled' if config['IGNORE_ADMINS'] else 'disabled'}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Settings", callback_data='settings_main')]
            ]))
        return

    elif query.data == 'set_model':
        keyboard = [
            [InlineKeyboardButton("NudeNet", callback_data='model_nudenet'),
             InlineKeyboardButton("TensorFlow", callback_data='model_tensorflow')],
            [InlineKeyboardButton("Both", callback_data='model_both'),
             InlineKeyboardButton("🔙 Back", callback_data='settings_main')]
        ]
        await query.edit_message_text(
            "Select detection model:",
            reply_markup=InlineKeyboardMarkup(keyboard))
        return

    elif query.data.startswith('model_'):
        model = query.data[6:]
        config['MODEL_SELECTION'] = model
        save_group_config(chat_id, config)
        # Return to settings menu to show updated value
        await query.edit_message_text(
            f"✅ Model selection changed to {model}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Settings", callback_data='settings_main')]
            ]))
        return

    elif query.data.startswith('set_'):
        setting_map = {
            'NSFW_THRESHOLD': 'NSFW_THRESHOLD',
            'SAFE_THRESHOLD': 'SAFE_THRESHOLD',
            'FRAME_ANALYSIS_COUNT': 'FRAME_ANALYSIS_COUNT',
            'MIN_DETECTION_CONFIDENCE': 'MIN_DETECTION_CONFIDENCE'
        }
        
        setting = query.data[4:]  # Gets the part after 'set_'
        config_key = setting_map.get(setting, setting)
        
        if config_key not in config:
            await query.edit_message_text(
                f"❌ Configuration error: {config_key} not found",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Back to Settings", callback_data='settings_main')]
                ])
            )
            return
            
        context.user_data['awaiting_setting'] = {
            'chat_id': chat_id,
            'setting': config_key,  # Use the correct config key
            'message_id': query.message.message_id
        }
        
        await query.edit_message_text(
            f"Enter new value for {setting.replace('_', ' ')} (current: {config[config_key]}):\n\n"
            f"• For thresholds, enter value between 0.0 and 1.0\n"
            f"• For frames, enter number between 1 and 10"
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    if 'awaiting_setting' not in context.user_data:
        return
        
    setting_data = context.user_data.pop('awaiting_setting')
    chat_id = setting_data['chat_id']
    setting = setting_data['setting']  # This will be the full setting name
    message_id = setting_data['message_id']
    
    try:
        config = load_group_config(chat_id)
        
        if setting == 'FRAME_ANALYSIS_COUNT':
            value = int(update.message.text)
            if not (1 <= value <= 10):
                raise ValueError("Frames must be between 1 and 10")
        else:
            value = float(update.message.text)
            if not (0 <= value <= 1):
                raise ValueError("Threshold must be between 0.0 and 1.0")
        
        config[setting] = value  # Using the full setting name
        save_group_config(chat_id, config)
        
        
        # Show confirmation and return to settings menu
        await update.message.reply_text(f"✅ {setting.replace('_', ' ')} updated to {value}")
        
        # Create fresh settings menu
        config = load_group_config(chat_id)  # Reload to confirm
        keyboard = [
            [InlineKeyboardButton(f"🔞 NSFW Threshold ({config['NSFW_THRESHOLD']:.2f})", callback_data='set_NSFW_THRESHOLD')],
            [InlineKeyboardButton(f"✅ Safe Threshold ({config['SAFE_THRESHOLD']:.2f})", callback_data='set_SAFE_THRESHOLD')],
            [InlineKeyboardButton(f"🎞 Frames Analyzed ({config['FRAME_ANALYSIS_COUNT']})", callback_data='set_FRAME_ANALYSIS_COUNT')],
            [InlineKeyboardButton(f"🎯 Min Confidence ({config['MIN_DETECTION_CONFIDENCE']:.2f})", callback_data='set_MIN_DETECTION_CONFIDENCE')],
            [InlineKeyboardButton(f"👑 Ignore Admins ({'ON' if config['IGNORE_ADMINS'] else 'OFF'})", callback_data='toggle_ignore')],
            [InlineKeyboardButton(f"🤖 Model ({config['MODEL_SELECTION']})", callback_data='set_model')],
            [InlineKeyboardButton("📦 Blocklist Management", callback_data='blocklist_menu')],
            [InlineKeyboardButton("❌ Close Menu", callback_data='close_menu')]
        ]
        
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚙️ Group Settings Panel ⚙️\nChoose a setting to modify:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
    except ValueError as e:
        await update.message.reply_text(f"❌ Invalid value: {str(e)}")
        # Restore the awaiting state so they can try again
        context.user_data['awaiting_setting'] = setting_data
    except Exception as e:
        await update.message.reply_text("❌ Failed to update setting. Please try again.")
        logger.error(f"Error updating setting: {str(e)}")


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        message = update.effective_message
        chat_id = message.chat.id
        config = load_group_config(chat_id)
        
        if config['IGNORE_ADMINS'] and await is_group_admin(update, context):
            logger.info("👑 Admin content ignored")
            return

        blocklist = load_group_blocklist(chat_id)
        
        # Check blocklists
        if message.sticker:
            sticker = message.sticker
            if sticker.file_unique_id in blocklist['stickers']:
                await message.delete()
                await message.reply_text("⛔ This sticker is blocked in this group")
                return
            if sticker.set_name and sticker.set_name in blocklist['packs']:
                await message.delete()
                await message.reply_text("⛔ This sticker pack is blocked in this group")
                return

        if message.animation:
            gif_id = message.animation.file_unique_id
            if gif_id in blocklist['gifs']:
                await message.delete()
                await message.reply_text("⛔ This GIF is blocked in this group")
                return

        file = None
        is_video = False
        media_type = "unknown"

        if message.photo:
            file = await message.photo[-1].get_file()
            media_type = "photo"
        elif message.sticker:
            if message.sticker.is_animated:
                logger.info("⚠️ Analyzing animated sticker frames")
                file = await message.sticker.get_file()
                media_type = "animated_sticker"
                is_video = True
            else:
                file = await message.sticker.get_file()
                media_type = "sticker"
        elif message.animation:
            file = await message.animation.get_file()
            media_type = "animation"
            is_video = True
        elif message.document:
            if message.document.mime_type.startswith('image/'):
                file = await message.document.get_file()
                media_type = "image_document"
            elif message.document.mime_type.startswith('video/'):
                file = await message.document.get_file()
                media_type = "video_document"
                is_video = True
            else:
                logger.info(f"⚠️ Unsupported document type: {message.document.mime_type}")
                return

        if not file:
            logger.warning("⚠️ No file found in message")
            return

        file_ext = os.path.splitext(file.file_path)[1] if file.file_path else '.jpg'
        file_id = str(uuid.uuid4())
        file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}{file_ext}")
        
        logger.info(f"⬇️ Downloading {media_type} (size: {file.file_size or 'unknown'} bytes)")
        await file.download_to_drive(custom_path=file_path)

        analysis_msg = await message.reply_text(
            f"🔍 Analyzing {media_type} with {config['MODEL_SELECTION']} model..."
            if not is_video else
            f"🎬 Analyzing {config['FRAME_ANALYSIS_COUNT']} frames from {media_type}..."
        )

        try:
            if is_video or file_ext in ['.gif', '.webm', '.mp4']:
                confidence = analyze_frames(file_path, chat_id)
                is_nsfw_content = confidence > config['NSFW_THRESHOLD']
            else:
                is_nsfw_content, confidence = detect_nsfw(file_path, chat_id)

            if is_nsfw_content and confidence > config['NSFW_THRESHOLD']:
                await message.delete()
                logger.info(f"🚫 Deleted NSFW content (Confidence: {confidence:.2%})")
                
                warning_msg = (
                    f"⚠️ Removed explicit content\n"
                    f"Confidence: {confidence:.2%}\n"
                    f"Model: {config['MODEL_SELECTION']}\n"
                    f"Media type: {media_type}\n\n"
                    f"*This action was performed automatically*"
                )
                
                await message.chat.send_message(
                    warning_msg,
                    reply_to_message_id=message.message_id if message.chat.type != 'private' else None
                )
            else:
                logger.info(f"✅ Safe content (Confidence: {confidence:.2%})")
                await analysis_msg.edit_text(
                    f"✅ Content approved\n"
                    f"Confidence: {confidence:.2%}\n"
                    f"Model: {config['MODEL_SELECTION']}"
                )

        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            await analysis_msg.edit_text("❌ Analysis failed. Please try again.")

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        logger.error(f"🔥 Critical error in media handler: {str(e)}")
        if 'analysis_msg' in locals():
            await analysis_msg.edit_text("❌ An error occurred during processing")


async def delete_messages(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_ids: list):
    """Delete multiple messages"""
    try:
        for msg_id in message_ids:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
    except Exception as e:
        logger.error(f"Failed to delete messages: {e}")

async def add_message_to_cleanup(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int):
    """Store message ID for later cleanup"""
    if 'messages_to_clean' not in context.chat_data:
        context.chat_data['messages_to_clean'] = []
    context.chat_data['messages_to_clean'].append(message_id)
    
async def post_init(application):
    await application.bot.set_my_commands([
        ("start", "Start the bot"),
        ("help", "Show help information"),
        ("settings", "Configure group settings (admins only)")
    ])
    logger.info("🤖 Bot initialization complete!")

async def post_stop(application):
    logger.info("🛑 Bot shutdown complete!")

def main():
    try:
        bot_token = os.environ.get("BOT_TOKEN")  # Get from Secrets
        
        # Start Flask server in a thread
        threading.Thread(target=run_flask, daemon=True).start()
        
        app = ApplicationBuilder() \
            .token(bot_token) \
            .post_init(post_init) \
            .post_stop(post_stop) \
            .build()

        # Command handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("settings", group_settings))
        app.add_handler(CommandHandler("block", block_command))  # Add this line

        # Button handler
        app.add_handler(CallbackQueryHandler(handle_button))

        # Message handlers
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Media handler - captures all supported media types
        media_filter = (
            filters.PHOTO |
            filters.Document.IMAGE |
            filters.Document.VIDEO |
            filters.Sticker.ALL |
            filters.ANIMATION
            )
        app.add_handler(MessageHandler(media_filter, handle_media))

        logger.info("🤖 Starting Advanced NSFW Detection Bot...")
        logger.info(f"✅ Models initialized! Default model: {DEFAULT_CONFIG['MODEL_SELECTION']}")
        # logger.info(f"🔧 Model selection: {MODEL_SELECTION}")
        logger.info(f"📁 Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
        logger.info("📸 Monitoring: Photos | Videos | GIFs | Stickers")
        
        # Run with better error handling
        logger.info("🤖 Starting Advanced NSFW Detection Bot...")
        app.run_polling(
            poll_interval=1.0,
            timeout=30,
            drop_pending_updates=True
        )

    except Exception as e:
        logger.critical(f"💥 Fatal error during startup: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        import asyncio
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
         # Start main application
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.critical(f"💥 Critical error: {str(e)}")
