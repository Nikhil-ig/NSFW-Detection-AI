# import os
# import logging
# from telegram.ext import ApplicationBuilder, MessageHandler, filters
# from handlers.media_handler import MediaHandler

# # Logging setup
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )
# logger = logging.getLogger("NSFWBot")

# def main():
#     bot_token = "7803429144:AAFXixTN0-Gb2eX1GE2KJnlTHdvfJBLrlnM"

#     # Ensure download directory exists
#     os.makedirs("downloads", exist_ok=True)

#     app = ApplicationBuilder().token(bot_token).build()

#      # "nudenet", "tf", or "both"
#     media_handler = MediaHandler(model_choice="nudenet")
#     app.add_handler(MessageHandler(media_handler.supported_filters, media_handler.handle))

#     logger.info("🤖 Bot started. Monitoring media...")
#     app.run_polling()

# if __name__ == "__main__":
#     main()

#######################################################################################
#######################################################################################
# Not Working
#######################################################################################
#######################################################################################
# import os
# import logging
# import uuid
# import cv2
# import asyncio
# import numpy as np
# from PIL import Image, ImageSequence
# from nudenet import NudeDetector
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from telegram import Update
# from telegram.ext import (
#     ApplicationBuilder,
#     MessageHandler,
#     ContextTypes,
#     filters,
#     CommandHandler
# )

# # Enhanced logging configuration
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO,
#     handlers=[
#         logging.FileHandler("nsfw_bot.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Configuration
# DOWNLOAD_DIR = "downloads"
# NSFW_THRESHOLD = 0.82  # Optimized threshold for better accuracy
# FRAME_ANALYSIS_COUNT = 5  # Increased frame analysis for videos/GIFs
# MIN_DETECTION_CONFIDENCE = 0.10  # Lower threshold to catch more potential cases

# # Model configuration
# MODEL_SELECTION = "nudenet"  # Options: "nudenet", "tensorflow", or "both"
# TF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'nsfw.299x299.h5')
# TF_INPUT_SIZE = (299, 299)  # For the 299x299 model

# # Comprehensive NSFW classes (expanded list with kisses category)
# NSFW_CLASSES = {
#     # Explicit nudity
#     'FEMALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED',
#     'FEMALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED',
#     'ANUS_EXPOSED', 'FEMALE_BREAST_COVERED', 'MALE_GENITALIA_COVERED',
#     'FEMALE_GENITALIA_COVERED', 'BUTTOCKS_COVERED',
    
#     # Partial/semi-nudity
#     'FEMALE_SEMI_NUDE', 'MALE_SEMI_NUDE', 'UNDRESS',
#     'EXPOSED_UNDERWEAR', 'SEE_THROUGH_CLOTHING',
#     'CLOTHING_LIFTING', 'CLOTHING_REMOVAL', 'CLEAVAGE',
#     'SHORT_CLOTHES', 'BATHING_SUIT', 'LINGERIE',
    
#     # Sexual activities
#     'SEXUAL_ACTIVITY', 'MASTURBATION', 'SEX_TOYS',
#     'PORNOGRAPHIC_POSES', 'SUGGESTIVE_POSES',
#     'KINK_ACTIVITY', 'BDSM', 'KISSING', 'INTIMATE_EMBRACE',
#     'FOREPLAY', 'SEXUAL_GESTURES', 'PROVOCATIVE_POSES',
    
#     # Kisses and intimate contact
#     'FRENCH_KISS', 'INTIMATE_KISS', 'MOUTH_TO_MOUTH',
#     'NECK_KISS', 'EAR_KISS', 'HICKY', 'LOVE_BITE',
#     'INTIMATE_TOUCH', 'FACE_TO_FACE_CLOSE',
    
#     # Fetish content
#     'FEET_FETISH', 'UNDERWEAR_FETISH', 'UNIFORM_FETISH',
#     'LEATHER_FETISH', 'LATEX_FETISH', 'BONDAGE',
    
#     # Inappropriate content
#     'DRUG_USE', 'ALCOHOL_CONSUMPTION', 'TOBACCO_USE',
#     'VIOLENCE', 'BLOOD', 'GORE', 'WEAPONS',
    
#     # Other sensitive content
#     'HATE_SYMBOLS', 'DISTURBING_CONTENT', 'SELF_HARM',
#     'BULLYING', 'HARASSMENT'
# }

# # Initialize models
# logger.info("⚙️ Loading advanced detection models...")
# nude_detector = NudeDetector()  # Removed provider parameter as it's not supported in current version

# # Load TensorFlow model with enhanced error handling
# tf_model = None
# try:
#     tf_model = load_model(
#         TF_MODEL_PATH,
#         custom_objects={'KerasLayer': tf.keras.layers.Layer}
#     )
#     logger.info(f"✅ TensorFlow model ({TF_MODEL_PATH}) loaded successfully!")
    
#     # Model warm-up and verification
#     dummy_input = np.random.rand(1, *TF_INPUT_SIZE, 3).astype(np.float32)
#     prediction = tf_model.predict(dummy_input)
#     logger.info(f"🔍 Model warm-up successful. Output shape: {prediction.shape}")
    
#     if tf_model is None:
#         raise ValueError("Model loaded but is None")
# except Exception as e:
#     logger.error(f"❌ Critical error loading TensorFlow model: {str(e)}")
#     if "CUDA" in str(e):
#         logger.warning("⚠️ Trying to load with CPU only...")
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         try:
#             tf_model = load_model(TF_MODEL_PATH)
#         except Exception as e:
#             logger.error(f"❌ CPU load failed: {str(e)}")
#             tf_model = None
#     if tf_model is None:
#         MODEL_SELECTION = "nudenet"
#         logger.warning("⚠️ Falling back to NudeNet only mode")

# logger.info(f"✅ Models initialized! Active model: {MODEL_SELECTION}")

# def preprocess_image_for_tf(image_path: str, target_size=TF_INPUT_SIZE) -> np.ndarray:
#     """Enhanced image preprocessing with better error handling"""
#     try:
#         # Load with explicit color mode
#         img = tf.keras.preprocessing.image.load_img(
#             image_path,
#             target_size=target_size,
#             color_mode='rgb'
#         )
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
        
#         # Advanced preprocessing
#         img_array = tf.expand_dims(img_array, 0)  # Create batch axis
#         img_array = img_array / 255.0  # Normalize to [0,1]
        
#         return img_array
#     except Exception as e:
#         logger.error(f"❌ Advanced preprocessing failed: {str(e)}")
#         # Return blank image with correct dimensions
#         return np.zeros((1, *target_size, 3), dtype=np.float32)

# def detect_nsfw_tensorflow(image_path: str) -> tuple[bool, float]:
#     """Enhanced TensorFlow detection with better output handling"""
#     if tf_model is None:
#         return False, 0.0
    
#     try:
#         img_array = preprocess_image_for_tf(image_path)
#         predictions = tf_model.predict(img_array)
        
#         # Handle different model output formats
#         if predictions.shape[1] == 2:  # Standard [sfw, nsfw] output
#             nsfw_confidence = float(predictions[0][1])
#         elif predictions.shape[1] == 5:  # Some models have 5 classes
#             # Sum probabilities of NSFW classes (adjust indices as needed)
#             nsfw_confidence = float(predictions[0][1] + predictions[0][2] + predictions[0][3])
#         else:  # Single output
#             nsfw_confidence = float(predictions[0][0])
            
#         is_nsfw = nsfw_confidence > NSFW_THRESHOLD
#         logger.debug(f"🔍 TF Detection - Confidence: {nsfw_confidence:.4f}, NSFW: {is_nsfw}")
#         return is_nsfw, nsfw_confidence
#     except Exception as e:
#         logger.error(f"🔞 TensorFlow detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_nudenet(image_path: str) -> tuple[bool, float]:
#     """Enhanced NudeNet detection with expanded classes"""
#     try:
#         detections = nude_detector.detect(image_path)
        
#         # Filter detections by our expanded NSFW classes
#         valid_detections = [
#             det for det in detections 
#             if det['class'] in NSFW_CLASSES 
#             and det['score'] >= MIN_DETECTION_CONFIDENCE
#         ]
        
#         if not valid_detections:
#             return False, 0.0
            
#         # Calculate weighted confidence based on class severity
#         class_weights = {
#             'FEMALE_GENITALIA_EXPOSED': 1.0,
#             'MALE_GENITALIA_EXPOSED': 1.0,
#             'SEXUAL_ACTIVITY': 0.9,
#             'FEMALE_BREAST_EXPOSED': 0.8,
#             'ANUS_EXPOSED': 0.9,
#             'KISSING': 0.5,
#             'INTIMATE_KISS': 0.6,
#             'FRENCH_KISS': 0.7,
#             # Add weights for other classes...
#         }
        
#         weighted_confidences = [
#             det['score'] * class_weights.get(det['class'], 0.5)  # Default weight 0.5
#             for det in valid_detections
#         ]
        
#         max_confidence = max(weighted_confidences)
#         avg_confidence = sum(weighted_confidences) / len(weighted_confidences)
        
#         # Use combination of max and average confidence
#         combined_confidence = (max_confidence * 0.7 + avg_confidence * 0.3)
        
#         is_nsfw = combined_confidence > NSFW_THRESHOLD
#         logger.debug(f"🔍 NudeNet Detection - Max: {max_confidence:.4f}, Avg: {avg_confidence:.4f}, Combined: {combined_confidence:.4f}")
#         return is_nsfw, combined_confidence
#     except Exception as e:
#         logger.error(f"🔞 NudeNet detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_combined(image_path: str) -> tuple[bool, float]:
#     """Advanced combined detection with confidence fusion"""
#     nudenet_result, nudenet_conf = detect_nsfw_nudenet(image_path)
#     tf_result, tf_conf = detect_nsfw_tensorflow(image_path)
    
#     # Dynamic weighting based on confidence levels
#     if nudenet_conf > 0.7 or tf_conf > 0.7:
#         # High confidence in either model gets more weight
#         combined_confidence = (nudenet_conf * 0.7 + tf_conf * 0.3)
#     else:
#         # Normal case - balanced weighting
#         combined_confidence = (nudenet_conf * 0.6 + tf_conf * 0.4)
    
#     # Additional check - if both models agree on NSFW, increase confidence
#     if nudenet_result and tf_result:
#         combined_confidence = min(1.0, combined_confidence * 1.2)
    
#     combined_result = combined_confidence > NSFW_THRESHOLD
    
#     logger.info(
#         f"🔍 Combined Detection\n"
#         f"NudeNet: {nudenet_conf:.4f} ({'NSFW' if nudenet_result else 'Safe'})\n"
#         f"TensorFlow: {tf_conf:.4f} ({'NSFW' if tf_result else 'Safe'})\n"
#         f"Final: {combined_confidence:.4f} ({'NSFW' if combined_result else 'Safe'})"
#     )
#     return combined_result, combined_confidence

# def detect_nsfw(image_path: str) -> tuple[bool, float]:
#     """Route to appropriate detection method based on model selection"""
#     if MODEL_SELECTION == "nudenet":
#         return detect_nsfw_nudenet(image_path)
#     elif MODEL_SELECTION == "tensorflow":
#         return detect_nsfw_tensorflow(image_path)
#     else:  # both
#         return detect_nsfw_combined(image_path)

# def analyze_frames(file_path: str) -> float:
#     """Analyze multiple frames for video/GIF content"""
#     max_confidence = 0.0
#     try:
#         if file_path.endswith(('.webm', '.mp4')):
#             vid = cv2.VideoCapture(file_path)
#             total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_step = max(1, total_frames // FRAME_ANALYSIS_COUNT)
            
#             for i in range(FRAME_ANALYSIS_COUNT):
#                 vid.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
#                 success, frame = vid.read()
#                 if success:
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     cv2.imwrite(frame_path, frame)
#                     _, confidence = detect_nsfw(frame_path)
#                     max_confidence = max(max_confidence, confidence)
#                     os.remove(frame_path)
#             vid.release()
            
#         elif file_path.endswith('.gif'):
#             with Image.open(file_path) as img:
#                 total_frames = img.n_frames
#                 frame_step = max(1, total_frames // FRAME_ANALYSIS_COUNT)
                
#                 for i in range(FRAME_ANALYSIS_COUNT):
#                     img.seek(i * frame_step)
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     img.convert('RGB').save(frame_path)
#                     _, confidence = detect_nsfw(frame_path)
#                     max_confidence = max(max_confidence, confidence)
#                     os.remove(frame_path)
                    
#         return max_confidence
#     except Exception as e:
#         logger.error(f"🎞 Frame analysis failed: {str(e)}")
#         return 0.0

# async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle model selection command"""
#     global MODEL_SELECTION
#     if not context.args:
#         await update.message.reply_text(
#             f"Current model: {MODEL_SELECTION}\n"
#             "Available options: /model nudenet, /model tensorflow, /model both"
#         )
#         return
    
#     selection = context.args[0].lower()
#     if selection in ("nudenet", "tensorflow", "both"):
#         if selection == "tensorflow" and tf_model is None:
#             await update.message.reply_text("TensorFlow model not available. Using NudeNet.")
#             return
            
#         MODEL_SELECTION = selection
#         await update.message.reply_text(f"Model selection changed to: {MODEL_SELECTION}")
#         logger.info(f"Model selection changed to: {MODEL_SELECTION}")
#     else:
#         await update.message.reply_text("Invalid model. Choose from: nudenet, tensorflow, both")

# async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Enhanced media handler with better feedback"""
#     try:
#         message = update.effective_message
#         user = message.from_user
#         chat = message.chat
        
#         logger.info(f"📩 New media from {user.full_name} ({user.id}) in {chat.title if chat.type != 'private' else 'private chat'}")
        
#         file = None
#         is_video = False
#         media_type = "unknown"

#         # Enhanced media type handling
#         if message.photo:
#             file = await message.photo[-1].get_file()
#             media_type = "photo"
#         elif message.sticker:
#             if message.sticker.is_animated:
#                 logger.info("⚠️ Skipping animated sticker")
#                 return
#             file = await message.sticker.get_file()
#             media_type = "sticker"
#         elif message.animation:
#             file = await message.animation.get_file()
#             media_type = "animation"
#             is_video = True
#         elif message.document:
#             if message.document.mime_type.startswith('image/'):
#                 file = await message.document.get_file()
#                 media_type = "image_document"
#             elif message.document.mime_type.startswith('video/'):
#                 file = await message.document.get_file()
#                 media_type = "video_document"
#                 is_video = True
#             else:
#                 logger.info(f"⚠️ Unsupported document type: {message.document.mime_type}")
#                 return

#         if not file:
#             logger.warning("⚠️ No file found in message")
#             return

#         # Download with proper extension and hash
#         file_ext = os.path.splitext(file.file_path)[1] if file.file_path else '.jpg'
#         file_id = str(uuid.uuid4())
#         file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}{file_ext}")
        
#         logger.info(f"⬇️ Downloading {media_type} (size: {file.file_size or 'unknown'} bytes)")
#         await file.download_to_drive(custom_path=file_path)

#         # Enhanced analysis with progress feedback
#         analysis_msg = await message.reply_text(
#             f"🔍 Analyzing {media_type} with {MODEL_SELECTION} model..."
#             if not is_video else
#             f"🎬 Analyzing {FRAME_ANALYSIS_COUNT} frames from video..."
#         )

#         try:
#             # Multi-frame analysis for videos/GIFs
#             if is_video or file_ext in ['.gif', '.webm', '.mp4']:
#                 confidence = analyze_frames(file_path)
#                 is_nsfw_content = confidence > NSFW_THRESHOLD
#             else:
#                 # Single image processing
#                 is_nsfw_content, confidence = detect_nsfw(file_path)

#             # Enhanced result handling
#             if is_nsfw_content:
#                 await message.delete()
#                 logger.info(f"🚫 Deleted NSFW content (Confidence: {confidence:.2%})")
                
#                 warning_msg = (
#                     f"⚠️ Removed explicit content\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}\n"
#                     f"Media type: {media_type}\n\n"
#                     f"*This action was performed automatically*"
#                 )
                
#                 await chat.send_message(
#                     warning_msg,
#                     reply_to_message_id=message.message_id if chat.type != 'private' else None
#                 )
#             else:
#                 logger.info(f"✅ Safe content (Confidence: {confidence:.2%})")
#                 await analysis_msg.edit_text(
#                     f"✅ Content approved\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}"
#                 )

#         except Exception as e:
#             logger.error(f"❌ Analysis failed: {str(e)}")
#             await analysis_msg.edit_text("❌ Analysis failed. Please try again.")
#             raise

#         finally:
#             # Cleanup
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#             else:
#                 logger.warning(f"⚠️ File not found during cleanup: {file_path}")

#     except Exception as e:
#         logger.error(f"🔥 Critical error in media handler: {str(e)}")
#         if 'analysis_msg' in locals():
#             await analysis_msg.edit_text("❌ An error occurred during processing.")

# async def start_bot():
#     await bot.initialize()
#     # Further async calls or startup logic here

# def main():
#     """Enhanced main function with better initialization"""
#     try:
#         bot_token = "7803429144:AAFXixTN0-Gb2eX1GE2KJnlTHdvfJBLrlnM"  # Replace with your actual token
        
#         # Create downloads directory if not exists
#         os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        
#         # Initialize application with better error handling
#         app = ApplicationBuilder() \
#             .token(bot_token) \
#             .post_init(lambda _: logger.info("🤖 Bot initialization complete!")) \
#             .post_stop(lambda _: logger.info("🛑 Bot shutdown complete!")) \
#             .build()
        
#         # Enhanced media filter
#         media_filter = (
#             filters.PHOTO |
#             filters.Document.IMAGE |
#             filters.Document.VIDEO |
#             filters.Sticker.STATIC |
#             filters.ANIMATION
#         )
        
#         # Add handlers with priority
#         app.add_handler(CommandHandler("model", handle_model_selection), group=1)
#         app.add_handler(MessageHandler(media_filter, handle_media), group=2)
        
#         # Startup message
#         logger.info("🤖 Advanced NSFW Detection Bot Starting...")
#         logger.info(f"🔧 Model selection: {MODEL_SELECTION}")
#         logger.info(f"📁 Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
#         logger.info("📸 Monitoring: Photos | Images | Videos | GIFs | Static Stickers")
        
#         # Run with better error handling
#         app.run_polling(
#             poll_interval=1,
#             timeout=30,
#             drop_pending_updates=True
#         )
        
#     except Exception as e:
#         logger.critical(f"💥 Fatal error during startup: {str(e)}")
#         raise

# if __name__ == "__main__":
#     try: 
#         asyncio.run(start_bot())  # Start the async bot
#         main()
#     except KeyboardInterrupt:
#         logger.info("🛑 Bot stopped by user")
#     except Exception as e:
#         logger.critical(f"💥 Critical error: {str(e)}")


#     import os
# import logging
# import uuid
# import cv2
# import numpy as np
# from PIL import Image, ImageSequence
# from nudenet import NudeDetector
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from telegram import Update
# from telegram.ext import (
#     ApplicationBuilder,
#     MessageHandler,
#     ContextTypes,
#     filters,
#     CommandHandler
# )

# # Setup logging
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )
# logger = logging.getLogger(__name__)

# # Configuration
# DOWNLOAD_DIR = "downloads"
# NSFW_THRESHOLD = 0.90  # Higher threshold for stricter detection
# FRAME_ANALYSIS_COUNT = 5  # More frames for better video analysis
# MIN_DETECTION_CONFIDENCE = 0.20  # Higher minimum confidence

# # Model selection
# MODEL_SELECTION = "both"  # "nudenet", "tensorflow", or "both"
# TF_MODEL_PATH = "nsfw.299x299.h5"

# # Enhanced NSFW classes
# NSFW_CLASSES = {
#     # Explicit nudity
#     'FEMALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED',
#     'FEMALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED',
#     'ANUS_EXPOSED', 'PUBIC_HAIR', 'NAKED',
    
#     # Partial/semi-nudity
#     'FEMALE_SEMI_NUDE', 'MALE_SEMI_NUDE', 'UNDRESS',
#     'EXPOSED_UNDERWEAR', 'SEE_THROUGH', 'CLOTHES_OFF',
#     'BATHING_SUIT', 'LINGERIE', 'UNDERWEAR',
    
#     # Sexual activity
#     'SEXUAL_ACTIVITY', 'SEX_TOYS', 'MASTURBATION',
#     'PORNOGRAPHY', 'SEXUAL_POSITIONS', 'KINKY',
    
#     # Suggestive content
#     'PROVOCATIVE_POSE', 'SPREAD_LEGS', 'FOCUS_ON_BREASTS',
#     'FOCUS_ON_GROIN', 'UPSKIRT', 'DOWNBLOS',
    
#     # Other explicit content
#     'BONDAGE', 'FETISH', 'BDSM', 'URINATION',
#     'DEFECATION', 'VOMITING', 'BLOOD'
# }

# # Initialize models
# logger.info("⚙️ Loading enhanced detection models...")
# nude_detector = NudeDetector()

# # Load TensorFlow model with custom objects
# def load_tf_model():
#     try:
#         model = load_model(TF_MODEL_PATH, compile=False)
#         logger.info(f"✅ TensorFlow model ({TF_MODEL_PATH}) loaded successfully!")
        
#         # Warm up the model
#         dummy_input = np.zeros((1, 299, 299, 3), dtype=np.float32)
#         _ = model.predict(dummy_input)
#         return model
#     except Exception as e:
#         logger.error(f"❌ Failed to load TensorFlow model: {str(e)}")
#         return None

# tf_model = load_tf_model()

# logger.info("✅ All models initialized!")

# def enhanced_preprocessing(image_path: str) -> np.ndarray:
#     """Advanced image preprocessing with multiple techniques"""
#     try:
#         # Load with PIL first for better format handling
#         img = Image.open(image_path)
        
#         # Convert to RGB if not already
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
            
#         # Resize with anti-aliasing
#         img = img.resize((299, 299), Image.LANCZOS)
        
#         # Convert to numpy array and normalize
#         img_array = np.array(img) / 255.0
        
#         # Add batch dimension
#         img_array = np.expand_dims(img_array, axis=0)
        
#         return img_array.astype(np.float32)
#     except Exception as e:
#         logger.error(f"❌ Enhanced preprocessing failed: {str(e)}")
#         return np.zeros((1, 299, 299, 3), dtype=np.float32)

# def detect_nsfw_tensorflow(image_path: str) -> tuple[bool, float]:
#     """Enhanced TensorFlow detection with multiple checks"""
#     if tf_model is None:
#         return False, 0.0
    
#     try:
#         img_array = enhanced_preprocessing(image_path)
#         predictions = tf_model.predict(img_array)
        
#         # Handle different model output formats
#         if predictions.shape[1] == 2:  # [sfw, nsfw]
#             nsfw_confidence = float(predictions[0][1])
#         elif predictions.shape[1] == 5:  # Multi-class model
#             # Sum probabilities of NSFW classes (adjust indices as needed)
#             nsfw_confidence = float(predictions[0][1] + predictions[0][2] + predictions[0][3])
#         else:
#             nsfw_confidence = float(predictions[0][0])  # Fallback
        
#         is_nsfw = nsfw_confidence > NSFW_THRESHOLD
#         return is_nsfw, nsfw_confidence
#     except Exception as e:
#         logger.error(f"🔞 TensorFlow detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_nudenet(image_path: str) -> tuple[bool, float]:
#     """Enhanced NudeNet detection with expanded classes"""
#     try:
#         detections = nude_detector.detect(image_path)
        
#         # Filter detections by our expanded NSFW classes
#         valid_detections = [
#             det for det in detections 
#             if det['class'] in NSFW_CLASSES 
#             and det['score'] >= MIN_DETECTION_CONFIDENCE
#         ]
        
#         if not valid_detections:
#             return False, 0.0
            
#         # Calculate weighted confidence based on class severity
#         weighted_confidences = []
#         for det in valid_detections:
#             weight = 1.0
#             # Increase weight for more explicit classes
#             if det['class'] in {'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED', 'SEXUAL_ACTIVITY'}:
#                 weight = 1.5
#             weighted_confidences.append(det['score'] * weight)
            
#         max_confidence = max(weighted_confidences)
#         return max_confidence > NSFW_THRESHOLD, max_confidence
#     except Exception as e:
#         logger.error(f"🔞 NudeNet detection failed: {str(e)}")
#         return False, 0.0

# def ensemble_detection(image_path: str) -> tuple[bool, float]:
#     """Advanced ensemble detection using multiple strategies"""
#     # Get results from both models
#     nudenet_result, nudenet_conf = detect_nsfw_nudenet(image_path)
#     tf_result, tf_conf = detect_nsfw_tensorflow(image_path)
    
#     # Apply different voting strategies
#     strategies = {
#         'conservative': nudenet_result or tf_result,  # Either model flags it
#         'average': (nudenet_conf + tf_conf) / 2 > NSFW_THRESHOLD,
#         'weighted': (nudenet_conf * 0.7 + tf_conf * 0.3) > NSFW_THRESHOLD
#     }
    
#     # Count how many strategies consider it NSFW
#     nsfw_votes = sum(1 for result in strategies.values() if result)
    
#     # Final decision (at least 2 out of 3 strategies must agree)
#     is_nsfw = nsfw_votes >= 2
#     combined_confidence = max(nudenet_conf, tf_conf)
    
#     return is_nsfw, combined_confidence

# def analyze_video_frames(file_path: str) -> tuple[bool, float]:
#     """Advanced frame analysis with keyframe detection"""
#     max_confidence = 0.0
#     try:
#         if file_path.endswith(('.webm', '.mp4', '.mov')):
#             cap = cv2.VideoCapture(file_path)
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
#             # Analyze more frames for longer videos
#             frame_count = min(FRAME_ANALYSIS_COUNT + (total_frames // 100), 10)
            
#             for i in range(frame_count):
#                 # Seek to key frames for better detection
#                 frame_pos = int((i / frame_count) * total_frames)
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
#                 ret, frame = cap.read()
#                 if ret:
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     cv2.imwrite(frame_path, frame)
                    
#                     # Use ensemble detection for each frame
#                     _, confidence = ensemble_detection(frame_path)
#                     max_confidence = max(max_confidence, confidence)
                    
#                     os.remove(frame_path)
#             cap.release()
            
#         elif file_path.endswith('.gif'):
#             with Image.open(file_path) as img:
#                 total_frames = img.n_frames
#                 frame_count = min(FRAME_ANALYSIS_COUNT + (total_frames // 20), 8)
                
#                 for i in range(frame_count):
#                     frame_pos = int((i / frame_count) * total_frames)
#                     img.seek(frame_pos)
                    
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     img.convert('RGB').save(frame_path)
                    
#                     _, confidence = ensemble_detection(frame_path)
#                     max_confidence = max(max_confidence, confidence)
                    
#                     os.remove(frame_path)
                    
#         return max_confidence > NSFW_THRESHOLD, max_confidence
#     except Exception as e:
#         logger.error(f"🎞 Advanced frame analysis failed: {str(e)}")
#         return False, 0.0
######################################################################
######################################################################
######################################################################
# import os
# import logging
# import uuid
# import cv2
# import numpy as np
# from PIL import Image, ImageSequence
# from nudenet import NudeDetector
# from telegram import Update
# from telegram.ext import (
#     ApplicationBuilder,
#     MessageHandler,
#     ContextTypes,
#     filters,
#     CommandHandler
# )

# # Setup logging
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO,
#     handlers=[
#         logging.FileHandler("nsfw_bot.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Configuration
# DOWNLOAD_DIR = "downloads"
# os.makedirs(DOWNLOAD_DIR, exist_ok=True)
# NSFW_THRESHOLD = 0.1  # Confidence threshold for NSFW detection
# MIN_DETECTION_CONFIDENCE = 0.1  # Minimum confidence to consider a detection valid
# FRAME_ANALYSIS_COUNT = 2  # Number of frames to analyze for videos/GIFs

# # Initialize detector
# logger.info("⚙️ Loading NudeNet model...")
# detector = NudeDetector()

# # NSFW classes to detect
# NSFW_CLASSES = {
#     'FEMALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED',
#     'FEMALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED',
#     'ANUS_EXPOSED', 'FEMALE_SEMI_NUDE', 'MALE_SEMI_NUDE',
#     'UNDRESS', 'SEXUAL_ACTIVITY', 'EXPOSED_UNDERWEAR',
#     'KISSING', 'INTIMATE_EMBRACE'
# }

# async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle all incoming media messages"""
#     try:
#         message = update.effective_message
#         if not message:
#             logger.warning("No message found in update")
#             return

#         # Get basic info about the sender and chat
#         user = message.from_user
#         chat = message.chat
#         logger.info(f"📩 New message from {user.full_name} ({user.id}) in {chat.title if chat.type != 'private' else 'private chat'}")

#         # Determine media type and get file object
#         file = None
#         is_video = False
#         file_ext = '.jpg'  # Default extension
        
#         if message.photo:
#             file = await message.photo[-1].get_file()  # Get highest resolution photo
#             media_type = "photo"
#         elif message.document:
#             if message.document.mime_type.startswith('image/'):
#                 file = await message.document.get_file()
#                 media_type = "image document"
#                 file_ext = os.path.splitext(message.document.file_name)[1] or '.jpg'
#             elif message.document.mime_type.startswith('video/'):
#                 file = await message.document.get_file()
#                 media_type = "video document"
#                 file_ext = os.path.splitext(message.document.file_name)[1] or '.mp4'
#                 is_video = True
#             else:
#                 logger.info(f"Skipping unsupported document type: {message.document.mime_type}")
#                 return
#         elif message.sticker:
#             if message.sticker.is_animated:
#                 logger.info("Skipping animated sticker")
#                 return
#             file = await message.sticker.get_file()
#             media_type = "sticker"
#             file_ext = '.png'
#         elif message.animation:
#             file = await message.animation.get_file()
#             media_type = "animation"
#             file_ext = '.mp4'
#             is_video = True
#         else:
#             logger.info("No supported media found in message")
#             return

#         if not file:
#             logger.warning("Failed to get file from message")
#             return

#         # Download the file
#         file_id = str(uuid.uuid4())
#         file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}{file_ext}")
        
#         logger.info(f"⬇️ Downloading {media_type} (size: {file.file_size or 'unknown'} bytes)")
#         await file.download_to_drive(custom_path=file_path)
#         logger.info(f"✅ Downloaded to {file_path}")

#         # Send processing message
#         processing_msg = await message.reply_text(
#             f"🔍 Analyzing {media_type}..." if not is_video else
#             f"🎬 Analyzing {FRAME_ANALYSIS_COUNT} frames from video..."
#         )

#         try:
#             # Analyze the content
#             is_nsfw = False
#             confidence = 0.0

#             if is_video or file_ext in ['.gif', '.webm', '.mp4']:
#                 # Video/GIF analysis
#                 try:
#                     vid = cv2.VideoCapture(file_path)
#                     total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#                     frames_to_analyze = min(FRAME_ANALYSIS_COUNT, total_frames)
                    
#                     for i in range(frames_to_analyze):
#                         frame_pos = i * (total_frames // frames_to_analyze)
#                         vid.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
#                         success, frame = vid.read()
#                         if success:
#                             frame_path = f"{file_path}_frame_{i}.jpg"
#                             cv2.imwrite(frame_path, frame)
#                             _, frame_confidence = detect_nsfw(frame_path)
#                             confidence = max(confidence, frame_confidence)
#                             os.remove(frame_path)
                    
#                     vid.release()
#                 except Exception as e:
#                     logger.error(f"Error analyzing video: {str(e)}")
#                     await processing_msg.edit_text("❌ Error analyzing video")
#                     return
#             else:
#                 # Image analysis
#                 is_nsfw, confidence = detect_nsfw(file_path)

#             logger.info(f"🔍 Analysis complete - Confidence: {confidence:.2%}")

#             # Handle results
#             if confidence > NSFW_THRESHOLD:
#                 await message.delete()
#                 logger.info(f"🚫 Deleted NSFW content (Confidence: {confidence:.2%})")
#                 await processing_msg.edit_text(
#                     f"⚠️ Removed explicit content\n"
#                     f"Confidence: {confidence:.2%}"
#                 )
#             else:
#                 logger.info(f"✅ Approved content (Confidence: {confidence:.2%})")
#                 await processing_msg.edit_text(
#                     f"✅ Content approved\n"
#                     f"Confidence: {confidence:.2%}"
#                 )

#         except Exception as e:
#             logger.error(f"Error during analysis: {str(e)}")
#             await processing_msg.edit_text("❌ Error during analysis")
#             raise

#         finally:
#             # Cleanup downloaded file
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#                 logger.info(f"🧹 Cleaned up {file_path}")

#     except Exception as e:
#         logger.error(f"🔥 Critical error in media handler: {str(e)}")
#         if 'processing_msg' in locals():
#             await processing_msg.edit_text("❌ An error occurred during processing")

# def detect_nsfw(image_path: str) -> tuple[bool, float]:
#     """Detect NSFW content in an image"""
#     try:
#         detections = detector.detect(image_path)
#         valid_detections = [
#             det for det in detections 
#             if det['class'] in NSFW_CLASSES 
#             and det['score'] >= MIN_DETECTION_CONFIDENCE
#         ]
        
#         if not valid_detections:
#             return False, 0.0
            
#         max_confidence = max(det['score'] for det in valid_detections)
#         return max_confidence > NSFW_THRESHOLD, max_confidence
#     except Exception as e:
#         logger.error(f"Detection failed: {str(e)}")
#         return False, 0.0

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle /start command"""
#     await update.message.reply_text(
#         "🚀 NSFW Detection Bot is running!\n"
#         "I'll automatically analyze photos, videos, and GIFs for explicit content."
#     )

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle /help command"""
#     await update.message.reply_text(
#         "ℹ️ This bot detects NSFW content in:\n"
#         "- Photos\n- Videos\n- GIFs\n- Static stickers\n\n"
#         "It will automatically remove content that exceeds the safety threshold."
#     )

# def main():
#     """Start the bot"""
#     try:
#         bot_token = "7803429144:AAFXixTN0-Gb2eX1GE2KJnlTHdvfJBLrlnM"  # Replace with your actual token
        
#         # Create application
#         app = ApplicationBuilder().token(bot_token).build()

#         # Add command handlers
#         app.add_handler(CommandHandler("start", start))
#         app.add_handler(CommandHandler("help", help_command))

#         # Add media handler
#         media_filter = (
#             filters.PHOTO | 
#             filters.Document.IMAGE | 
#             filters.Document.VIDEO |
#             filters.Sticker.ALL |
#             filters.ANIMATION
#         )
#         app.add_handler(MessageHandler(media_filter, handle_media))

#         logger.info("🤖 Starting bot...")
#         app.run_polling()

#     except Exception as e:
#         logger.critical(f"💥 Failed to start bot: {str(e)}")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         logger.info("🛑 Bot stopped by user")
#     except Exception as e:
#         logger.critical(f"💥 Critical error: {str(e)}")
##############################################################################################
##############################################################################################
##############################################################################################
# import os
# import logging
# import uuid
# import cv2
# import numpy as np
# from PIL import Image, ImageSequence
# from nudenet import NudeDetector
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from telegram import Update
# from telegram.ext import (
#     ApplicationBuilder,
#     MessageHandler,
#     ContextTypes,
#     filters,
#     CommandHandler
# )

# # Enhanced logging configuration
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO,
#     handlers=[
#         logging.FileHandler("nsfw_bot.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Configuration
# DOWNLOAD_DIR = "downloads"
# NSFW_THRESHOLD = 0.65  # Adjusted for better sensitivity
# FRAME_ANALYSIS_COUNT = 5  # Increased frame analysis
# MIN_DETECTION_CONFIDENCE = 0.15  # Lower threshold for more detections

# # Model configuration
# MODEL_SELECTION = "both"  # Force using both models
# TF_MODEL_PATH = "nsfw.299x299.h5"
# TF_INPUT_SIZE = (299, 299)

# # Enhanced NSFW classes with 100+ categories
# NSFW_CLASSES = {
#     # Explicit nudity (25 classes)
#     'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
#     'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED',
#     'PUBIC_HAIR_EXPOSED', 'AREOLA_EXPOSED', 'NIPPLE_EXPOSED',
#     'GENITALIA_PARTIAL_EXPOSURE', 'UNDERWEAR_VISIBLE',
#     'CLOTHING_SEE_THROUGH', 'WET_CLOTHING', 'TIGHT_CLOTHING',
#     'CLOTHING_LIFTING', 'TOPLESS', 'BOTTOMLESS', 'NAKED',
#     'EXPOSED_CHEST', 'EXPOSED_THIGHS', 'EXPOSED_BACK',
#     'EXPOSED_SHOULDER', 'EXPOSED_STOMACH', 'EXPOSED_LEGS',
#     'EXPOSED_ARMPIT', 'EXPOSED_FEET',

#     # Sexual activities (30 classes)
#     'SEXUAL_ACTIVITY', 'MASTURBATION', 'ORAL_SEX', 'ANAL_SEX',
#     'SEX_TOYS', 'PORNOGRAPHIC_POSES', 'SUGGESTIVE_POSES',
#     'SEXUAL_GESTURES', 'FOREPLAY', 'GROUP_SEX', 'BDSM',
#     'FETISH_ACTIVITY', 'DOMINATION', 'SUBMISSION', 'BONDAGE',
#     'SPANKING', 'ROLE_PLAY', 'STRAPON_USE', 'HUMILIATION',
#     'EXHIBITIONISM', 'VOYEURISM', 'FROTTAGE', 'AUTOEROTIC_ASPHYXIATION',
#     'CYBERSEX', 'TELEPHONE_SEX', 'SEXTING', 'SEXUAL_SIMULATION',
#     'SEXUAL_INTERCOURSE', 'GRINDING', 'TRIBADISM',

#     # Kissing & intimate contact (25 classes)
#     'KISSING', 'FRENCH_KISS', 'DEEP_KISS', 'TONGUE_KISS',
#     'PASSIONATE_KISS', 'MOUTH_TO_MOUTH', 'NECK_KISS',
#     'EAR_KISS', 'FACE_KISS', 'HICKY', 'LOVE_BITE',
#     'INTIMATE_EMBRACE', 'FACE_TO_FACE_CLOSE', 'BODY_PRESS',
#     'INTIMATE_TOUCH', 'HAND_HOLDING', 'CARESSING', 'LAP_SITTING',
#     'INTIMATE_WHISPERING', 'EYE_CONTACT_INTIMATE', 'LIP_BITING',
#     'SEDUCTIVE_GAZE', 'INTIMATE_CUDDLING', 'FACE_STROKING',
#     'NECK_NUZZLING',

#     # Fetish content (20 classes)
#     'FEET_FETISH', 'UNDERWEAR_FETISH', 'UNIFORM_FETISH',
#     'LEATHER_FETISH', 'LATEX_FETISH', 'BODY_PART_FETISH',
#     'FOOD_FETISH', 'WATERSPORTS', 'SCAT_FETISH', 'SHOE_FETISH',
#     'HOOD_FETISH', 'MASK_FETISH', 'GAG_FETISH', 'CORSET_FETISH',
#     'HIGH_HEEL_FETISH', 'STOCKING_FETISH', 'SWIMSUIT_FETISH',
#     'LINGERIE_FETISH', 'DIAPER_FETISH', 'AGE_PLAY',

#     # Other sensitive content (15 classes)
#     'DRUG_USE', 'ALCOHOL_CONSUMPTION', 'SMOKING',
#     'VIOLENCE', 'BLOOD', 'GORE', 'WEAPONS',
#     'HATE_SYMBOLS', 'DISTURBING_CONTENT', 'SELF_HARM',
#     'BULLYING', 'HARASSMENT', 'EXPLOITATION',
#     'NON_CONSENSUAL_ACT', 'INCEST_THEME'
# }

# # Initialize models
# logger.info("⚙️ Loading advanced detection models...")
# nude_detector = NudeDetector()

# # Load TensorFlow model with improved error handling
# tf_model = None
# try:
#     tf_model = load_model(
#         TF_MODEL_PATH,
#         custom_objects={'KerasLayer': tf.keras.layers.Layer},
#         compile=False
#     )
#     logger.info(f"✅ TensorFlow model ({TF_MODEL_PATH}) loaded successfully!")
    
#     # Model warm-up with proper error handling
#     try:
#         dummy_input = np.random.rand(1, *TF_INPUT_SIZE, 3).astype(np.float32)
#         prediction = tf_model.predict(dummy_input)
#         logger.info(f"🔍 Model warm-up successful. Output shape: {prediction.shape}")
#     except Exception as e:
#         logger.error(f"❌ Model warm-up failed: {str(e)}")

# except Exception as e:
#     logger.error(f"❌ Critical error loading TensorFlow model: {str(e)}")
#     if "CUDA" in str(e):
#         logger.warning("⚠️ Trying to load with CPU only...")
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         try:
#             tf_model = load_model(TF_MODEL_PATH, compile=False)
#         except Exception as e:
#             logger.error(f"❌ CPU load failed: {str(e)}")
#     if tf_model is None:
#         logger.warning("⚠️ TensorFlow model disabled")
        
# logger.info(f"✅ Models initialized! Active model: {MODEL_SELECTION}")

# def preprocess_image_for_tf(image_path: str, target_size=TF_INPUT_SIZE) -> np.ndarray:
#     """Enhanced image preprocessing with error recovery"""
#     try:
#         img = tf.keras.preprocessing.image.load_img(
#             image_path,
#             target_size=target_size,
#             color_mode='rgb',
#             interpolation='bilinear'
#         )
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0)
#         return img_array / 255.0
#     except Exception as e:
#         logger.error(f"❌ Preprocessing failed: {str(e)}")
#         return np.zeros((1, *target_size, 3), dtype=np.float32)

# def detect_nsfw_tensorflow(image_path: str) -> tuple[bool, float]:
#     """Improved TensorFlow detection with multiple output support"""
#     if tf_model is None:
#         return False, 0.0
    
#     try:
#         img_array = preprocess_image_for_tf(image_path)
#         predictions = tf_model.predict(img_array)
        
#         # Handle different model output formats
#         if predictions.shape[-1] == 5:  # Common NSFW model format
#             # Classes: drawings, hentai, neutral, porn, sexy
#             nsfw_score = predictions[0][1] + predictions[0][3] + predictions[0][4]
#         elif predictions.shape[-1] == 2:  # Binary classification
#             nsfw_score = predictions[0][1]
#         else:
#             nsfw_score = predictions[0][0]
            
#         confidence = float(nsfw_score)
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 TensorFlow detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_nudenet(image_path: str) -> tuple[bool, float]:
#     """Enhanced NudeNet detection with class weighting"""
#     try:
#         detections = nude_detector.detect(image_path)
#         valid_detections = [
#             det for det in detections 
#             if det['class'] in NSFW_CLASSES 
#             and det['score'] >= MIN_DETECTION_CONFIDENCE
#         ]
        
#         if not valid_detections:
#             return False, 0.0
            
#         # Class weights (1.0 = most severe)
#         weights = {
#             # Explicit nudity
#             'FEMALE_GENITALIA_EXPOSED': 1.0,
#             'MALE_GENITALIA_EXPOSED': 1.0,
#             'ANUS_EXPOSED': 0.95,
            
#             # Sexual acts
#             'SEXUAL_ACTIVITY': 0.9,
#             'ORAL_SEX': 0.9,
#             'MASTURBATION': 0.85,
            
#             # Kissing/intimacy
#             'FRENCH_KISS': 0.7,
#             'PASSIONATE_KISS': 0.65,
#             'INTIMATE_EMBRACE': 0.6,
            
#             # Default weight
#             '_default': 0.5
#         }
        
#         weighted = [d['score'] * weights.get(d['class'], weights['_default']) for d in valid_detections]
#         confidence = max(weighted)
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 NudeNet detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_combined(image_path: str) -> tuple[bool, float]:
#     """Intelligent confidence combination"""
#     try:
#         n_result, n_conf = detect_nsfw_nudenet(image_path)
#         t_result, t_conf = detect_nsfw_tensorflow(image_path)
        
#         # If either model is very confident, prioritize it
#         if max(n_conf, t_conf) > 0.85:
#             confidence = max(n_conf, t_conf)
#         else:
#             # Weighted average favoring NudeNet for specific detections
#             confidence = (n_conf * 0.7 + t_conf * 0.3)
        
#         # If both models agree, boost confidence
#         if n_result and t_result:
#             confidence = min(1.0, confidence * 1.2)
            
#         logger.info(f"🔍 Combined confidence: NudeNet={n_conf:.2f}, TF={t_conf:.2f}, Final={confidence:.2f}")
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 Combined detection failed: {str(e)}")
#         return False, 0.0

# # Rest of the code (analyze_frames, handlers, main) remains the same as previous version
# # [Include the rest of the code from previous implementation here]
# def detect_nsfw(image_path: str) -> tuple[bool, float]:
#     """Route to appropriate detection method"""
#     if MODEL_SELECTION == "nudenet":
#         return detect_nsfw_nudenet(image_path)
#     elif MODEL_SELECTION == "tensorflow":
#         return detect_nsfw_tensorflow(image_path)
#     else:
#         return detect_nsfw_combined(image_path)


# async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle all incoming media messages"""
#     try:
#         message = update.effective_message
#         if not message:
#             return

#         # Get file based on message type
#         file = None
#         is_video = False
#         is_animated = False
#         file_ext = '.jpg'
        
#         if message.photo:
#             file = await message.photo[-1].get_file()
#             media_type = "photo"
#         elif message.document:
#             if message.document.mime_type.startswith('image/'):
#                 file = await message.document.get_file()
#                 media_type = "image document"
#                 file_ext = os.path.splitext(message.document.file_name)[1] or '.jpg'
#             elif message.document.mime_type.startswith('video/'):
#                 file = await message.document.get_file()
#                 media_type = "video document"
#                 file_ext = os.path.splitext(message.document.file_name)[1] or '.mp4'
#                 is_video = True
#         elif message.sticker:
#             if message.sticker.is_animated:
#                 media_type = "animated sticker"
#                 is_animated = True
#                 # Convert animated sticker to video for analysis
#                 file = await message.sticker.get_file()
#                 file_ext = '.webm'
#             else:
#                 file = await message.sticker.get_file()
#                 media_type = "static sticker"
#                 file_ext = '.png'
#         elif message.animation:
#             file = await message.animation.get_file()
#             media_type = "animation"
#             file_ext = '.mp4'
#             is_video = True

#         if not file and not is_animated:
#             logger.info("No supported media found")
#             return

#         # Download the file
#         file_id = str(uuid.uuid4())
#         file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}{file_ext}")
        
#         if not is_animated:  # Skip download for animated stickers (handled differently)
#             await file.download_to_drive(custom_path=file_path)
#             logger.info(f"Downloaded {media_type} to {file_path}")

#         # Send processing message
#         processing_msg = await message.reply_text(
#             f"🔍 Analyzing {media_type} with {MODEL_SELECTION} model..."
#             if not is_video else
#             f"🎬 Analyzing {FRAME_ANALYSIS_COUNT} frames from {media_type}..."
#         )

#         try:
#             # Analyze the content
#             is_nsfw = False
#             confidence = 0.0

#             if is_animated:
#                 # Special handling for animated stickers
#                 await processing_msg.edit_text("⚠️ Animated stickers can't be fully analyzed")
#                 return
#             elif is_video or file_ext in ['.gif', '.webm', '.mp4']:
#                 # Video/GIF analysis
#                 vid = cv2.VideoCapture(file_path)
#                 total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#                 frames_to_analyze = min(FRAME_ANALYSIS_COUNT, total_frames)
                
#                 for i in range(frames_to_analyze):
#                     frame_pos = i * (total_frames // frames_to_analyze)
#                     vid.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
#                     success, frame = vid.read()
#                     if success:
#                         frame_path = f"{file_path}_frame_{i}.jpg"
#                         cv2.imwrite(frame_path, frame)
#                         _, frame_confidence = detect_nsfw(frame_path)
#                         confidence = max(confidence, frame_confidence)
#                         os.remove(frame_path)
                
#                 vid.release()
#             else:
#                 # Image analysis
#                 is_nsfw, confidence = detect_nsfw(file_path)

#             logger.info(f"Analysis result - Confidence: {confidence:.2%}")

#             # Handle results
#             if confidence > NSFW_THRESHOLD:
#                 await message.delete()
#                 logger.info(f"Deleted NSFW content (Confidence: {confidence:.2%})")
#                 await processing_msg.edit_text(
#                     f"⚠️ Removed explicit content\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}"
#                 )
#             else:
#                 logger.info(f"Approved content (Confidence: {confidence:.2%})")
#                 await processing_msg.edit_text(
#                     f"✅ Content approved\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}"
#                 )

#         except Exception as e:
#             logger.error(f"Error during analysis: {str(e)}")
#             await processing_msg.edit_text("❌ Error during analysis")
#             raise

#         finally:
#             # Cleanup
#             if os.path.exists(file_path):
#                 os.remove(file_path)

#     except Exception as e:
#         logger.error(f"Critical error in media handler: {str(e)}")


# async def change_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle model selection command"""
#     global MODEL_SELECTION
    
#     if not context.args:
#         await update.message.reply_text(
#             f"Current model: {MODEL_SELECTION}\n"
#             "Available options: /model nudenet, /model tensorflow, /model both"
#         )
#         return
    
#     selection = context.args[0].lower()
#     if selection in ("nudenet", "tensorflow", "both"):
#         if selection == "tensorflow" and tf_model is None:
#             await update.message.reply_text("TensorFlow model not available. Using NudeNet.")
#             return
            
#         MODEL_SELECTION = selection
#         await update.message.reply_text(f"Model selection changed to: {MODEL_SELECTION}")
#         logger.info(f"Model changed to: {MODEL_SELECTION}")
#     else:
#         await update.message.reply_text("Invalid model. Choose from: nudenet, tensorflow, both")

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle /start command"""
#     await update.message.reply_text(
#         "🚀 NSFW Detection Bot\n\n"
#         "I analyze photos, videos, GIFs, and stickers for explicit content.\n"
#         "Current model: {MODEL_SELECTION}\n\n"
#         "Use /model to change detection model"
#     )

# def main():
#     """Start the bot"""
#     try:
#         bot_token = "7803429144:AAFXixTN0-Gb2eX1GE2KJnlTHdvfJBLrlnM"  # Replace with your token
        
#         app = ApplicationBuilder().token(bot_token).build()

#         # Add handlers
#         app.add_handler(CommandHandler("start", start))
#         app.add_handler(CommandHandler("model", change_model))
        
#         # Media handler
#         media_filter = (
#             filters.PHOTO |
#             filters.Document.IMAGE |
#             filters.Document.VIDEO |
#             filters.Sticker.ALL |
#             filters.ANIMATION
#         )
#         app.add_handler(MessageHandler(media_filter, handle_media))

#         logger.info("🤖 Bot is running...")
#         app.run_polling()

#     except Exception as e:
#         logger.critical(f"Failed to start bot: {str(e)}")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         logger.info("Bot stopped by user")
#     except Exception as e:
#         logger.critical(f"Critical error: {str(e)}")

##########################################################################
##########################################################################
# Super Code
##########################################################################
##########################################################################

# import os
# import logging
# import uuid
# import cv2
# import numpy as np
# from PIL import Image, ImageSequence
# from nudenet import NudeDetector
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from telegram import Update
# from telegram.ext import (
#     ApplicationBuilder,
#     MessageHandler,
#     ContextTypes,
#     filters,
#     CommandHandler
# )

# # Enhanced logging configuration
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO,
#     handlers=[
#         logging.FileHandler("nsfw_bot.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Configuration
# DOWNLOAD_DIR = "downloads"
# os.makedirs(DOWNLOAD_DIR, exist_ok=True)
# NSFW_THRESHOLD = 0.25  # Adjusted for better sensitivity
# FRAME_ANALYSIS_COUNT = 2  # Increased frame analysis
# MIN_DETECTION_CONFIDENCE = 0.15  # Lower threshold for more detections

# # Model configuration
# MODEL_SELECTION = "both"  # Using both models for maximum accuracy
# TF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'nsfw.299x299.h5')
# TF_INPUT_SIZE = (299, 299)

# # Enhanced NSFW classes with 100+ categories
# NSFW_CLASSES = {
#     # Explicit nudity (25 classes)
#     'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
#     'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED',
#     'PUBIC_HAIR_EXPOSED', 'AREOLA_EXPOSED', 'NIPPLE_EXPOSED',
#     'GENITALIA_PARTIAL_EXPOSURE', 'UNDERWEAR_VISIBLE',
#     'CLOTHING_SEE_THROUGH', 'WET_CLOTHING', 'TIGHT_CLOTHING',
#     'CLOTHING_LIFTING', 'TOPLESS', 'BOTTOMLESS', 'NAKED',
#     'EXPOSED_CHEST', 'EXPOSED_THIGHS', 'EXPOSED_BACK',
#     'EXPOSED_SHOULDER', 'EXPOSED_STOMACH', 'EXPOSED_LEGS',
#     'EXPOSED_ARMPIT', 'EXPOSED_FEET',

#     # Sexual activities (30 classes)
#     'SEXUAL_ACTIVITY', 'MASTURBATION', 'ORAL_SEX', 'ANAL_SEX',
#     'SEX_TOYS', 'PORNOGRAPHIC_POSES', 'SUGGESTIVE_POSES',
#     'SEXUAL_GESTURES', 'FOREPLAY', 'GROUP_SEX', 'BDSM',
#     'FETISH_ACTIVITY', 'DOMINATION', 'SUBMISSION', 'BONDAGE',
#     'SPANKING', 'ROLE_PLAY', 'STRAPON_USE', 'HUMILIATION',
#     'EXHIBITIONISM', 'VOYEURISM', 'FROTTAGE', 'AUTOEROTIC_ASPHYXIATION',
#     'CYBERSEX', 'TELEPHONE_SEX', 'SEXTING', 'SEXUAL_SIMULATION',
#     'SEXUAL_INTERCOURSE', 'GRINDING', 'TRIBADISM',

#     # Kissing & intimate contact (25 classes)
#     'KISSING', 'FRENCH_KISS', 'DEEP_KISS', 'TONGUE_KISS',
#     'PASSIONATE_KISS', 'MOUTH_TO_MOUTH', 'NECK_KISS',
#     'EAR_KISS', 'FACE_KISS', 'HICKY', 'LOVE_BITE',
#     'INTIMATE_EMBRACE', 'FACE_TO_FACE_CLOSE', 'BODY_PRESS',
#     'INTIMATE_TOUCH', 'HAND_HOLDING', 'CARESSING', 'LAP_SITTING',
#     'INTIMATE_WHISPERING', 'EYE_CONTACT_INTIMATE', 'LIP_BITING',
#     'SEDUCTIVE_GAZE', 'INTIMATE_CUDDLING', 'FACE_STROKING',
#     'NECK_NUZZLING',

#     # Fetish content (20 classes)
#     'FEET_FETISH', 'UNDERWEAR_FETISH', 'UNIFORM_FETISH',
#     'LEATHER_FETISH', 'LATEX_FETISH', 'BODY_PART_FETISH',
#     'FOOD_FETISH', 'WATERSPORTS', 'SCAT_FETISH', 'SHOE_FETISH',
#     'HOOD_FETISH', 'MASK_FETISH', 'GAG_FETISH', 'CORSET_FETISH',
#     'HIGH_HEEL_FETISH', 'STOCKING_FETISH', 'SWIMSUIT_FETISH',
#     'LINGERIE_FETISH', 'DIAPER_FETISH', 'AGE_PLAY',

#     # Other sensitive content (15 classes)
#     'DRUG_USE', 'ALCOHOL_CONSUMPTION', 'SMOKING',
#     'VIOLENCE', 'BLOOD', 'GORE', 'WEAPONS',
#     'HATE_SYMBOLS', 'DISTURBING_CONTENT', 'SELF_HARM',
#     'BULLYING', 'HARASSMENT', 'EXPLOITATION',
#     'NON_CONSENSUAL_ACT', 'INCEST_THEME'
# }

# # Initialize models
# logger.info("⚙️ Loading advanced detection models...")
# nude_detector = NudeDetector()

# # Load TensorFlow model with improved error handling
# tf_model = None
# try:
#     tf_model = load_model(
#         TF_MODEL_PATH,
#         custom_objects={'KerasLayer': tf.keras.layers.Layer},
#         compile=False
#     )
#     logger.info(f"✅ TensorFlow model ({TF_MODEL_PATH}) loaded successfully!")
    
#     # Model warm-up with proper error handling
#     try:
#         dummy_input = np.random.rand(1, *TF_INPUT_SIZE, 3).astype(np.float32)
#         prediction = tf_model.predict(dummy_input)
#         logger.info(f"🔍 Model warm-up successful. Output shape: {prediction.shape}")
#     except Exception as e:
#         logger.error(f"❌ Model warm-up failed: {str(e)}")

# except Exception as e:
#     logger.error(f"❌ Critical error loading TensorFlow model: {str(e)}")
#     if "CUDA" in str(e):
#         logger.warning("⚠️ Trying to load with CPU only...")
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         try:
#             tf_model = load_model(TF_MODEL_PATH, compile=False)
#         except Exception as e:
#             logger.error(f"❌ CPU load failed: {str(e)}")
#     if tf_model is None:
#         logger.warning("⚠️ TensorFlow model disabled")
        
# logger.info(f"✅ Models initialized! Active model: {MODEL_SELECTION}")

# def preprocess_image_for_tf(image_path: str, target_size=TF_INPUT_SIZE) -> np.ndarray:
#     """Enhanced image preprocessing with error recovery"""
#     try:
#         img = tf.keras.preprocessing.image.load_img(
#             image_path,
#             target_size=target_size,
#             color_mode='rgb',
#             interpolation='bilinear'
#         )
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0)
#         return img_array / 255.0
#     except Exception as e:
#         logger.error(f"❌ Preprocessing failed: {str(e)}")
#         return np.zeros((1, *target_size, 3), dtype=np.float32)

# def detect_nsfw_tensorflow(image_path: str) -> tuple[bool, float]:
#     """Improved TensorFlow detection with multiple output support"""
#     if tf_model is None:
#         return False, 0.0
    
#     try:
#         img_array = preprocess_image_for_tf(image_path)
#         predictions = tf_model.predict(img_array)
        
#         # Handle different model output formats
#         if predictions.shape[-1] == 5:  # Common NSFW model format
#             # Classes: drawings, hentai, neutral, porn, sexy
#             nsfw_score = predictions[0][1] + predictions[0][3] + predictions[0][4]
#         elif predictions.shape[-1] == 2:  # Binary classification
#             nsfw_score = predictions[0][1]
#         else:
#             nsfw_score = predictions[0][0]
            
#         confidence = float(nsfw_score)
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 TensorFlow detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_nudenet(image_path: str) -> tuple[bool, float]:
#     """Enhanced NudeNet detection with class weighting"""
#     try:
#         detections = nude_detector.detect(image_path)
#         valid_detections = [
#             det for det in detections 
#             if det['class'] in NSFW_CLASSES 
#             and det['score'] >= MIN_DETECTION_CONFIDENCE
#         ]
        
#         if not valid_detections:
#             return False, 0.0
            
#         # Class weights (1.0 = most severe)
#         weights = {
#             # Explicit nudity
#             'FEMALE_GENITALIA_EXPOSED': 1.0,
#             'MALE_GENITALIA_EXPOSED': 1.0,
#             'ANUS_EXPOSED': 0.95,
            
#             # Sexual acts
#             'SEXUAL_ACTIVITY': 0.9,
#             'ORAL_SEX': 0.9,
#             'MASTURBATION': 0.85,
            
#             # Kissing/intimacy
#             'KISSING': 0.5,
#             'FRENCH_KISS': 0.7,
#             'PASSIONATE_KISS': 0.65,
#             'INTIMATE_EMBRACE': 0.6,
            
#             # Default weight
#             '_default': 0.5
#         }


        
#         weighted = [d['score'] * weights.get(d['class'], weights['_default']) for d in valid_detections]
#         confidence = max(weighted)
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 NudeNet detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_combined(image_path: str) -> tuple[bool, float]:
#     """Intelligent confidence combination"""
#     try:
#         n_result, n_conf = detect_nsfw_nudenet(image_path)
#         t_result, t_conf = detect_nsfw_tensorflow(image_path)
        
#         # If either model is very confident, prioritize it
#         if max(n_conf, t_conf) > 0.85:
#             confidence = max(n_conf, t_conf)
#         else:
#             # Weighted average favoring NudeNet for specific detections
#             confidence = (n_conf * 0.7 + t_conf * 0.3)
        
#         # If both models agree, boost confidence
#         if n_result and t_result:
#             confidence = min(1.0, confidence * 1.2)
            
#         logger.info(f"🔍 Combined confidence: NudeNet={n_conf:.2f}, TF={t_conf:.2f}, Final={confidence:.2f}")
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 Combined detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw(image_path: str) -> tuple[bool, float]:
#     """Route to appropriate detection method"""
#     if MODEL_SELECTION == "nudenet":
#         return detect_nsfw_nudenet(image_path)
#     elif MODEL_SELECTION == "tensorflow":
#         return detect_nsfw_tensorflow(image_path)
#     else:
#         return detect_nsfw_combined(image_path)

# def analyze_frames(file_path: str) -> float:
#     """Analyze multiple frames for video/GIF content"""
#     max_confidence = 0.0
#     try:
#         if file_path.endswith(('.webm', '.mp4')):
#             vid = cv2.VideoCapture(file_path)
#             total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_step = max(1, total_frames // FRAME_ANALYSIS_COUNT)
            
#             for i in range(FRAME_ANALYSIS_COUNT):
#                 vid.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
#                 success, frame = vid.read()
#                 if success:
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     cv2.imwrite(frame_path, frame)
#                     _, confidence = detect_nsfw(frame_path)
#                     max_confidence = max(max_confidence, confidence)
#                     os.remove(frame_path)
#             vid.release()
            
#         elif file_path.endswith('.gif'):
#             with Image.open(file_path) as img:
#                 total_frames = img.n_frames
#                 frame_step = max(1, total_frames // FRAME_ANALYSIS_COUNT)
                
#                 for i in range(FRAME_ANALYSIS_COUNT):
#                     img.seek(i * frame_step)
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     img.convert('RGB').save(frame_path)
#                     _, confidence = detect_nsfw(frame_path)
#                     max_confidence = max(max_confidence, confidence)
#                     os.remove(frame_path)
                    
#         return max_confidence
#     except Exception as e:
#         logger.error(f"🎞 Frame analysis failed: {str(e)}")
#         return 0.0

# async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle model selection command"""
#     global MODEL_SELECTION
#     if not context.args:
#         await update.message.reply_text(
#             f"Current model: {MODEL_SELECTION}\n"
#             "Available options: /model nudenet, /model tensorflow, /model both"
#         )
#         return
    
#     selection = context.args[0].lower()
#     if selection in ("nudenet", "tensorflow", "both"):
#         if selection == "tensorflow" and tf_model is None:
#             await update.message.reply_text("TensorFlow model not available. Using NudeNet.")
#             return
            
#         MODEL_SELECTION = selection
#         await update.message.reply_text(f"Model selection changed to: {MODEL_SELECTION}")
#         logger.info(f"Model selection changed to: {MODEL_SELECTION}")
#     else:
#         await update.message.reply_text("Invalid model. Choose from: nudenet, tensorflow, both")

# async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Enhanced media handler with better feedback"""
#     try:
#         message = update.effective_message
#         user = message.from_user
#         chat = message.chat
        
#         logger.info(f"📩 New media from {user.full_name} ({user.id}) in {chat.title if chat.type != 'private' else 'private chat'}")
        
#         file = None
#         is_video = False
#         media_type = "unknown"

#         # Enhanced media type handling
#         if message.photo:
#             file = await message.photo[-1].get_file()
#             media_type = "photo"
#         elif message.sticker:
#             if message.sticker.is_animated:
#                 logger.info("⚠️ Analyzing animated sticker frames")
#                 file = await message.sticker.get_file()
#                 media_type = "animated_sticker"
#                 is_video = True
#             else:
#                 file = await message.sticker.get_file()
#                 media_type = "sticker"
#         elif message.animation:
#             file = await message.animation.get_file()
#             media_type = "animation"
#             is_video = True
#         elif message.document:
#             if message.document.mime_type.startswith('image/'):
#                 file = await message.document.get_file()
#                 media_type = "image_document"
#             elif message.document.mime_type.startswith('video/'):
#                 file = await message.document.get_file()
#                 media_type = "video_document"
#                 is_video = True
#             else:
#                 logger.info(f"⚠️ Unsupported document type: {message.document.mime_type}")
#                 return

#         if not file:
#             logger.warning("⚠️ No file found in message")
#             return

#         # Download with proper extension and hash
#         file_ext = os.path.splitext(file.file_path)[1] if file.file_path else '.jpg'
#         file_id = str(uuid.uuid4())
#         file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}{file_ext}")
        
#         logger.info(f"⬇️ Downloading {media_type} (size: {file.file_size or 'unknown'} bytes)")
#         await file.download_to_drive(custom_path=file_path)

#         # Enhanced analysis with progress feedback
#         analysis_msg = await message.reply_text(
#             f"🔍 Analyzing {media_type} with {MODEL_SELECTION} model..."
#             if not is_video else
#             f"🎬 Analyzing {FRAME_ANALYSIS_COUNT} frames from {media_type}..."
#         )

#         try:
#             # Multi-frame analysis for videos/GIFs/animated stickers
#             if is_video or file_ext in ['.gif', '.webm', '.mp4']:
#                 confidence = analyze_frames(file_path)
#                 is_nsfw_content = confidence > NSFW_THRESHOLD
#             else:
#                 # Single image processing
#                 is_nsfw_content, confidence = detect_nsfw(file_path)

#             # Enhanced result handling
#             if is_nsfw_content or confidence > NSFW_THRESHOLD:
#                 await message.delete()
#                 logger.info(f"🚫 Deleted NSFW content (Confidence: {confidence:.2%})")
                
#                 warning_msg = (
#                     f"⚠️ Removed explicit content\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}\n"
#                     f"Media type: {media_type}\n\n"
#                     f"*This action was performed automatically*"
#                 )
                
#                 await chat.send_message(
#                     warning_msg,
#                     reply_to_message_id=message.message_id if chat.type != 'private' else None
#                 )
#             else:
#                 logger.info(f"✅ Safe content (Confidence: {confidence:.2%})")
#                 await analysis_msg.edit_text(
#                     f"✅ Content approved\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}"
#                 )

#         except Exception as e:
#             logger.error(f"❌ Analysis failed: {str(e)}")
#             await analysis_msg.edit_text("❌ Analysis failed. Please try again.")
#             raise

#         finally:
#             # Cleanup
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#             else:
#                 logger.warning(f"⚠️ File not found during cleanup: {file_path}")

#     except Exception as e:
#         logger.error(f"🔥 Critical error in media handler: {str(e)}")
#         if 'analysis_msg' in locals():
#             await analysis_msg.edit_text("❌ An error occurred during processing")

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Send a message when the command /start is issued."""
#     await update.message.reply_text(
#         "🚀 NSFW Detection Bot is running!\n"
#         "I automatically analyze photos, videos, GIFs and stickers for explicit content.\n\n"
#         "Commands:\n"
#         "/model - Change detection model\n"
#         "/help - Show help information"
#     )

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Send a message when the command /help is issued."""
#     await update.message.reply_text(
#         "ℹ️ NSFW Detection Bot Help\n\n"
#         "I analyze these media types:\n"
#         "- Photos (sent as photo or document)\n"
#         "- Videos (up to 20MB)\n"
#         "- GIFs\n"
#         "- Static and animated stickers\n\n"
#         "Commands:\n"
#         "/model [nudenet|tensorflow|both] - Change detection model\n"
#         "/help - Show this message\n\n"
#         "Current settings:\n"
#         f"Detection threshold: {NSFW_THRESHOLD*100:.0f}%\n"
#         f"Active model: {MODEL_SELECTION}"
#     )

# async def post_init(application):
#     """Run after bot is initialized"""
#     await application.bot.set_my_commands([
#         ("model", "Change detection model"),
#         ("help", "Show help information"),
#     ])
#     logger.info("🤖 Bot initialization complete!")

# async def post_stop(application):
#     """Run before bot is stopped"""
#     logger.info("🛑 Bot shutdown complete!")

# def main():
#     """Start the bot."""
#     try:
#         bot_token = "7803429144:AAFXixTN0-Gb2eX1GE2KJnlTHdvfJBLrlnM"  # Replace with your actual token
        
#         # Create application with error handling
#         app = ApplicationBuilder() \
#             .token(bot_token) \
#             .post_init(post_init) \
#             .post_stop(post_stop) \
#             .build()

#         # Add command handlers
#         app.add_handler(CommandHandler("start", start))
#         app.add_handler(CommandHandler("help", help_command))
#         app.add_handler(CommandHandler("model", handle_model_selection))

#         # Media handler - captures all supported media types
#         media_filter = (
#             filters.PHOTO |
#             filters.Document.IMAGE |
#             filters.Document.VIDEO |
#             filters.Sticker.ALL |
#             filters.ANIMATION
#         )
#         app.add_handler(MessageHandler(media_filter, handle_media))

#         logger.info("🤖 Starting Advanced NSFW Detection Bot...")
#         logger.info(f"🔧 Model selection: {MODEL_SELECTION}")
#         logger.info(f"📁 Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
#         logger.info("📸 Monitoring: Photos | Videos | GIFs | Stickers")
        
#         # Run with better error handling
#         app.run_polling(
#             poll_interval=1.0,
#             timeout=30,
#             drop_pending_updates=True
#         )

#     except Exception as e:
#         logger.critical(f"💥 Fatal error during startup: {str(e)}")
#         raise

# if __name__ == "__main__":
#     try:
#         import asyncio
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         logger.info("🛑 Bot stopped by user")
#     except Exception as e:
#         logger.critical(f"💥 Critical error: {str(e)}")
##########################################################################
##########################################################################
# Super Code
##########################################################################
##########################################################################

# import os
# import logging
# import uuid
# import cv2
# import numpy as np
# from PIL import Image, ImageSequence
# from nudenet import NudeDetector
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from telegram import Update
# from telegram.ext import (
#     ApplicationBuilder,
#     MessageHandler,
#     ContextTypes,
#     filters,
#     CommandHandler
# )

# # Enhanced logging configuration
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO,
#     handlers=[
#         logging.FileHandler("nsfw_bot.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Configuration
# DOWNLOAD_DIR = "downloads"
# os.makedirs(DOWNLOAD_DIR, exist_ok=True)
# NSFW_THRESHOLD = 0.85  # Higher threshold to reduce false positives
# SAFE_CONTENT_THRESHOLD = 0.15  # Content below this is definitely safe
# FRAME_ANALYSIS_COUNT = 3  # Balanced frame analysis
# MIN_DETECTION_CONFIDENCE = 0.35  # Higher threshold to reduce false detections

# # Model configuration
# MODEL_SELECTION = "both"  # Using both models for maximum accuracy
# TF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'nsfw.299x299.h5')
# TF_INPUT_SIZE = (299, 299)

# # Focused NSFW classes - only clear cases of nudity/sexual content
# NSFW_CLASSES = {
#     # Explicit nudity
#     'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
#     'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED',
#     'PUBIC_HAIR_EXPOSED', 'AREOLA_EXPOSED', 'NIPPLE_EXPOSED',
    
#     # Sexual activities
#     'SEXUAL_ACTIVITY', 'MASTURBATION', 'ORAL_SEX', 'ANAL_SEX',
#     'SEX_TOYS', 'PORNOGRAPHIC_POSES', 'GROUP_SEX', 'BDSM'
# }

# # Safe content indicators (if these are detected, content is considered safe)
# SAFE_CONTENT_INDICATORS = {
#     'CHILD', 'BABY', 'KID', 'FAMILY', 'DANCE', 'SPORTS',
#     'SWIMMING', 'BEACH', 'MEDICAL', 'ART', 'HISTORICAL',
#     'EDUCATIONAL', 'CLOTHED', 'OUTDOOR', 'GROUP', 'PLAYING',
#     'COSTUME', 'UNIFORM', 'SCHOOL', 'GAME', 'PET', 'ANIMAL'
# }

# # Initialize models
# logger.info("⚙️ Loading advanced detection models...")
# nude_detector = NudeDetector()

# # Load TensorFlow model with improved error handling
# tf_model = None
# try:
#     tf_model = load_model(
#         TF_MODEL_PATH,
#         custom_objects={'KerasLayer': tf.keras.layers.Layer},
#         compile=False
#     )
#     logger.info(f"✅ TensorFlow model ({TF_MODEL_PATH}) loaded successfully!")
    
#     # Model warm-up with proper error handling
#     try:
#         dummy_input = np.random.rand(1, *TF_INPUT_SIZE, 3).astype(np.float32)
#         prediction = tf_model.predict(dummy_input)
#         logger.info(f"🔍 Model warm-up successful. Output shape: {prediction.shape}")
#     except Exception as e:
#         logger.error(f"❌ Model warm-up failed: {str(e)}")

# except Exception as e:
#     logger.error(f"❌ Critical error loading TensorFlow model: {str(e)}")
#     if "CUDA" in str(e):
#         logger.warning("⚠️ Trying to load with CPU only...")
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         try:
#             tf_model = load_model(TF_MODEL_PATH, compile=False)
#         except Exception as e:
#             logger.error(f"❌ CPU load failed: {str(e)}")
#     if tf_model is None:
#         logger.warning("⚠️ TensorFlow model disabled")
        
# logger.info(f"✅ Models initialized! Active model: {MODEL_SELECTION}")

# def preprocess_image_for_tf(image_path: str, target_size=TF_INPUT_SIZE) -> np.ndarray:
#     """Enhanced image preprocessing with error recovery"""
#     try:
#         img = tf.keras.preprocessing.image.load_img(
#             image_path,
#             target_size=target_size,
#             color_mode='rgb',
#             interpolation='bilinear'
#         )
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0)
#         return img_array / 255.0
#     except Exception as e:
#         logger.error(f"❌ Preprocessing failed: {str(e)}")
#         return np.zeros((1, *target_size, 3), dtype=np.float32)

# def detect_nsfw_tensorflow(image_path: str) -> tuple[bool, float]:
#     """Improved TensorFlow detection with safe content checks"""
#     if tf_model is None:
#         return False, 0.0
    
#     try:
#         img_array = preprocess_image_for_tf(image_path)
#         predictions = tf_model.predict(img_array)
        
#         # Handle different model output formats
#         if predictions.shape[-1] == 5:  # Common NSFW model format
#             # Classes: drawings, hentai, neutral, porn, sexy
#             nsfw_score = predictions[0][1] + predictions[0][3] + predictions[0][4]
#         elif predictions.shape[-1] == 2:  # Binary classification
#             nsfw_score = predictions[0][1]
#         else:
#             nsfw_score = predictions[0][0]
            
#         confidence = float(nsfw_score)
        
#         # If confidence is low, definitely safe
#         if confidence < SAFE_CONTENT_THRESHOLD:
#             return False, 0.0
            
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 TensorFlow detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_nudenet(image_path: str) -> tuple[bool, float]:
#     """Enhanced NudeNet detection with safe content filtering"""
#     try:
#         detections = nude_detector.detect(image_path)
        
#         # First check for safe content indicators
#         safe_content_detected = any(
#             any(safe_word in det['class'].upper() for safe_word in SAFE_CONTENT_INDICATORS)
#             for det in detections
#         )
        
#         if safe_content_detected:
#             return False, 0.0
            
#         # Then check for NSFW content
#         valid_detections = [
#             det for det in detections 
#             if det['class'] in NSFW_CLASSES 
#             and det['score'] >= MIN_DETECTION_CONFIDENCE
#         ]
        
#         if not valid_detections:
#             return False, 0.0
            
#         # Weight detections by severity
#         weights = {
#             'FEMALE_GENITALIA_EXPOSED': 1.0,
#             'MALE_GENITALIA_EXPOSED': 1.0,
#             'ANUS_EXPOSED': 0.95,
#             'SEXUAL_ACTIVITY': 0.9,
#             'ORAL_SEX': 0.9,
#             '_default': 0.7
#         }
        
#         weighted = [d['score'] * weights.get(d['class'], weights['_default']) for d in valid_detections]
#         confidence = max(weighted)
        
#         # If confidence is borderline, be more conservative
#         if 0.4 < confidence < 0.7:
#             confidence *= 0.7
            
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 NudeNet detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_combined(image_path: str) -> tuple[bool, float]:
#     """Intelligent confidence combination with safe content checks"""
#     try:
#         # First check if image is definitely safe
#         img = Image.open(image_path)
        
#         # Simple size check (very small images often false positive)
#         if min(img.size) < 100:
#             return False, 0.0
            
#         # Check for common safe content patterns
#         if is_likely_safe_content(image_path):
#             return False, 0.0
            
#         # Proceed with normal detection
#         n_result, n_conf = detect_nsfw_nudenet(image_path)
#         t_result, t_conf = detect_nsfw_tensorflow(image_path)
        
#         # If either model says it's definitely safe
#         if n_conf < SAFE_CONTENT_THRESHOLD or t_conf < SAFE_CONTENT_THRESHOLD:
#             return False, min(n_conf, t_conf)
            
#         # Weighted average favoring the more conservative model
#         confidence = min(n_conf, t_conf) * 0.6 + max(n_conf, t_conf) * 0.4
        
#         # Only consider it NSFW if both models agree to some degree
#         if n_result and t_result:
#             return True, confidence
#         elif n_conf > 0.7 or t_conf > 0.7:
#             return True, confidence * 0.9  # Slightly reduce confidence for single-model detection
#         else:
#             return False, confidence
            
#     except Exception as e:
#         logger.error(f"🔞 Combined detection failed: {str(e)}")
#         return False, 0.0

# def is_likely_safe_content(image_path: str) -> bool:
#     """Check for obvious safe content patterns"""
#     try:
#         # Check image properties
#         img = Image.open(image_path)
        
#         # Very bright images (likely memes or text)
#         if np.mean(img) > 200:
#             return True
            
#         # Very dark images (likely false positives)
#         if np.mean(img) < 50:
#             return True
            
#         return False
#     except Exception as e:
#         logger.error(f"❌ Safe content check failed: {str(e)}")
#         return False

# def detect_nsfw(image_path: str) -> tuple[bool, float]:
#     """Route to appropriate detection method with safe content checks"""
#     # Quick check for obviously safe content
#     if is_likely_safe_content(image_path):
#         return False, 0.0
        
#     if MODEL_SELECTION == "nudenet":
#         return detect_nsfw_nudenet(image_path)
#     elif MODEL_SELECTION == "tensorflow":
#         return detect_nsfw_tensorflow(image_path)
#     else:
#         return detect_nsfw_combined(image_path)

# def analyze_frames(file_path: str) -> float:
#     """Analyze multiple frames for video/GIF/animated sticker content"""
#     max_confidence = 0.0
#     safe_frames = 0
#     try:
#         if file_path.endswith(('.webm', '.mp4')):
#             vid = cv2.VideoCapture(file_path)
#             total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_step = max(1, total_frames // FRAME_ANALYSIS_COUNT)
            
#             for i in range(FRAME_ANALYSIS_COUNT):
#                 vid.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
#                 success, frame = vid.read()
#                 if success:
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     cv2.imwrite(frame_path, frame)
#                     is_nsfw, confidence = detect_nsfw(frame_path)
                    
#                     # If any frame is definitely safe, reduce overall confidence
#                     if confidence < SAFE_CONTENT_THRESHOLD:
#                         safe_frames += 1
#                         max_confidence = max(max_confidence, confidence * 0.5)
#                     else:
#                         max_confidence = max(max_confidence, confidence)
                    
#                     os.remove(frame_path)
#             vid.release()
            
#         elif file_path.endswith('.gif') or file_path.endswith('.tgs'):
#             with Image.open(file_path) as img:
#                 try:
#                     total_frames = img.n_frames
#                 except AttributeError:
#                     total_frames = 1  # Fallback for static images
                
#                 frame_step = max(1, total_frames // FRAME_ANALYSIS_COUNT)
                
#                 for i in range(FRAME_ANALYSIS_COUNT):
#                     try:
#                         img.seek(i * frame_step)
#                         frame_path = f"{file_path}_frame_{i}.jpg"
#                         img.convert('RGB').save(frame_path)
#                         is_nsfw, confidence = detect_nsfw(frame_path)
                        
#                         if confidence < SAFE_CONTENT_THRESHOLD:
#                             safe_frames += 1
#                             max_confidence = max(max_confidence, confidence * 0.5)
#                         else:
#                             max_confidence = max(max_confidence, confidence)
                        
#                         os.remove(frame_path)
#                     except EOFError:
#                         break  # Reached end of animation
        
#         # If most frames are safe, reduce confidence
#         if safe_frames > FRAME_ANALYSIS_COUNT / 2:
#             max_confidence *= 0.7
            
#         return max_confidence
#     except Exception as e:
#         logger.error(f"🎞 Frame analysis failed: {str(e)}")
#         return 0.0

# async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle model selection command"""
#     global MODEL_SELECTION
#     if not context.args:
#         await update.message.reply_text(
#             f"Current model: {MODEL_SELECTION}\n"
#             "Available options: /model nudenet, /model tensorflow, /model both"
#         )
#         return
    
#     selection = context.args[0].lower()
#     if selection in ("nudenet", "tensorflow", "both"):
#         if selection == "tensorflow" and tf_model is None:
#             await update.message.reply_text("TensorFlow model not available. Using NudeNet.")
#             return
            
#         MODEL_SELECTION = selection
#         await update.message.reply_text(f"Model selection changed to: {MODEL_SELECTION}")
#         logger.info(f"Model selection changed to: {MODEL_SELECTION}")
#     else:
#         await update.message.reply_text("Invalid model. Choose from: nudenet, tensorflow, both")

# async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Enhanced media handler with better feedback and safe content checks"""
#     try:
#         message = update.effective_message
#         user = message.from_user
#         chat = message.chat
        
#         logger.info(f"📩 New media from {user.full_name} ({user.id}) in {chat.title if chat.type != 'private' else 'private chat'}")
        
#         # Skip processing in private chats if needed
#         if chat.type == 'private':
#             logger.info("💬 Message in private chat, skipping analysis")
#             return
            
#         file = None
#         is_video = False
#         is_animated_sticker = False
#         media_type = "unknown"

#         # Media type handling
#         if message.photo:
#             file = await message.photo[-1].get_file()
#             media_type = "photo"
#         elif message.sticker:
#             if message.sticker.is_animated:
#                 logger.info("⚠️ Analyzing animated sticker frames")
#                 file = await message.sticker.get_file()
#                 media_type = "animated_sticker"
#                 is_animated_sticker = True
#             else:
#                 logger.info("ℹ️ Static sticker, likely safe")
#                 return
#         elif message.animation:
#             file = await message.animation.get_file()
#             media_type = "animation"
#             is_video = True
#         elif message.document:
#             if message.document.mime_type.startswith('image/'):
#                 file = await message.document.get_file()
#                 media_type = "image_document"
#             elif message.document.mime_type.startswith('video/'):
#                 file = await message.document.get_file()
#                 media_type = "video_document"
#                 is_video = True
#             else:
#                 logger.info(f"⚠️ Unsupported document type: {message.document.mime_type}")
#                 return

#         if not file:
#             logger.warning("⚠️ No file found in message")
#             return

#         # Download with proper extension
#         file_ext = os.path.splitext(file.file_path)[1] if file.file_path else '.jpg'
#         file_id = str(uuid.uuid4())
#         file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}{file_ext}")
        
#         logger.info(f"⬇️ Downloading {media_type} (size: {file.file_size or 'unknown'} bytes)")
#         await file.download_to_drive(custom_path=file_path)

#         # Analysis with progress feedback
#         analysis_msg = await message.reply_text(
#             f"🔍 Analyzing {media_type} with {MODEL_SELECTION} model..."
#             if not (is_video or is_animated_sticker) else
#             f"🎬 Analyzing {FRAME_ANALYSIS_COUNT} frames from {media_type}..."
#         )

#         try:
#             # Multi-frame analysis for videos/GIFs/animated stickers
#             if is_video or is_animated_sticker or file_ext.lower() in ['.gif', '.webm', '.mp4', '.tgs']:
#                 confidence = analyze_frames(file_path)
#                 is_nsfw_content = confidence > NSFW_THRESHOLD
#             else:
#                 # Single image processing
#                 is_nsfw_content, confidence = detect_nsfw(file_path)

#             # More conservative action - only delete very confident cases
#             if is_nsfw_content and confidence > 0.9:
#                 await message.delete()
#                 logger.info(f"🚫 Deleted NSFW content (Confidence: {confidence:.2%})")
                
#                 warning_msg = (
#                     f"⚠️ Removed explicit content\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}\n"
#                     f"Media type: {media_type}\n\n"
#                     f"*This action was performed automatically*"
#                 )
                
#                 await chat.send_message(
#                     warning_msg,
#                     reply_to_message_id=message.message_id
#                 )
#             elif confidence > NSFW_THRESHOLD:
#                 # For borderline cases, just warn without deleting
#                 logger.info(f"⚠️ Borderline content (Confidence: {confidence:.2%})")
#                 await analysis_msg.edit_text(
#                     f"⚠️ Borderline content detected\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Not removed due to conservative settings"
#                 )
#             else:
#                 logger.info(f"✅ Safe content (Confidence: {confidence:.2%})")
#                 await analysis_msg.edit_text(
#                     f"✅ Content approved\n"
#                     f"Confidence: {confidence:.2%}"
#                 )

#         except Exception as e:
#             logger.error(f"❌ Analysis failed: {str(e)}")
#             await analysis_msg.edit_text("❌ Analysis failed. Please try again.")
#             raise

#         finally:
#             if os.path.exists(file_path):
#                 os.remove(file_path)

#     except Exception as e:
#         logger.error(f"🔥 Critical error in media handler: {str(e)}")
#         if 'analysis_msg' in locals():
#             await analysis_msg.edit_text("❌ An error occurred during processing")

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Send a message when the command /start is issued."""
#     await update.message.reply_text(
#         "🚀 NSFW Detection Bot is running!\n"
#         "I automatically analyze photos, videos, GIFs and stickers for explicit content.\n\n"
#         "Commands:\n"
#         "/model - Change detection model\n"
#         "/help - Show help information"
#     )

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Send a message when the command /help is issued."""
#     await update.message.reply_text(
#         "ℹ️ NSFW Detection Bot Help\n\n"
#         "I analyze these media types:\n"
#         "- Photos (sent as photo or document)\n"
#         "- Videos (up to 20MB)\n"
#         "- GIFs\n"
#         "- Animated stickers\n\n"
#         "Commands:\n"
#         "/model [nudenet|tensorflow|both] - Change detection model\n"
#         "/help - Show this message\n\n"
#         "Current settings:\n"
#         f"Detection threshold: {NSFW_THRESHOLD*100:.0f}%\n"
#         f"Active model: {MODEL_SELECTION}"
#     )

# async def post_init(application):
#     """Run after bot is initialized"""
#     await application.bot.set_my_commands([
#         ("model", "Change detection model"),
#         ("help", "Show help information"),
#     ])
#     logger.info("🤖 Bot initialization complete!")

# async def post_stop(application):
#     """Run before bot is stopped"""
#     logger.info("🛑 Bot shutdown complete!")

# def main():
#     """Start the bot."""
#     try:
#         bot_token = "7803429144:AAFXixTN0-Gb2eX1GE2KJnlTHdvfJBLrlnM"  # Replace with your actual token
        
#         # Create application with error handling
#         app = ApplicationBuilder() \
#             .token(bot_token) \
#             .post_init(post_init) \
#             .post_stop(post_stop) \
#             .build()

#         # Add command handlers
#         app.add_handler(CommandHandler("start", start))
#         app.add_handler(CommandHandler("help", help_command))
#         app.add_handler(CommandHandler("model", handle_model_selection))

#         # Media handler - captures all supported media types
#         media_filter = (
#             filters.PHOTO |
#             filters.Document.IMAGE |
#             filters.Document.VIDEO |
#             filters.Sticker.ALL |
#             filters.ANIMATION
#         )
#         app.add_handler(MessageHandler(media_filter, handle_media))

#         logger.info("🤖 Starting Advanced NSFW Detection Bot...")
#         logger.info(f"🔧 Model selection: {MODEL_SELECTION}")
#         logger.info(f"📁 Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
#         logger.info("📸 Monitoring: Photos | Videos | GIFs | Stickers")
        
#         # Run with better error handling
#         app.run_polling(
#             poll_interval=1.0,
#             timeout=30,
#             drop_pending_updates=True
#         )

#     except Exception as e:
#         logger.critical(f"💥 Fatal error during startup: {str(e)}")
#         raise

# if __name__ == "__main__":
#     try:
#         import asyncio
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         logger.info("🛑 Bot stopped by user")
#     except Exception as e:
#         logger.critical(f"💥 Critical error: {str(e)}")

#################################################################################
#################################################################################
#################################################################################
#################################################################################

# STRICT_NSFW_CLASSES = {
#     # Explicit Nudity (28 classes)
#     'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
#     'ANUS_EXPOSED', 'PUBIC_HAIR_EXPOSED', 'AREOLA_EXPOSED',
#     'NIPPLE_EXPOSED', 'BUTTOCKS_EXPOSED', 'FEMALE_BREAST_EXPOSED',
#     'UNDERWEAR_VISIBLE_EXPLICIT', 'CLOTHING_SEE_THROUGH',
#     'WET_CLOTHING_EXPLICIT', 'EXPOSED_CHEST_MALE', 'EXPOSED_CHEST_FEMALE',
#     'EXPOSED_THIGHS_EXPLICIT', 'EXPOSED_BACK_EXPLICIT',
#     'EXPOSED_STOMACH_EXPLICIT', 'EXPOSED_LEGS_EXPLICIT',
#     'NAKED_BODY_EXPLICIT', 'EXPOSED_ARMPIT_EXPLICIT',
#     'EXPOSED_FEET_FETISH', 'CLOTHING_RIPPED_EXPLICIT',
#     'TOPLESS_EXPLICIT', 'BOTTOMLESS_EXPLICIT',
#     'EXPOSED_BUTTOCKS_CLEAVAGE', 'EXPOSED_BREAST_CLEAVAGE',
#     'EXPOSED_UPPER_BODY_EXPLICIT', 'EXPOSED_LOWER_BODY_EXPLICIT',
#     'PARTIAL_NUDITY_EXPLICIT',

#     # Sexual Activities (32 classes)
#     'SEXUAL_INTERCOURSE', 'ORAL_SEX_ACTIVE', 'ANAL_SEX_ACTIVE',
#     'MASTURBATION_EXPLICIT', 'SEX_TOYS_VISIBLE', 'PORNOGRAPHIC_POSES',
#     'GROUP_SEX_ACTIVITY', 'BDSM_EQUIPMENT', 'FETISH_ACTIVITY_EXPLICIT',
#     'DOMINATION_ACT', 'SUBMISSION_ACT', 'BONDAGE_ACT',
#     'HUMAN_TRAFFICKING_INDICATORS', 'CHILD_EXPLOITATION',
#     'REVENGE_PORN_INDICATORS', 'UPSKIRT_SHOTS', 'DOWNBLOUSE_SHOTS',
#     'NON_CONSENSUAL_ACT', 'EXHIBITIONISM_ACT', 'VOYEURISM_ACT',
#     'SEXUAL_HARASSMENT_ACT', 'CYBERSEX_ACTIVITY', 'SEXTING_INDICATORS',
#     'SEXUAL_SIMULATION_ACT', 'GRINDING_EXPLICIT', 'FROTTAGE_ACT',
#     'INCEST_THEME', 'BESTIALITY_INDICATORS', 'NECROPHILIA_INDICATORS',
#     'SCAT_FETISH_ACT', 'WATERSPORTS_ACT', 'SADOMASOCHISM_ACT',

#     # Exploitation & Illegal (18 classes)
#     'MINOR_IN_EXPLICIT_CONTENT', 'CHILD_ABUSE_MATERIAL',
#     'SEXUALIZED_MINORS', 'TEEN_EXPLOITATION',
#     'REVENGE_PORN_CONTENT', 'BLACKMAIL_CONTENT',
#     'HUMAN_TRAFFICKING_SIGNS', 'DRUG_FACILITATED_ASSAULT',
#     'DATE_RAPE_INDICATORS', 'ALCOHOL_INTOXICATION_EXPLOITATION',
#     'UNCONSCIOUS_PERSON_EXPLOITATION', 'SLEEPING_PERSON_EXPLOITATION',
#     'DRUGGED_PERSON_EXPLOITATION', 'NON_CONSENSUAL_SHARING',
#     'DEEPFAKE_PORN', 'AI_GENERATED_EXPLICIT',
#     'MORPHED_EXPLICIT_CONTENT', 'PRIVATE_PARTS_ZOOM',

#     # Extreme Content (22 classes)
#     'VIOLENT_SEX_ACT', 'BLOOD_IN_SEXUAL_CONTEXT',
#     'GORE_IN_SEXUAL_CONTEXT', 'WEAPONS_IN_SEXUAL_CONTEXT',
#     'HATE_SYMBOLS_EXPLICIT', 'RACIST_SEXUAL_CONTENT',
#     'HOMOPHOBIC_SEXUAL_CONTENT', 'TRANSPHOBIC_SEXUAL_CONTENT',
#     'SELF_HARM_EXPLICIT', 'SUICIDE_IN_SEXUAL_CONTEXT',
#     'ANIMAL_CRUELTY_SEXUAL', 'DRUG_USE_EXPLICIT',
#     'HARD_DRUG_USE_SEXUAL', 'INCEST_EXPLICIT',
#     'RAPE_DEPICTION', 'ABUSE_DEPICTION',
#     'TORTURE_IN_SEXUAL_CONTEXT', 'SNUFF_FILM_INDICATORS',
#     'EXTREME_FETISH_ACT', 'DANGEROUS_ACTS_SEXUAL',
#     'CHOKING_ACT_EXPLICIT', 'AUTOEROTIC_ASPHYXIATION',

#     # Fetish/BDSM (15 classes)
#     'BDSM_HARDCORE', 'SHIBARI_EXPLICIT', 'GOLDEN_SHOWER_ACT',
#     'SCATOLOGY_ACT', 'NECROPHILIA_ACT', 'BESTIALITY_ACT',
#     'FISTING_ACT', 'FEMDOM_EXTREME', 'CBT_EXTREME',
#     'FOOT_FETISH_EXPLICIT', 'UNDERAGE_FETISH',
#     'INCEST_FETISH', 'RAPE_FETISH', 'FORCED_ACT_FETISH',
#     'HYDROPHILIA_EXTREME'
#     }

# # Expanded safe content indicators with contextual categories
# SAFE_CONTENT_INDICATORS = {
#     # Activities & Settings (35 classes)
#     'FAMILY_GATHERING', 'CHILDREN_PLAYING', 'BABY_CARE',
#     'SCHOOL_ACTIVITY', 'DANCE_PERFORMANCE', 'SPORTS_EVENT',
#     'GYM_WORKOUT', 'YOGA_PRACTICE', 'SWIMMING_COMPETITION',
#     'BEACH_ACTIVITIES', 'MEDICAL_EXAM', 'BREASTFEEDING',
#     'CHILDBIRTH_EDUCATION', 'ART_MODEL_SESSION', 'NATURE_PHOTOGRAPHY',
#     'WILDLIFE_DOCUMENTARY', 'CULTURAL_EVENT', 'TRADITIONAL_CLOTHING',
#     'RELIGIOUS_CEREMONY', 'HISTORICAL_ART', 'ANATOMY_EDUCATION',
#     'SEX_EDUCATION', 'MEDICAL_ILLUSTRATION', 'FASHION_SHOW',
#     'THEATRE_PERFORMANCE', 'CIRCUS_ACT', 'ACROBATICS',
#     'MARTIAL_ARTS', 'WRESTLING_MATCH', 'BALLET_PRACTICE',
#     'CHEERLEADING', 'AEROBICS_CLASS', 'PHYSICAL_THERAPY',
#     'SURGICAL_PROCEDURE', 'FIREFIGHTER_TRAINING',

#     # Clothing & Appearance (25 classes)
#     'FULLY_COVERED_BODY', 'PROFESSIONAL_UNIFORM',
#     'SPORTS_UNIFORM', 'SWIMSUIT_NON_SEXUAL',
#     'UNDERWEAR_NON_EXPLICIT', 'CULTURAL_ATTIRE',
#     'HISTORICAL_COSTUME', 'PROTECTIVE_GEAR',
#     'MEDICAL_SCRUBS', 'MILITARY_UNIFORM',
#     'RELIGIOUS_GARB', 'TRADITIONAL_DRESS',
#     'ARTISTIC_NUDITY', 'BODY_PAINT_ART',
#     'SCARIFICATION_ART', 'TATTOO_ART',
#     'PIERCING_ART', 'HAIRSTYLING_DEMO',
#     'MAKEUP_TUTORIAL', 'FASHION_DESIGN',
#     'COSTUME_DESIGN', 'THEATRICAL_MAKEUP',
#     'AGE_APPROPRIATE_CLOTHING', 'MODEST_DRESS',
#     'PROFESSIONAL_ATTIRE',

#     # Nature & Animals (20 classes)
#     'ANIMAL_CARE', 'VETERINARY_PROCEDURE',
#     'WILDLIFE_CONSERVATION', 'MARINE_LIFE',
#     'BIRDWATCHING', 'SAFARI_TOUR',
#     'PET_GROOMING', 'ANIMAL_TRAINING',
#     'FARM_ACTIVITIES', 'HORSE_RIDING',
#     'DOG_SHOW', 'CAT_SHOW',
#     'AQUARIUM_VISIT', 'ZOO_ENVIRONMENT',
#     'NATIONAL_PARK', 'BOTANICAL_GARDEN',
#     'HIKING_ACTIVITY', 'CAMPING_SCENE',
#     'BEACH_CLEANUP', 'ECOLOGY_STUDY',

#     # Medical & Educational (18 classes)
#     'MEDICAL_TRAINING', 'NURSING_PRACTICE',
#     'FIRST_AID_DEMO', 'ANATOMY_CLASS',
#     'BIOLOGY_LAB', 'PSYCHOLOGY_STUDY',
#     'SEX_ED_CLASS', 'BIRTH_CONTROL_ED',
#     'GYNECOLOGY_ED', 'URBAN_HEALTH_ED',
#     'NUTRITION_CLASS', 'FITNESS_EDUCATION',
#     'SURGICAL_TRAINING', 'PHYSIOTHERAPY',
#     'RADIOLOGY_IMAGING', 'MEDICAL_ULTRASOUND',
#     'DERMATOLOGY_STUDY', 'BURN_VICTIM_CARE',

#     # Cultural & Historical (15 classes)
#     'MUSEUM_EXHIBIT', 'ARCHAEOLOGICAL_FIND',
#     'HISTORICAL_ARTIFACT', 'CAVE_PAINTINGS',
#     'TEMPLE_ART', 'RELIGIOUS_ART',
#     'FOLK_DANCE', 'TRADITIONAL_CEREMONY',
#     'CULTURAL_FESTIVAL', 'HISTORICAL_REENACTMENT',
#     'ANCIENT_SCULPTURE', 'ETHNOGRAPHIC_STUDY',
#     'INDIGENOUS_PRACTICES', 'TRIBAL_ART',
#     'ANCESTRAL_WISDOM'
#     }


#################################################################################
#################################################################################
#################################################################################
#################################################################################

# import os
# import logging
# import uuid
# import cv2
# import numpy as np
# from PIL import Image, ImageSequence
# from nudenet import NudeDetector
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from telegram import Update
# from telegram.ext import (
#     ApplicationBuilder,
#     MessageHandler,
#     ContextTypes,
#     filters,
#     CommandHandler
# )

# # Enhanced logging configuration
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO,
#     handlers=[
#         logging.FileHandler("nsfw_bot.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Configuration
# DOWNLOAD_DIR = "downloads"
# os.makedirs(DOWNLOAD_DIR, exist_ok=True)
# NSFW_THRESHOLD = 0.25  # Higher threshold to reduce false positives
# SAFE_THRESHOLD = 0.30  # Below this = definitely safe
# FRAME_ANALYSIS_COUNT = 3  # Balanced frame analysis
# MIN_DETECTION_CONFIDENCE = 0.25  # Higher threshold for more reliable detections

# # Model configuration
# MODEL_SELECTION = "both"  # Using both models for maximum accuracy
# TF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'nsfw.299x299.h5')
# TF_INPUT_SIZE = (299, 299)

# # Enhanced NSFW classes with strict filtering
# # Strict NSFW classes with granular categorization
# STRICT_NSFW_CLASSES = {
#     # Explicit Nudity (28 classes)
#     'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
#     'ANUS_EXPOSED', 'PUBIC_HAIR_EXPOSED', 'AREOLA_EXPOSED',
#     'NIPPLE_EXPOSED', 'BUTTOCKS_EXPOSED', 'FEMALE_BREAST_EXPOSED',
#     'UNDERWEAR_VISIBLE_EXPLICIT', 'CLOTHING_SEE_THROUGH',
#     'WET_CLOTHING_EXPLICIT', 'EXPOSED_CHEST_MALE', 'EXPOSED_CHEST_FEMALE',
#     'EXPOSED_THIGHS_EXPLICIT', 'EXPOSED_BACK_EXPLICIT',
#     'EXPOSED_STOMACH_EXPLICIT', 'EXPOSED_LEGS_EXPLICIT',
#     'NAKED_BODY_EXPLICIT', 'EXPOSED_ARMPIT_EXPLICIT',
#     'EXPOSED_FEET_FETISH', 'CLOTHING_RIPPED_EXPLICIT',
#     'TOPLESS_EXPLICIT', 'BOTTOMLESS_EXPLICIT',
#     'EXPOSED_BUTTOCKS_CLEAVAGE', 'EXPOSED_BREAST_CLEAVAGE',
#     'EXPOSED_UPPER_BODY_EXPLICIT', 'EXPOSED_LOWER_BODY_EXPLICIT',
#     'PARTIAL_NUDITY_EXPLICIT',

#     # Sexual Activities (32 classes)
#     'SEXUAL_INTERCOURSE', 'ORAL_SEX_ACTIVE', 'ANAL_SEX_ACTIVE',
#     'MASTURBATION_EXPLICIT', 'SEX_TOYS_VISIBLE', 'PORNOGRAPHIC_POSES',
#     'GROUP_SEX_ACTIVITY', 'BDSM_EQUIPMENT', 'FETISH_ACTIVITY_EXPLICIT',
#     'DOMINATION_ACT', 'SUBMISSION_ACT', 'BONDAGE_ACT',
#     'HUMAN_TRAFFICKING_INDICATORS', 'CHILD_EXPLOITATION',
#     'REVENGE_PORN_INDICATORS', 'UPSKIRT_SHOTS', 'DOWNBLOUSE_SHOTS',
#     'NON_CONSENSUAL_ACT', 'EXHIBITIONISM_ACT', 'VOYEURISM_ACT',
#     'SEXUAL_HARASSMENT_ACT', 'CYBERSEX_ACTIVITY', 'SEXTING_INDICATORS',
#     'SEXUAL_SIMULATION_ACT', 'GRINDING_EXPLICIT', 'FROTTAGE_ACT',
#     'INCEST_THEME', 'BESTIALITY_INDICATORS', 'NECROPHILIA_INDICATORS',
#     'SCAT_FETISH_ACT', 'WATERSPORTS_ACT', 'SADOMASOCHISM_ACT',

#     # Exploitation & Illegal (18 classes)
#     'MINOR_IN_EXPLICIT_CONTENT', 'CHILD_ABUSE_MATERIAL',
#     'SEXUALIZED_MINORS', 'TEEN_EXPLOITATION',
#     'REVENGE_PORN_CONTENT', 'BLACKMAIL_CONTENT',
#     'HUMAN_TRAFFICKING_SIGNS', 'DRUG_FACILITATED_ASSAULT',
#     'DATE_RAPE_INDICATORS', 'ALCOHOL_INTOXICATION_EXPLOITATION',
#     'UNCONSCIOUS_PERSON_EXPLOITATION', 'SLEEPING_PERSON_EXPLOITATION',
#     'DRUGGED_PERSON_EXPLOITATION', 'NON_CONSENSUAL_SHARING',
#     'DEEPFAKE_PORN', 'AI_GENERATED_EXPLICIT',
#     'MORPHED_EXPLICIT_CONTENT', 'PRIVATE_PARTS_ZOOM',

#     # Extreme Content (22 classes)
#     'VIOLENT_SEX_ACT', 'BLOOD_IN_SEXUAL_CONTEXT',
#     'GORE_IN_SEXUAL_CONTEXT', 'WEAPONS_IN_SEXUAL_CONTEXT',
#     'HATE_SYMBOLS_EXPLICIT', 'RACIST_SEXUAL_CONTENT',
#     'HOMOPHOBIC_SEXUAL_CONTENT', 'TRANSPHOBIC_SEXUAL_CONTENT',
#     'SELF_HARM_EXPLICIT', 'SUICIDE_IN_SEXUAL_CONTEXT',
#     'ANIMAL_CRUELTY_SEXUAL', 'DRUG_USE_EXPLICIT',
#     'HARD_DRUG_USE_SEXUAL', 'INCEST_EXPLICIT',
#     'RAPE_DEPICTION', 'ABUSE_DEPICTION',
#     'TORTURE_IN_SEXUAL_CONTEXT', 'SNUFF_FILM_INDICATORS',
#     'EXTREME_FETISH_ACT', 'DANGEROUS_ACTS_SEXUAL',
#     'CHOKING_ACT_EXPLICIT', 'AUTOEROTIC_ASPHYXIATION',

#     # Fetish/BDSM (15 classes)
#     'BDSM_HARDCORE', 'SHIBARI_EXPLICIT', 'GOLDEN_SHOWER_ACT',
#     'SCATOLOGY_ACT', 'NECROPHILIA_ACT', 'BESTIALITY_ACT',
#     'FISTING_ACT', 'FEMDOM_EXTREME', 'CBT_EXTREME',
#     'FOOT_FETISH_EXPLICIT', 'UNDERAGE_FETISH',
#     'INCEST_FETISH', 'RAPE_FETISH', 'FORCED_ACT_FETISH',
#     'HYDROPHILIA_EXTREME'
#     }

# # Expanded safe content indicators with contextual categories
# SAFE_CONTENT_INDICATORS = {
#     # Activities & Settings (35 classes)
#     'FAMILY_GATHERING', 'CHILDREN_PLAYING', 'BABY_CARE',
#     'SCHOOL_ACTIVITY', 'DANCE_PERFORMANCE', 'SPORTS_EVENT',
#     'GYM_WORKOUT', 'YOGA_PRACTICE', 'SWIMMING_COMPETITION',
#     'BEACH_ACTIVITIES', 'MEDICAL_EXAM', 'BREASTFEEDING',
#     'CHILDBIRTH_EDUCATION', 'ART_MODEL_SESSION', 'NATURE_PHOTOGRAPHY',
#     'WILDLIFE_DOCUMENTARY', 'CULTURAL_EVENT', 'TRADITIONAL_CLOTHING',
#     'RELIGIOUS_CEREMONY', 'HISTORICAL_ART', 'ANATOMY_EDUCATION',
#     'SEX_EDUCATION', 'MEDICAL_ILLUSTRATION', 'FASHION_SHOW',
#     'THEATRE_PERFORMANCE', 'CIRCUS_ACT', 'ACROBATICS',
#     'MARTIAL_ARTS', 'WRESTLING_MATCH', 'BALLET_PRACTICE',
#     'CHEERLEADING', 'AEROBICS_CLASS', 'PHYSICAL_THERAPY',
#     'SURGICAL_PROCEDURE', 'FIREFIGHTER_TRAINING',

#     # Clothing & Appearance (25 classes)
#     'FULLY_COVERED_BODY', 'PROFESSIONAL_UNIFORM',
#     'SPORTS_UNIFORM', 'SWIMSUIT_NON_SEXUAL',
#     'UNDERWEAR_NON_EXPLICIT', 'CULTURAL_ATTIRE',
#     'HISTORICAL_COSTUME', 'PROTECTIVE_GEAR',
#     'MEDICAL_SCRUBS', 'MILITARY_UNIFORM',
#     'RELIGIOUS_GARB', 'TRADITIONAL_DRESS',
#     'ARTISTIC_NUDITY', 'BODY_PAINT_ART',
#     'SCARIFICATION_ART', 'TATTOO_ART',
#     'PIERCING_ART', 'HAIRSTYLING_DEMO',
#     'MAKEUP_TUTORIAL', 'FASHION_DESIGN',
#     'COSTUME_DESIGN', 'THEATRICAL_MAKEUP',
#     'AGE_APPROPRIATE_CLOTHING', 'MODEST_DRESS',
#     'PROFESSIONAL_ATTIRE',

#     # Nature & Animals (20 classes)
#     'ANIMAL_CARE', 'VETERINARY_PROCEDURE',
#     'WILDLIFE_CONSERVATION', 'MARINE_LIFE',
#     'BIRDWATCHING', 'SAFARI_TOUR',
#     'PET_GROOMING', 'ANIMAL_TRAINING',
#     'FARM_ACTIVITIES', 'HORSE_RIDING',
#     'DOG_SHOW', 'CAT_SHOW',
#     'AQUARIUM_VISIT', 'ZOO_ENVIRONMENT',
#     'NATIONAL_PARK', 'BOTANICAL_GARDEN',
#     'HIKING_ACTIVITY', 'CAMPING_SCENE',
#     'BEACH_CLEANUP', 'ECOLOGY_STUDY',

#     # Medical & Educational (18 classes)
#     'MEDICAL_TRAINING', 'NURSING_PRACTICE',
#     'FIRST_AID_DEMO', 'ANATOMY_CLASS',
#     'BIOLOGY_LAB', 'PSYCHOLOGY_STUDY',
#     'SEX_ED_CLASS', 'BIRTH_CONTROL_ED',
#     'GYNECOLOGY_ED', 'URBAN_HEALTH_ED',
#     'NUTRITION_CLASS', 'FITNESS_EDUCATION',
#     'SURGICAL_TRAINING', 'PHYSIOTHERAPY',
#     'RADIOLOGY_IMAGING', 'MEDICAL_ULTRASOUND',
#     'DERMATOLOGY_STUDY', 'BURN_VICTIM_CARE',

#     # Cultural & Historical (15 classes)
#     'MUSEUM_EXHIBIT', 'ARCHAEOLOGICAL_FIND',
#     'HISTORICAL_ARTIFACT', 'CAVE_PAINTINGS',
#     'TEMPLE_ART', 'RELIGIOUS_ART',
#     'FOLK_DANCE', 'TRADITIONAL_CEREMONY',
#     'CULTURAL_FESTIVAL', 'HISTORICAL_REENACTMENT',
#     'ANCIENT_SCULPTURE', 'ETHNOGRAPHIC_STUDY',
#     'INDIGENOUS_PRACTICES', 'TRIBAL_ART',
#     'ANCESTRAL_WISDOM'
#     }

# # Initialize models
# logger.info("⚙️ Loading advanced detection models...")
# nude_detector = NudeDetector()

# # Load TensorFlow model with improved error handling
# tf_model = None
# try:
#     tf_model = load_model(
#         TF_MODEL_PATH,
#         custom_objects={'KerasLayer': tf.keras.layers.Layer},
#         compile=False
#     )
#     logger.info(f"✅ TensorFlow model ({TF_MODEL_PATH}) loaded successfully!")
    
#     # Model warm-up
#     try:
#         dummy_input = np.random.rand(1, *TF_INPUT_SIZE, 3).astype(np.float32)
#         prediction = tf_model.predict(dummy_input)
#         logger.info(f"🔍 Model warm-up successful. Output shape: {prediction.shape}")
#     except Exception as e:
#         logger.error(f"❌ Model warm-up failed: {str(e)}")

# except Exception as e:
#     logger.error(f"❌ Critical error loading TensorFlow model: {str(e)}")
#     if "CUDA" in str(e):
#         logger.warning("⚠️ Trying to load with CPU only...")
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         try:
#             tf_model = load_model(TF_MODEL_PATH, compile=False)
#         except Exception as e:
#             logger.error(f"❌ CPU load failed: {str(e)}")
#     if tf_model is None:
#         logger.warning("⚠️ TensorFlow model disabled")
        
# logger.info(f"✅ Models initialized! Active model: {MODEL_SELECTION}")

# def preprocess_image_for_tf(image_path: str, target_size=TF_INPUT_SIZE) -> np.ndarray:
#     """Enhanced image preprocessing with error recovery"""
#     try:
#         img = tf.keras.preprocessing.image.load_img(
#             image_path,
#             target_size=target_size,
#             color_mode='rgb',
#             interpolation='bilinear'
#         )
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0)
#         return img_array / 255.0
#     except Exception as e:
#         logger.error(f"❌ Preprocessing failed: {str(e)}")
#         return np.zeros((1, *target_size, 3), dtype=np.float32)

# def detect_nsfw_tensorflow(image_path: str) -> tuple[bool, float]:
#     """Improved TensorFlow detection with multiple output support"""
#     if tf_model is None:
#         return False, 0.0
    
#     try:
#         img_array = preprocess_image_for_tf(image_path)
#         predictions = tf_model.predict(img_array)
        
#         # Handle different model output formats
#         if predictions.shape[-1] == 5:  # Common NSFW model format
#             # Classes: drawings, hentai, neutral, porn, sexy
#             nsfw_score = predictions[0][1] + predictions[0][3] * 0.8  # Reduced weight for 'sexy'
#         elif predictions.shape[-1] == 2:  # Binary classification
#             nsfw_score = predictions[0][1]
#         else:
#             nsfw_score = predictions[0][0]
            
#         confidence = float(nsfw_score)
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 TensorFlow detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw_nudenet(image_path: str) -> tuple[bool, float]:
#     """Enhanced NudeNet detection with strict class filtering"""
#     try:
#         detections = nude_detector.detect(image_path)
#         valid_detections = [
#             det for det in detections 
#             if det['class'] in STRICT_NSFW_CLASSES 
#             and det['score'] >= MIN_DETECTION_CONFIDENCE
#         ]
        
#         if not valid_detections:
#             return False, 0.0
            
#         # Get the highest confidence detection
#         max_detection = max(valid_detections, key=lambda x: x['score'])
#         confidence = max_detection['score']
        
#         # Additional checks for common false positives
#         if max_detection['class'] in {'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED'}:
#             # Require higher confidence for these ambiguous cases
#             if confidence < 0.65:
#                 return False, confidence * 0.5  # Reduce confidence for marginal cases
                
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 NudeNet detection failed: {str(e)}")
#         return False, 0.0

# def detect_safe_content(image_path: str) -> bool:
#     """Check for clear safe content indicators"""
#     try:
#         detections = nude_detector.detect(image_path)
#         safe_detections = [
#             det for det in detections 
#             if det['class'] in SAFE_CONTENT_INDICATORS 
#             and det['score'] >= 0.4
#         ]
#         return len(safe_detections) > 0
#     except Exception as e:
#         logger.error(f"🌿 Safe content detection failed: {str(e)}")
#         return False

# def detect_nsfw_combined(image_path: str) -> tuple[bool, float]:
#     """Intelligent confidence combination with safe content check"""
#     try:
#         # First check for clear safe content
#         if detect_safe_content(image_path):
#             return False, 0.0
            
#         n_result, n_conf = detect_nsfw_nudenet(image_path)
#         t_result, t_conf = detect_nsfw_tensorflow(image_path)
        
#         # If content is clearly safe based on either model
#         if n_conf < SAFE_THRESHOLD and t_conf < SAFE_THRESHOLD:
#             return False, max(n_conf, t_conf)
            
#         # If either model is very confident, prioritize it
#         if max(n_conf, t_conf) > 0.9:
#             confidence = max(n_conf, t_conf)
#         else:
#             # Weighted average favoring NudeNet for specific detections
#             confidence = (n_conf * 0.6 + t_conf * 0.4)
        
#         # If both models agree, boost confidence
#         if n_result and t_result:
#             confidence = min(1.0, confidence * 1.1)
            
#         logger.info(f"🔍 Combined confidence: NudeNet={n_conf:.2f}, TF={t_conf:.2f}, Final={confidence:.2f}")
#         return confidence > NSFW_THRESHOLD, confidence
#     except Exception as e:
#         logger.error(f"🔞 Combined detection failed: {str(e)}")
#         return False, 0.0

# def detect_nsfw(image_path: str) -> tuple[bool, float]:
#     """Route to appropriate detection method"""
#     if MODEL_SELECTION == "nudenet":
#         return detect_nsfw_nudenet(image_path)
#     elif MODEL_SELECTION == "tensorflow":
#         return detect_nsfw_tensorflow(image_path)
#     else:
#         return detect_nsfw_combined(image_path)

# def analyze_frames(file_path: str) -> float:
#     """Analyze multiple frames for video/GIF content with better sampling"""
#     max_confidence = 0.0
#     frame_count = 0
    
#     try:
#         if file_path.endswith(('.webm', '.mp4')):
#             vid = cv2.VideoCapture(file_path)
#             total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_step = max(1, total_frames // FRAME_ANALYSIS_COUNT)
            
#             for i in range(FRAME_ANALYSIS_COUNT):
#                 vid.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
#                 success, frame = vid.read()
#                 if success:
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     cv2.imwrite(frame_path, frame)
                    
#                     # First check if frame is clearly safe
#                     if detect_safe_content(frame_path):
#                         os.remove(frame_path)
#                         continue
                        
#                     _, confidence = detect_nsfw(frame_path)
#                     max_confidence = max(max_confidence, confidence)
#                     os.remove(frame_path)
#                     frame_count += 1
                    
#                     # Early exit if we find very explicit content
#                     if max_confidence > 0.95:
#                         break
#             vid.release()
            
#         elif file_path.endswith('.gif'):
#             with Image.open(file_path) as img:
#                 total_frames = img.n_frames
#                 frame_step = max(1, total_frames // FRAME_ANALYSIS_COUNT)
                
#                 for i in range(FRAME_ANALYSIS_COUNT):
#                     img.seek(i * frame_step)
#                     frame_path = f"{file_path}_frame_{i}.jpg"
#                     img.convert('RGB').save(frame_path)
                    
#                     if detect_safe_content(frame_path):
#                         os.remove(frame_path)
#                         continue
                        
#                     _, confidence = detect_nsfw(frame_path)
#                     max_confidence = max(max_confidence, confidence)
#                     os.remove(frame_path)
#                     frame_count += 1
                    
#                     if max_confidence > 0.95:
#                         break
                    
#         # If we analyzed frames but most were safe, reduce confidence
#         if frame_count > 0 and max_confidence < NSFW_THRESHOLD:
#             safe_frame_ratio = (FRAME_ANALYSIS_COUNT - frame_count) / FRAME_ANALYSIS_COUNT
#             if safe_frame_ratio > 0.7:  # If >70% frames were safe
#                 max_confidence *= 0.7  # Reduce confidence
                
#         return max_confidence
#     except Exception as e:
#         logger.error(f"🎞 Frame analysis failed: {str(e)}")
#         return 0.0

# async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle model selection command"""
#     global MODEL_SELECTION
#     if not context.args:
#         await update.message.reply_text(
#             f"Current model: {MODEL_SELECTION}\n"
#             "Available options: /model nudenet, /model tensorflow, /model both"
#         )
#         return
    
#     selection = context.args[0].lower()
#     if selection in ("nudenet", "tensorflow", "both"):
#         if selection == "tensorflow" and tf_model is None:
#             await update.message.reply_text("TensorFlow model not available. Using NudeNet.")
#             return
            
#         MODEL_SELECTION = selection
#         await update.message.reply_text(f"Model selection changed to: {MODEL_SELECTION}")
#         logger.info(f"Model selection changed to: {MODEL_SELECTION}")
#     else:
#         await update.message.reply_text("Invalid model. Choose from: nudenet, tensorflow, both")

# async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Enhanced media handler with better feedback and safety checks"""
#     try:
#         message = update.effective_message
#         user = message.from_user
#         chat = message.chat
        
#         logger.info(f"📩 New media from {user.full_name} ({user.id}) in {chat.title if chat.type != 'private' else 'private chat'}")
        
#         file = None
#         is_video = False
#         media_type = "unknown"

#         # Enhanced media type handling
#         if message.photo:
#             file = await message.photo[-1].get_file()
#             media_type = "photo"
#         elif message.sticker:
#             if message.sticker.is_animated:
#                 logger.info("⚠️ Analyzing animated sticker frames")
#                 file = await message.sticker.get_file()
#                 media_type = "animated_sticker"
#                 is_video = True
#             else:
#                 file = await message.sticker.get_file()
#                 media_type = "sticker"
#         elif message.animation:
#             file = await message.animation.get_file()
#             media_type = "animation"
#             is_video = True
#         elif message.document:
#             if message.document.mime_type.startswith('image/'):
#                 file = await message.document.get_file()
#                 media_type = "image_document"
#             elif message.document.mime_type.startswith('video/'):
#                 file = await message.document.get_file()
#                 media_type = "video_document"
#                 is_video = True
#             else:
#                 logger.info(f"⚠️ Unsupported document type: {message.document.mime_type}")
#                 return

#         if not file:
#             logger.warning("⚠️ No file found in message")
#             return

#         # Download with proper extension and hash
#         file_ext = os.path.splitext(file.file_path)[1] if file.file_path else '.jpg'
#         file_id = str(uuid.uuid4())
#         file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}{file_ext}")
        
#         logger.info(f"⬇️ Downloading {media_type} (size: {file.file_size or 'unknown'} bytes)")
#         await file.download_to_drive(custom_path=file_path)

#         # Enhanced analysis with progress feedback
#         analysis_msg = await message.reply_text(
#             f"🔍 Analyzing {media_type} with {MODEL_SELECTION} model..."
#             if not is_video else
#             f"🎬 Analyzing {FRAME_ANALYSIS_COUNT} frames from {media_type}..."
#         )

#         try:
#             # Multi-frame analysis for videos/GIFs/animated stickers
#             if is_video or file_ext in ['.gif', '.webm', '.mp4']:
#                 confidence = analyze_frames(file_path)
#                 is_nsfw_content = confidence > NSFW_THRESHOLD
#             else:
#                 # Single image processing
#                 is_nsfw_content, confidence = detect_nsfw(file_path)

#             # Enhanced result handling
#             if is_nsfw_content and confidence > NSFW_THRESHOLD:
#                 await message.delete()
#                 logger.info(f"🚫 Deleted NSFW content (Confidence: {confidence:.2%})")
                
#                 warning_msg = (
#                     f"⚠️ Removed explicit content\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}\n"
#                     f"Media type: {media_type}\n\n"
#                     f"*This action was performed automatically*"
#                 )
                
#                 await chat.send_message(
#                     warning_msg,
#                     reply_to_message_id=message.message_id if chat.type != 'private' else None
#                 )
#             else:
#                 logger.info(f"✅ Safe content (Confidence: {confidence:.2%})")
#                 await analysis_msg.edit_text(
#                     f"✅ Content approved\n"
#                     f"Confidence: {confidence:.2%}\n"
#                     f"Model: {MODEL_SELECTION}"
#                 )

#         except Exception as e:
#             logger.error(f"❌ Analysis failed: {str(e)}")
#             await analysis_msg.edit_text("❌ Analysis failed. Please try again.")
#             raise

#         finally:
#             # Cleanup
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#             else:
#                 logger.warning(f"⚠️ File not found during cleanup: {file_path}")

#     except Exception as e:
#         logger.error(f"🔥 Critical error in media handler: {str(e)}")
#         if 'analysis_msg' in locals():
#             await analysis_msg.edit_text("❌ An error occurred during processing")

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Send a message when the command /start is issued."""
#     await update.message.reply_text(
#         "🚀 NSFW Detection Bot is running!\n"
#         "I automatically analyze photos, videos, GIFs and stickers for explicit content.\n\n"
#         "Commands:\n"
#         "/model - Change detection model\n"
#         "/help - Show help information"
#     )

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Send a message when the command /help is issued."""
#     await update.message.reply_text(
#         "ℹ️ NSFW Detection Bot Help\n\n"
#         "I analyze these media types:\n"
#         "- Photos (sent as photo or document)\n"
#         "- Videos (up to 20MB)\n"
#         "- GIFs\n"
#         "- Static and animated stickers\n\n"
#         "Commands:\n"
#         "/model [nudenet|tensorflow|both] - Change detection model\n"
#         "/help - Show this message\n\n"
#         "Current settings:\n"
#         f"Detection threshold: {NSFW_THRESHOLD*100:.0f}%\n"
#         f"Active model: {MODEL_SELECTION}"
#     )

# async def post_init(application):
#     """Run after bot is initialized"""
#     await application.bot.set_my_commands([
#         ("model", "Change detection model"),
#         ("help", "Show help information"),
#     ])
#     logger.info("🤖 Bot initialization complete!")

# async def post_stop(application):
#     """Run before bot is stopped"""
#     logger.info("🛑 Bot shutdown complete!")

# def main():
#     """Start the bot."""
#     try:
#         bot_token = "7803429144:AAFXixTN0-Gb2eX1GE2KJnlTHdvfJBLrlnM"  # Replace with your actual token
        
#         # Create application with error handling
#         app = ApplicationBuilder() \
#             .token(bot_token) \
#             .post_init(post_init) \
#             .post_stop(post_stop) \
#             .build()

#         # Add command handlers
#         app.add_handler(CommandHandler("start", start))
#         app.add_handler(CommandHandler("help", help_command))
#         app.add_handler(CommandHandler("model", handle_model_selection))

#         # Media handler - captures all supported media types
#         media_filter = (
#             filters.PHOTO |
#             filters.Document.IMAGE |
#             filters.Document.VIDEO |
#             filters.Sticker.ALL |
#             filters.ANIMATION
#         )
#         app.add_handler(MessageHandler(media_filter, handle_media))

#         logger.info("🤖 Starting Advanced NSFW Detection Bot...")
#         logger.info(f"🔧 Model selection: {MODEL_SELECTION}")
#         logger.info(f"📁 Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
#         logger.info("📸 Monitoring: Photos | Videos | GIFs | Stickers")
        
#         # Run with better error handling
#         app.run_polling(
#             poll_interval=1.0,
#             timeout=30,
#             drop_pending_updates=True
#         )

#     except Exception as e:
#         logger.critical(f"💥 Fatal error during startup: {str(e)}")
#         raise

# if __name__ == "__main__":
#     try:
#         import asyncio
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         logger.info("🛑 Bot stopped by user")
#     except Exception as e:
#         logger.critical(f"💥 Critical error: {str(e)}")
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

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
        bot_token = "7803429144:AAFXixTN0-Gb2eX1GE2KJnlTHdvfJBLrlnM"  # Replace with your actual token
        
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
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.critical(f"💥 Critical error: {str(e)}")