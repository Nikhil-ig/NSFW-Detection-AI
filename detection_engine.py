from nudenet import NudeDetector
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os

class DetectionEngine:
    def __init__(self):
        self.nude_classifier = NudeDetector()
        self.tf_model = os.path.join(os.path.dirname(__file__), 'nsfw.299x299.h5')


    def analyze(self, path):
        nudity_score = self._nudenet_score(path)
        tf_score, tf_class = self._tf_score(path)

        confidence = (nudity_score + tf_score) / 2

        if confidence > 0.7:
            return {"is_nsfw": True, "reason": f"Confidence: {confidence:.2f} ({tf_class})"}
        else:
            return {"is_nsfw": False}

    def _nudenet_score(self, path):
        result = self.nude_classifier.classify(path)
        return result.get(path, {}).get("unsafe", 0)

    def _tf_score(self, path):
        img = keras_image.load_img(path, target_size=(299, 299))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        preds = self.tf_model.predict(x)[0]
        classes = ["drawings", "hentai", "neutral", "porn", "sexy"]
        nsfw_classes = {"porn", "sexy", "hentai"}
        nsfw_score = sum(preds[i] for i, cls in enumerate(classes) if cls in nsfw_classes)
        top_class = classes[np.argmax(preds)]
        return nsfw_score, top_class
