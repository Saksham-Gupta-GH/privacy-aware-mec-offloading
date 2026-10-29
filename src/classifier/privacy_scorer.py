from pathlib import Path
from typing import List, Tuple
import joblib
import config
from src.models.task import HealthcareTask

class PrivacyScorer:
    """
    ML Privacy Scorer that infers privacy levels and confidence scores
    for incoming healthcare tasks.
    """
    def __init__(self, model_dir: Path = None):
        if model_dir is None:
            model_dir = config.BASE_DIR / "data" / "trained_models"

        model_path = model_dir / "privacy_classifier.joblib"
        vec_path = model_dir / "tfidf_vectorizer.joblib"

        if not model_path.exists() or not vec_path.exists():
            raise FileNotFoundError(
                f"Trained model artifacts missing in {model_dir}. Please run train_classifier.py first."
            )

        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)

    def score_text(self, text: str) -> Tuple[int, float]:
        """
        Returns (predicted_level, confidence) for raw text.
        """
        vec = self.vectorizer.transform([text])
        pred_level = int(self.classifier.predict(vec)[0])
        probabilities = self.classifier.predict_proba(vec)[0]
        confidence = float(probabilities[pred_level])
        return pred_level, confidence

    def annotate_task(self, task: HealthcareTask) -> HealthcareTask:
        """
        Predicts privacy level and confidence for a HealthcareTask object in place.
        """
        level, conf = self.score_text(task.description)
        task.predicted_privacy_level = level
        task.privacy_confidence = conf
        return task

    def annotate_batch(self, tasks: List[HealthcareTask]) -> List[HealthcareTask]:
        """
        Efficient batch annotation for task streams.
        """
        if not tasks:
            return tasks

        texts = [t.description for t in tasks]
        vecs = self.vectorizer.transform(texts)
        preds = self.classifier.predict(vecs)
        probs = self.classifier.predict_proba(vecs)

        for i, task in enumerate(tasks):
            pred_level = int(preds[i])
            task.predicted_privacy_level = pred_level
            task.privacy_confidence = float(probs[i][pred_level])

        return tasks

