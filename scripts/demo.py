"""Small release demo using the public package API and trusted baseline."""

from pathlib import Path

from romanized_nepali_nlp import PipelineConfig, PreprocessingPipeline, tokenize
from romanized_nepali_nlp.sentiment import SentimentModel

text = "yo geet raamrooo chha dheraiii 🔥"
result = PreprocessingPipeline(PipelineConfig(reduce_repetitions=True)).preprocess(text)
print("Input:", text)
print("Tokens:", [(t.text, t.category.value) for t in tokenize(text)])
print("Normalized:", result.normalized_text)
model = SentimentModel.load(Path(__file__).parents[1] / "runs/char/model.pkl")
prediction = model.predict_texts([text], ids=["demo"])[0]
print("Prediction:", prediction.predicted_label)
print("Scores:", {str(k): round(v, 4) for k, v in prediction.scores.items()})
