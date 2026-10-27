
import joblib
import os


def model_fn(model_dir):
    # SageMaker extracts model.tar.gz into model_dir.
    # model.joblib was saved to SM_MODEL_DIR during training.
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)
