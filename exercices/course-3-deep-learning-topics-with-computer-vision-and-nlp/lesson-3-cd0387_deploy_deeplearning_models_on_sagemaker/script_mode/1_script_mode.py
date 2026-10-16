# from sagemaker.pytorch import PyTorch              <- removed in v3 (framework estimators -> ModelTrainer)
# from sagemaker import get_execution_role           <- moved in v3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core import image_uris
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import (
    Compute,
    InputData,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)


session = Session()
role = get_execution_role()
region = session.boto_region_name
bucket = session.default_bucket()

# The training script detects CUDA and uses the GPU if present, so either works:
#   "ml.g4dn.xlarge" -> GPU image is retrieved and training runs on the GPU
#   "ml.m5.xlarge"   -> CPU image is retrieved and training runs on the CPU
# image_uris.retrieve picks the matching cpu/gpu image from this value.
INSTANCE_TYPE = "ml.g4dn.xlarge"

# Hyperparameters are passed to the script as CLI args (--epochs 2 --lr 0.01 ...).
# Values as strings: they travel over the API as strings either way.
hyperparameters = {
    "epochs": "2",
    "batch-size": "64",
    "test-batch-size": "1000",
    "lr": "0.01",
    "momentum": "0.9",
}

# Replaces v2's PyTorch(entry_point=..., source_dir=..., framework_version=...)
estimator = ModelTrainer(
    training_image=image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.1",
        py_version="py310",
        instance_type=INSTANCE_TYPE,
        image_scope="training",
    ),
    source_code=SourceCode(
        source_dir="scripts",              # relative to this file (project root)
        entry_script="pytorch_cifar.py",   # relative to source_dir
    ),
    role=role,
    compute=Compute(
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        volume_size_in_gb=10,
    ),
    output_data_config=OutputDataConfig(
        s3_output_path=f"s3://{bucket}/cifar10-script-mode/output",
    ),
    stopping_condition=StoppingCondition(max_runtime_in_seconds=900),
    hyperparameters=hyperparameters,
    base_job_name="pytorch-cifar10",
)

# An input channel is required since we are not downloading the dataset directly from the container. We are fetching it from S3
# Replaces v2's estimator.fit()
estimator.train(
    input_data_config=[
        InputData(channel_name="training", data_source=f"s3://{bucket}/course3/lesson3/cifar10/data/")
    ],
    wait=True,
)

print(estimator._latest_training_job.model_artifacts.s3_model_artifacts)
