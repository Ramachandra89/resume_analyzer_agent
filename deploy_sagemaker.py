import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.pytorch import PyTorch
import boto3
import os
from pathlib import Path

def create_sagemaker_model():
    # Initialize SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Define model configuration
    model_config = {
        'model_name': 'llama-2-7b',
        'instance_type': 'ml.g5.xlarge',  # GPU instance with 24GB memory
        'volume_size': 100,  # GB
        'max_runtime_in_seconds': 3600,
        'environment': {
            'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600',
            'SAGEMAKER_MODEL_SERVER_WORKERS': '1',
            'TRANSFORMERS_CACHE': '/opt/ml/model'
        }
    }
    
    # Create SageMaker model
    pytorch_model = PyTorchModel(
        model_data=f"s3://{session.default_bucket()}/models/llama-2-7b/model.tar.gz",
        role=role,
        framework_version='2.1.0',
        py_version='py310',
        entry_point='inference.py',
        source_dir='.',
        image_uri=f"{session.account_id}.dkr.ecr.{session.boto_region_name}.amazonaws.com/resume-coach:latest",
        **model_config
    )
    
    return pytorch_model

def deploy_endpoint(model):
    # Deploy the model to an endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.g5.xlarge',
        endpoint_name='resume-coach-endpoint'
    )
    return predictor

def main():
    # Create and deploy the model
    model = create_sagemaker_model()
    predictor = deploy_endpoint(model)
    print(f"Endpoint deployed: {predictor.endpoint_name}")

if __name__ == "__main__":
    main() 