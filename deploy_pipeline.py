import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor

def create_deployment_pipeline(
    role,
    sagemaker_session,
    model_artifacts_uri,
    deploy_output_uri,
    deployment_instance_type='ml.t3.medium',
):
    """
    Create a SageMaker Pipeline specifically for deploying a model.

    Args:
        role (str): The IAM role ARN.
        sagemaker_session (sagemaker.Session): SageMaker session.
        model_artifacts_uri (str): S3 URI of the model artifacts.
        deploy_output_uri (str): S3 URI for deployment artifacts.
        deployment_instance_type (str): Instance type for deployment.

    Returns:
        sagemaker.workflow.pipeline.Pipeline: SageMaker pipeline for deployment.
    """

    # Create a pipeline session
    pipeline_session = PipelineSession()

    image_uri = "750573229682.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgb:latest"

    # ScriptProcessor for deployment
    deployment_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role,
        instance_type=deployment_instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session
    )

    # Deployment step
    deployment_step = ProcessingStep(
        name='DeployIrisModel',
        processor=deployment_processor,
        # inputs=[
        #     ProcessingInput(
        #         source=model_artifacts_uri,
        #         destination='/opt/ml/processing/input/model'
        #     )
        # ],
        # outputs=[
        #     ProcessingOutput(
        #         source='/opt/ml/processing/output',
        #         destination=deploy_output_uri,
        #         output_name='DeploymentArtifacts'
        #     )
        # ],
        code='src/deploy.py'
    )

    # Create the deployment pipeline
    deployment_pipeline = Pipeline(
        name='iris-deployment-pipeline',
        steps=[deployment_step],
        sagemaker_session=pipeline_session
    )

    return deployment_pipeline

def main():
    # Initialize SageMaker session and get role
    sagemaker_session = sagemaker.Session()

    role = 'arn:aws:iam::750573229682:role/service-role/AmazonSageMaker-ExecutionRole-20241211T150457'

    # # S3 URIs for model artifacts and deployment output
    # model_artifacts_uri = "s3://mlflow-sagemaker-us-east-1-750573229682/iris-model-output/"
    # deploy_output_uri = "s3://mlflow-sagemaker-us-east-1-750573229682/iris-deploy-output/"
    model_artifacts_uri = "s3://sagemaker-xgb/iris-model-output/"
    deploy_output_uri = "s3://sagemaker-xgb/iris-deploy-output/"

    # Create deployment pipeline
    deployment_pipeline = create_deployment_pipeline(
        role,
        sagemaker_session,
        model_artifacts_uri,
        deploy_output_uri
    )
    # Upsert pipeline
    deployment_pipeline.upsert(role_arn=role)

    # Execute the pipeline
    execution = deployment_pipeline.start()

    # Wait for the pipeline to finish
    execution.wait()

    print("Deployment pipeline execution completed.")
    print("Pipeline Execution Status:", execution.describe()['PipelineExecutionStatus'])

if __name__ == '__main__':
    main()