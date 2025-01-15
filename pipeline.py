import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor

def create_sagemaker_pipeline(
    role,
    sagemaker_session,
    input_data_uri,
    output_data_uri,
    model_output_uri,
    deploy_output_uri,
    processing_instance_type='ml.t3.medium',
    training_instance_type='ml.c4.xlarge',
    deployment_instance_type='ml.t3.medium',
):
    
     # Create a pipeline session
    pipeline_session = PipelineSession()

    image_uri = "750573229682.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgb:latest"

    # ScriptProcessor for preprocessing 
    script_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session
    )

    # Preprocessing step
    processing_step = ProcessingStep(
        name='PreprocessIrisData',
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=input_data_uri,
                destination='/opt/ml/processing/input' 
            )
        ],
        outputs=[
            ProcessingOutput(
                source='/opt/ml/processing/output',
                destination=output_data_uri,
                output_name='ProcessedData'
            )
        ],
        code='src/preprocessing.py'
    )


    # ScriptProcessor for training 
    training_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role,
        instance_type=training_instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session
    )

    # Training step
    training_step = ProcessingStep(
        name='TrainIrisModel',
        processor=training_processor,
        inputs=[
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri,
                destination='/opt/ml/processing/input/train'
            )
        ],
        outputs=[
            ProcessingOutput(
                source='/opt/ml/processing/output',
                destination=model_output_uri,
                output_name='ModelArtifacts'
            )
        ],
        code='src/train.py'
    )


     # Create pipeline
    pipeline = Pipeline(
        name='xgb-iris-mlflow-pipeline',
        steps=[processing_step, training_step],
        sagemaker_session=pipeline_session
    )

    return pipeline

def main():
    # Initialize SageMaker session and get role
    sagemaker_session = sagemaker.Session()

    role = 'arn:aws:iam::750573229682:role/service-role/AmazonSageMaker-ExecutionRole-20241211T150457'

    # S3 URIs for input and output data
    input_data_uri = "s3://mlflow-sagemaker-us-east-1-750573229682/iris-dataset/iris_xgb/"
    output_data_uri = "s3://mlflow-sagemaker-us-east-1-750573229682/iris-output/"
    model_output_uri = "s3://mlflow-sagemaker-us-east-1-750573229682/iris-model-output/"
    deploy_output_uri = "s3://mlflow-sagemaker-us-east-1-750573229682/iris-deploy-output/"

    # Create pipeline
    pipeline = create_sagemaker_pipeline(
        role,
        sagemaker_session,
        input_data_uri,
        output_data_uri,
        model_output_uri,
        deploy_output_uri,
    )

    # Upsert pipeline
    pipeline.upsert(role_arn=role)

    # Execute the pipeline
    execution = pipeline.start()

    # Wait for the pipeline to finish
    execution.wait()

    print("Pipeline execution completed.")
    print("Pipeline Execution Status:", execution.describe()['PipelineExecutionStatus'])


if __name__ == '__main__':
    main()
