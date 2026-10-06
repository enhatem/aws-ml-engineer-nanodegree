
import json
import boto3

client = boto3.client('sagemaker')

def lambda_handler(event, context):
    response = client.start_pipeline_execution(
        PipelineName="l3e3-pipeline"
    )
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Pipeline execution started',
            'executionArn': response['PipelineExecutionArn'],
        })
    }
