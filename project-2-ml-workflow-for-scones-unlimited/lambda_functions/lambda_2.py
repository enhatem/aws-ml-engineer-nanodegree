import json
import base64
import boto3

ENDPOINT = "image-classification-endpoint"

runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):

    image = base64.b64decode(event['image_data'])

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='image/png',
        Body=image,
    )

    inferences = json.loads(response['Body'].read().decode('utf-8'))

    event["inferences"] = inferences
    return {
        'statusCode': 200,
        'body': event
    }