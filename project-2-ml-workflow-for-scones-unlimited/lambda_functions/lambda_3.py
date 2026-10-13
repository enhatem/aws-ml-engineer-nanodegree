import json

THRESHOLD = .93

def lambda_handler(event, context):

    inferences = event['inferences']

    meets_threshold = max(inferences) > THRESHOLD

    if not meets_threshold:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event
    }