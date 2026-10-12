import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

response = client.converse(
    modelId="amazon.nova-lite-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "In one sentence, what is Amazon Bedrock?"}
            ]
        }
    ],
    inferenceConfig={
        "maxTokens": 100,
        "temperature": 0
    }
)

text = response["output"]["message"]["content"][0]["text"]

print(text)