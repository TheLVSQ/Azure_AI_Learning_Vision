import os
import config
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Set the value of my computer vision endpoint and key from the config.py file

try:
    endpoint = config.VISION_ENDPOINT
    key = config.VISION_KEY
except KeyError:
    print("Please set the VISION_ENDPOINT and VISION_KEY environment variables.")
    exit()


# Create an Image Analysis client

client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Get a caption for the image - this is a synchoronously (blocking) call
result = client.analyze_from_url(
    image_url="https://media.pinatafarm.com/protected/13C23573-190B-4CD5-8692-28D5AEE4DC73/7baf2543-0f2f-48f1-a4a1-a1d6d4ce7ac7-1678507752103-pfarm-with-png-watermarked.png",
    visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.READ,
        VisualFeatures.OBJECTS,
        VisualFeatures.TAGS,
    ],
    gender_neutral_caption=False,
)

# Print out the results!
print("Image analysis results:")

print(" Caption:")

if result.caption is not None:
    print(f"  {result.caption.text}', Confidence: {result.caption.confidence:.4f}")

# Print out the text (OCR) analysis (if any) to the console:

print(" Read:")
if result.read is not None:
    for line in result.read.blocks[0].lines:
        print(f"  Line: '{line.text}', Bounding box {line.bounding_polygon}")
        for word in line.words:
            print(
                f"   Word: '{word.text}', Bouding polygon: {word.bounding_polygon}, Confidence: {word.confidence:.4f}"
            )

# Print out the entire result object to see what else is available
print(result)
