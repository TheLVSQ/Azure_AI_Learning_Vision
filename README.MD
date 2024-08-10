# Azure AI Vision Image Description - Learning

This repository contains a Python script I modified to utilize Azure AI services to describe images. The script follows the official Azure AI services guide to implement the Vision API for image description.

## Getting Started

### Prerequisites

To run this script, you'll need:

- Python 3.x
- An Azure account with access to the Vision API
- An API key and endpoint URL for the Vision service

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/azure-ai-vision-image-description.git
    cd azure-ai-vision-image-description
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Create the `config.py` file (example included) with your Azure Vision API key and endpoint.

2. Replace the image_url variable to the url of the image you want to analyze

2. Run the script:

    ```bash
    python analyze_image.py
    ```

### Example

Here's an example of how to use the script:

```bash
python analyze_image.py 

### Learning Resources

This project was created as a learning exercise by following the Azure AI services guide: https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/

### Contributing

Feel free to fork this repository and submit pull requests if you'd like to contribute!

