import streamlit as st
from streamlit.hello.utils import show_code
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image

import segmentation_models_pytorch as smp


def segmentation_demo():
    model_weights = st.file_uploader("Choose a model file")
    image = st.file_uploader("Choose an image")

    if image is not None:
        image = Image.open(image).convert("RGB")
        st.image(image, "Uploaded image")

    if model_weights is not None:
        ### BEGIN SOLUTION
        # Define the model using the same architecture as used during training.
        model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
        ### END SOLUTION

        ### BEGIN SOLUTION
        # Load the model weights from the uploaded file
        model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
        ### END SOLUTION

        ### BEGIN SOLUTION
        # Set the model to evaluation mode
        model.eval()
        ### END SOLUTION

        ### BEGIN SOLUTION
        # Define the image transformation pipeline
        transforms = v2.Compose([
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ### END SOLUTION

        if image is not None:
            ### BEGIN SOLUTION
            # Apply the transformations and convert the image to a tensor
            tensor = transforms(tv_tensors.Image(image))[None, ...]
            with torch.no_grad():
                # Do the forward pass
                mask = model(tensor).numpy().clip(0, 1)
            # Show image
            st.image(mask[0, 0], "Predicted mask")

# Setup page
st.set_page_config(page_title="Segmentation Demo", page_icon="🔬")
st.markdown("# Segmentation Demo")
st.sidebar.header("Segmentation Demo")

# Run page
segmentation_demo()

show_code(segmentation_demo)