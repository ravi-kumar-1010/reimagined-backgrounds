import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
import cv2
import numpy as np



# Initialize the Stable Diffusion Inpainting Pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)

pipe.to("cuda")

def zoom_out(image, n):
    # Original dimensions
    original_width, original_height = image.size

    # Calculate new dimensions after shrinking
    new_width = int(original_width // n)
    new_height = int(original_height // n)

    # Resize the image (shrink the object)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new white canvas that is the same size as the original image
    new_image = Image.new("RGB", (original_width, original_height), "white")

    # Calculate the position to paste the resized object in the center
    x_offset = (original_width - new_width) // 2
    y_offset = (original_height - new_height) // 2

    # Paste the resized image onto the center of the new canvas
    new_image.paste(resized_image, (x_offset, y_offset))

    # Calculate coordinates to crop the central 512x512 area
    left = (original_width - 512) // 2
    top = (original_height - 512) // 2
    right = left + 512
    bottom = top + 512

    # Crop the center 512x512 region
    final_image = new_image.crop((left, top, right, bottom))

    return final_image


def create_contour_mask(image, tolerance=10):
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Define a tolerance range around white
    lower_bound = np.array([255 - tolerance] * 3)
    upper_bound = np.array([255] * 3)

    # Create a mask: black for background, white for object
    mask = np.ones((image_array.shape[0], image_array.shape[1]), dtype=np.uint8) * 255
    non_white_pixels = np.any((image_array < lower_bound) | (image_array > upper_bound), axis=-1)
    mask[non_white_pixels] = 0  # Non-white pixels become black in the mask

    # Perform dilation to fill in any gaps (like white text within the object)
    kernel = np.ones((2, 2), np.uint8)  # You can adjust the size of the kernel as needed
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    # Convert to PIL image for easier visualization and further processing
    mask_image = Image.fromarray(dilated_mask)

    return mask_image


def process_image(image: Image.Image, prompt: str, slider_value: float) -> Image.Image:
    # Perform the image processing
    if slider_value != 1:
        image = zoom_out(image, slider_value)
    mask = create_contour_mask(image, tolerance=10)
    processed_image = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    return processed_image


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate an inpainted image using Stable Diffusion")
    parser.add_argument('--image', required=True, help="Path to the input image")
    parser.add_argument('--text-prompt', required=True, help="Text prompt for generating the image")
    parser.add_argument('--output', required=True, help="Path to save the generated image")
    parser.add_argument('--zoom', type=float, default=1, help="Zoom level (default is 1, no zoom)")

    # Parse arguments
    args = parser.parse_args()

    # Load the image
    image = Image.open(args.image)

    # Process the image
    processed_image = process_image(image, args.text_prompt, args.zoom)

    # Save the output image
    processed_image.save(args.output)
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
