import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def _collect_images_by_class(dataset):
    """Collect images for each class."""
    class_images = {}
    for img, label in dataset.imgs:
        if label not in class_images:
            class_images[label] = []
        class_images[label].append(img)
    return class_images

def _select_random_images(class_images):
    """Randomly select a specified number of images from each class."""
    selected_images_by_class = {}
    for label, images in class_images.items():
        selected_images_by_class[label] = random.sample(images, 10)
    return selected_images_by_class

def _visualize_saliency(image_path, model, image_processor, device, input_size=(224,224)):
    """Visualize saliency maps for a given image."""
    model.eval()

    original_image = Image.open(image_path).convert("RGB")
    resized_image = original_image.resize(input_size)  # Resize to model input size
    inputs = image_processor(images=resized_image, return_tensors="pt")

    # Move the pixel values to the appropriate device
    pixel_values = inputs['pixel_values'].to(device)  # Set to the same device as the model
    pixel_values.requires_grad_()  # Enable gradients for the input image tensor

    # Forward pass to get model predictions
    outputs = model(pixel_values=pixel_values)

    # Get the predicted class
    pred_class = outputs.logits.argmax(dim=-1)

    # Backpropagate to get the gradients
    model.zero_grad()
    loss = outputs.logits[0, pred_class]
    loss.backward()

    # Get the gradients
    gradients = pixel_values.grad.data.cpu().numpy()[0]  # Get gradients of the input image

    # Take the maximum value across the channels and normalize
    saliency = np.max(np.abs(gradients), axis=0)  # Take absolute value
    saliency = saliency / saliency.max()  # Normalize

    return original_image, saliency  # Return the original image and saliency map

def _plot_saliency_maps(model, image_processor, selected_images_by_class, device, num_classes, label_to_class, input_size=(224,224)):
    """Plot saliency maps for selected images grouped by class."""
    images_per_row = 10
    total_images = images_per_row * num_classes

    num_rows = (total_images + images_per_row - 1) // images_per_row  # Calculate number of rows needed
    plt.figure(figsize=(15, 3 * num_rows))

    for i, (label, images) in enumerate(selected_images_by_class.items()):
        for j, image_path in enumerate(images[:total_images]):
            original_image, saliency = _visualize_saliency(image_path, model, image_processor, device, input_size=input_size)

            # Display the saliency map in the first row
            ax = plt.subplot(num_rows * 2, images_per_row, (i * 2) * images_per_row + (j % images_per_row) + 1)
            saliency_resized = np.array(
                Image.fromarray((saliency * 255).astype(np.uint8)).resize(original_image.size))
            ax.imshow(saliency_resized, cmap='hot')
            ax.axis('off')  # Turn off axis
            if j == 0:  # Set the title for the class only on the first image
                ax.set_title(f'Class: {label_to_class[label]}', fontsize=12)

            # Display the original image in the second row
            ax = plt.subplot(num_rows * 2, images_per_row,
                             (i * 2 + 1) * images_per_row + (j % images_per_row) + 1)
            ax.imshow(original_image)
            ax.axis('off')  # Turn off axis

    plt.tight_layout()
    plt.show()

def saliency_evaluation(dataset, model, image_processor, device, num_classes, label_to_class, input_size=(224,224)):
    class_images = _collect_images_by_class(dataset)
    selected_images_by_class = _select_random_images(class_images)
    _plot_saliency_maps(model, image_processor, selected_images_by_class, device, num_classes, label_to_class, input_size=input_size)