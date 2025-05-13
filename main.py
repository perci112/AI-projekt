from PIL import Image
import os


def standardize_images(input_folder, output_folder, size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert('RGB')  # Konwersja do RGB
        img = img.resize(size)
        img.save(os.path.join(output_folder, img_name))