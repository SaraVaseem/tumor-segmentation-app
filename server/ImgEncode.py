import base64
import labelme

def image_data(img_path):
    print(f"Received image path: {img_path}")  # Debug: Print the received path
    try:
        # Load the image file
        data = labelme.LabelFile.load_image_file(img_path)
        # Encode the image data as base64
        image = base64.b64encode(data).decode('utf-8')
        print("Image encoded successfully")  # Debug: Confirm encoding worked
        return image
    except Exception as e:
        print(f"Error: {e}")  # Debug: Print any errors
        return None

# Call the function and store the result in a variable
encoded_image = image_data(path)