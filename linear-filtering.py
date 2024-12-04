from PIL import Image
import numpy as np
import os

def apply_filter(image_path, weights, scale, offset):
    # load image and convert to numpy array
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    # kernel size
    kernel_size = len(weights)
    pad = kernel_size // 2

    # pad image with zeros to handle borders
    padded_img = np.pad(img_array, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # output image array
    output_img = np.zeros_like(img_array, dtype=np.int32)

    # loop through each pixel
    for y in range(pad, padded_img.shape[0] - pad):
        print(f"Processing row {y}/{padded_img.shape[0] - pad - 1}")
        for x in range(pad, padded_img.shape[1] - pad):
            for c in range(3):  # RGB channels
                # apply filter
                region = padded_img[y-pad:y+pad+1, x-pad:x+pad+1, c]
                result = np.sum(region * weights) / scale + offset
                output_img[y-pad, x-pad, c] = np.clip(result, 0, 255)

    # convert result to a PIL image
    filtered_image = Image.fromarray(output_img.astype(np.uint8))
    return filtered_image

if __name__ == "__main__":
    # 3x3 filter example
    weights = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0]     
    ]
    
    scale = 1
    if scale == 0:
        print("Scale cannot be zero. Setting scale to 1.")
        scale = 1
    offset = 128

    image_path = "images/input/input_image2.jpg" 
    if not os.path.exists(image_path):
        print("Image file not found:", image_path)
        exit()
    
    filtered_image = apply_filter(image_path, weights, scale, offset)

    filtered_image.show()  # display image
    filtered_image.save("images/output/filtered_image.jpg")  # save filtered image
