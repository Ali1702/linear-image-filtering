from PIL import Image
import numpy as np
import os

def parse_weights(size, non_zero_entries):
    weights = np.zeros((size, size), dtype=int)
    for row, col, value in non_zero_entries:
        weights[row - 1, col - 1] = value  # convert 1-based to 0-based indexing
    return weights

def apply_filter(image_path, weights, scale, offset):
    # load image and convert to numpy array
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.int32)

    # kernel size
    kernel_size = weights.shape[0]
    pad = kernel_size // 2

    # pad image with zeros to handle borders
    padded_img = np.pad(img_array, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # initialize the output image
    output_img = np.zeros_like(img_array, dtype=np.int32)

    # vectorized convolution for each channel
    for c in range(3):  # RGB channels
        # extract channel and convolve
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                output_img[:, :, c] += weights[dy, dx] * padded_img[dy:dy+img_array.shape[0], dx:dx+img_array.shape[1], c]
    
    # apply scale and offset
    output_img = output_img // scale + offset
    output_img = np.clip(output_img, 0, 255)

    # convert result to a PIL image
    return Image.fromarray(output_img.astype(np.uint8))

if __name__ == "__main__":
    # user input for filter parameters
    print("Welcome to the Image Filter Tool!")

    # input image path
    image_path = "images/input/input_image2.jpg" # change this accordingly
    if not os.path.exists(image_path):
        print("Image file not found:", image_path)
        exit()

    # input kernel size
    while True:
        try:
            kernel_size = int(input("Enter the kernel size (3 for 3x3, 5 for 5x5): ").strip())
            if kernel_size not in [3, 5]:
                raise ValueError("Kernel size must be 3 or 5.")
            break
        except ValueError as e:
            print("Invalid input:", e)

    # input non-zero weights
    print(f"Enter non-zero weights for the {kernel_size}x{kernel_size} kernel.")
    print("Format: row,col,value (1-based indexing), or press Enter to finish.")
    non_zero_entries = []
    while True:
        entry = input("Enter a non-zero weight (or press Enter to finish): ").strip()
        if not entry:
            break
        try:
            row, col, value = map(int, entry.split(","))
            if not (1 <= row <= kernel_size and 1 <= col <= kernel_size):
                raise ValueError("Row and column must be within kernel size.")
            non_zero_entries.append((row, col, value))
        except ValueError as e:
            print("Invalid input:", e)

    # generate weights matrix
    weights = parse_weights(kernel_size, non_zero_entries)
    print("Generated weight matrix:")
    print(weights)

    # input scale and offset
    while True:
        try:
            scale = int(input("Enter the scale value (positive integer): ").strip())
            if scale <= 0:
                raise ValueError("Scale must be greater than 0.")
            break
        except ValueError as e:
            print("Invalid input:", e)

    while True:
        try:
            offset = int(input("Enter the offset value (integer): ").strip())
            break
        except ValueError:
            print("Invalid input. Offset must be an integer.")

    # apply the filter
    filtered_image = apply_filter(image_path, weights, scale, offset)

    # save and display the filtered image
    output_path = "filtered_image.jpg"
    filtered_image.save(output_path)
    print("Filtered image saved to:", output_path)
    filtered_image.show()
