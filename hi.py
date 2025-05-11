import os
import cv2

if __name__ == '__main__':
    
    # read all training images and print out their shapes (in set)
    entry = os.path.join("hw4_release_dataset", "train", "degraded")
    images_size = set()
    
    for filename in os.listdir(entry):
        if filename.endswith(".png"):
            
            image_path = os.path.join(entry, filename)
            degrad_img = cv2.imread(image_path)
            degrad_img = cv2.cvtColor(degrad_img, cv2.COLOR_BGR2RGB)
            original_size = tuple( degrad_img.shape[:2] )
            images_size.add(original_size)

    print("training images size: ", images_size)

    # read all test images and print out their shapes (in set)
    entry = os.path.join("hw4_release_dataset", "test", "degraded")
    images_size = set()
    for filename in os.listdir(entry):
        if filename.endswith(".png"):
            
            image_path = os.path.join(entry, filename)
            degrad_img = cv2.imread(image_path)
            degrad_img = cv2.cvtColor(degrad_img, cv2.COLOR_BGR2RGB)
            original_size = tuple( degrad_img.shape[:2] )
            images_size.add(original_size)
    print("test images size: ", images_size)