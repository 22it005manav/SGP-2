import numpy as np
import cv2
from cv2 import dnn
import glob


proto_file = 'Model/colorization_deploy_v2.prototxt'
model_file = 'Model/colorization_release_v2.caffemodel'
hull_pts = 'Model/pts_in_hull.npy'

net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)

#for cluster
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

#  For output resolution (width x height)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

#   For image
def colorize_full_image(img):
    
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    
    L_original = cv2.split(lab_img)[0]
    colorized = np.concatenate((L_original[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

   
    colorized = (255 * colorized).astype("uint8")

    return colorized


def colorize_selected_region(img):
   
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_3ch = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR) 

    roi = cv2.selectROI("Select Object to Colorize", grayscale_3ch)
    if roi == (0, 0, 0, 0):  # No selection made
        print("No ROI selected, skipping image.")
        return None

    x, y, w, h = roi
    selected_region = img[y:y + h, x:x + w]

   
    scaled = selected_region.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

   
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

   
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (w, h))

   
    L_original = cv2.split(lab_img)[0]
    colorized = np.concatenate((L_original[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

  
    colorized = (255 * colorized).astype("uint8")

    return colorized, (x, y, w, h)

image_directory = './images/*'
image_paths = glob.glob(image_directory)

cv2.namedWindow("Image Colorization", cv2.WINDOW_NORMAL)


for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        continue

  
    img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))

   
    full_colorized_img = colorize_full_image(img_resized)

    
    selected_colorized_img, roi_coords = colorize_selected_region(img_resized)

    if selected_colorized_img is not None:
        
        half_height = TARGET_HEIGHT
        half_width = TARGET_WIDTH // 2
        output_canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype="uint8")

       
        output_canvas[:, :half_width] = cv2.resize(full_colorized_img, (half_width, half_height))

      
        x, y, w, h = roi_coords
        roi_resized = cv2.resize(selected_colorized_img, (half_width, half_height))  # Resize the ROI to fit half the screen
        output_canvas[:, half_width:] = roi_resized

       
        cv2.imshow("Image Colorization", output_canvas)
        cv2.waitKey(0)  


cv2.destroyAllWindows()
