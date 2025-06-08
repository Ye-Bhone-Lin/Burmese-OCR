import streamlit as st
from PIL import Image
import easyocr
import cv2
import numpy as np
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# <------------ Functions for text detection and slicing ------------>

# Load the easyocr model
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'], gpu=True) 
    # model = load_model('src/CNN.keras')
    model = None
    return model, reader

def detect_text(image, reader):
    
    # Convert the image to RGB format and then to NumPy arra
    image = image.convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_copy = image_np.copy()
    
    # Perform text detection
    # output : [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], text, confidence]
    results = reader.readtext(image_np, detail=1)
    
    # To store the bounding boxes and slices
    sliced_images_pil = []
    
    # Draw bounding boxes and slice the detected regions

    for detection in results:
        
        box_points = detection[0] # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        
        min_x, min_y, max_x, max_y = evalute_box(box_points, image_copy)
        
        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)

        cropped_image = image_copy[min_y:max_y, min_x:max_x]

        sliced_images_pil.append(Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)))
        
        # Draw the bounding box on the image
        cv2.rectangle(image_np, top_left, bottom_right, (0, 255, 0), 2)
        
    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)), sliced_images_pil
        
def evalute_box(box_points, image_copy):
    
        # Collect all the x-values and y-values into two separate lists
        all_x_coords = [int(point[0]) for point in box_points]
        all_y_coords = [int(point[1]) for point in box_points]

        # Find the minimum and maximum x and y coordinates
        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)

        # Ensure coordinates are within image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(image_copy.shape[1], max_x)
        max_y = min(image_copy.shape[0], max_y)

        return (min_x, min_y, max_x, max_y)
    
def preprocess_and_slice(pil_image_slice):
    
    image_np_bgr = cv2.cvtColor(np.array(pil_image_slice.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    precessed_image = prepare_image(image_np_bgr)
    
    contour_image, char_boxes = detect_char(precessed_image, image_np_bgr)
    
    sliced_images = []
    
    box = []
    
    for idx, (x, y, w, h) in enumerate(char_boxes):
        
        # Ensure the bounding box is within the image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(precessed_image.shape[1] - x, w)
        h = min(precessed_image.shape[0] - y, h)

        # Crop the character region from the original image
        char_image = precessed_image[y:y + h, x:x + w]
        
        # Convert to PIL Image for further processing if needed
        char_image_pil = Image.fromarray(cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB))

        # Append the sliced character image to the list
        sliced_images.append(char_image_pil)
        box.append("x={}, y={}, w={}, h={}".format(x, y, w, h))
    
    return Image.fromarray(cv2.cvtColor(precessed_image, cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)), sliced_images, box
    
def prepare_image(image):
    
   # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to smooth edges & reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive thresholding to dynamically adjust contrast
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Use a **very small** kernel to avoid character merging
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    # Apply **erosion first** to thin characters  slightly (prevent merging)
    processed_image = cv2.erode(binary, kernel, iterations=1)
    
    return processed_image

def detect_char(processed_image, original_image):

    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)
    
    # Extract bounding boxes for each contour
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Filter small noise by size
            bounding_boxes.append((x, y, w, h))
    
    # for i, bbox in enumerate(bounding_boxes):
    #     print(f"Box {i}: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    sorted_boxes = sorted(bounding_boxes, key=lambda b: (b[0], b[1]))
    
    merged_boxes = merge_boxes(sorted_boxes)
    
    return original_image, merged_boxes
    
# Merge nearby bounding boxes (grouping related characters)
def merge_boxes(boxes, vertical_threshold=20, horizontal_threshold=10):
        merged = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            x1, y1, w1, h1 = box1
            group = [box1]
            for j, box2 in enumerate(boxes):
                if j <= i or j in used:
                    continue
                x2, y2, w2, h2 = box2
                # Check if boxes are close enough to be merged
                if (
                    abs(x1 - x2) < horizontal_threshold
                    and abs(y1 - y2) < vertical_threshold
                ):
                    group.append(box2)
                    used.add(j)
            used.add(i)
            
            # Merge all grouped boxes into one bounding box
            gx = min([g[0] for g in group])
            gy = min([g[1] for g in group])
            gw = max([g[0] + g[2] for g in group]) - gx
            gh = max([g[1] + g[3] for g in group]) - gy
            merged.append((gx, gy, gw, gh))
            
        return merged
    
# <------------ Streamlit App ------------>

st.set_page_config(page_title="Image Upload App", layout="centered")

# Load the EasyOCR model
model, reader = load_models()

st.title("🖼️ Text detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        st.subheader("Original Uploaded Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)


    with st.spinner("Detecting text and slicing regions..."):

        # Perform text detection and slicing
        detected_image_pil, sliced_images_pil = detect_text(image, reader)


    with col2:

        st.subheader("Image with Detected Text")
        st.image(detected_image_pil, caption="Detected Text with Bounding Boxes", use_container_width=True)

    st.divider() 
    
    # Display sliced images with detected text
    if sliced_images_pil:
        st.subheader(f"Detected Text Slices ({len(sliced_images_pil)} regions found)")
        
        st.subheader("Full Segmentation Results")
        
        for i, slice_img in enumerate(sliced_images_pil):
            
            
            preprocessed, contour_image, char_slices, box = preprocess_and_slice(slice_img)


            # Create 3 columns
            col1, col2, col3 = st.columns(3)

            # Show sliced image
            with col1:
                st.image(slice_img, caption=f"Sliced {i+1}", use_container_width=True)

            # Show preprocessed image
            with col2:
                st.image(preprocessed, caption=f"Preprocessed {i+1}", use_container_width=True)

            # Show contuored images (list)
            with col3:
                st.image(contour_image, caption=f"{len(char_slices) } characters found", use_container_width=True)
                
                            
            num_columns = 5
            if char_slices:
                for i in range(0, len(char_slices), num_columns):
                    cols = st.columns(num_columns)
                    for j in range(num_columns):
                        if i + j < len(char_slices):
                            with cols[j]:
                                if char_slices[i + j] is not None:
                                    # st.text(f"type: {type(char_slices[i + j])}")
                                    char_img =char_slices[i + j]
                                    # img_array = np.array(char_img, dtype=np.float32)
                                    # img_array = cv2.resize(img_array, (224, 224))
                                    
                                    # # If grayscale, convert to RGB
                                    # if len(img_array.shape) == 2 or img_array.shape[2] == 1:
                                    #     img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

                                    # # Expand dims and preprocess
                                    # img_array = np.expand_dims(img_array, axis=0)
                                    # img_array = preprocess_input(img_array)  # <-- Use it here


                                    # pred = model.predict(img_array)
                                    # idx = np.argmax(pred[0])
                                    st.image(char_slices[i + j], use_container_width=True)
                                    # st.markdown("<h4 style='text-align: center; color: black;'>{}</h4>".format(idx), unsafe_allow_html=True)
                else:
                    st.write("No images found")
        
            
        else:
            st.write("No characters found")

    else:
        st.info("No text regions were detected or could be sliced.")

else:
    st.info("👈 Please upload an image file to begin.")
    
st.markdown("---")

st.markdown("Powered by EasyOCR and Streamlit.")