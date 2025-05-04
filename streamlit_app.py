import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageColor
import io
import re
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'edited_image' not in st.session_state:
    st.session_state.edited_image = None
if 'detected_numbers' not in st.session_state:
    st.session_state.detected_numbers = []
if 'replacement_mapping' not in st.session_state:
    st.session_state.replacement_mapping = {}
if 'font_properties' not in st.session_state:
    st.session_state.font_properties = {}

def detect_phone_numbers(image):
    """Detect phone numbers in the image using OCR and regex."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    phone_numbers = []
    current_number = ""
    current_bbox = None
    phone_pattern = re.compile(r'(\+?\d[\d\s\-\(\)]{7,}\d)')
    
    for i in range(len(d['text'])):
        text = d['text'][i].strip()
        if text:
            if phone_pattern.search(text):
                if not current_number:
                    current_bbox = (
                        d['left'][i], 
                        d['top'][i], 
                        d['left'][i] + d['width'][i], 
                        d['top'][i] + d['height'][i]
                    )
                    current_number = text
                else:
                    current_number += " " + text
                    current_bbox = (
                        min(current_bbox[0], d['left'][i]),
                        min(current_bbox[1], d['top'][i]),
                        max(current_bbox[2], d['left'][i] + d['width'][i]),
                        max(current_bbox[3], d['top'][i] + d['height'][i])
                    )
            else:
                if current_number:
                    # Store font properties for each detected number
                    region = image.crop(current_bbox)
                    region_np = np.array(region)
                    if region_np.size > 0:
                        avg_color = np.mean(region_np.reshape(-1, 3), axis=0)
                        text_color = tuple(avg_color.astype(int))
                    else:
                        text_color = (0, 0, 0)
                    
                    phone_numbers.append({
                        'number': current_number,
                        'bbox': current_bbox,
                        'confidence': d['conf'][i],
                        'font_size': d['height'][i],
                        'text_color': text_color
                    })
                    current_number = ""
                    current_bbox = None
    
    if current_number:
        phone_numbers.append({
            'number': current_number,
            'bbox': current_bbox,
            'confidence': d['conf'][i],
            'font_size': d['height'][i],
            'text_color': text_color
        })
    
    return phone_numbers

def replace_text_in_image(image, original_text, new_text, bbox, font_size, text_color):
    """Replace text in the image while maintaining exact formatting."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(image)
    
    # Calculate font size to match original
    font_size = int(font_size * 0.8)  # Adjust for better visual match
    
    try:
        # Try to find a matching font
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Calculate text position (centered in the bbox)
    text_width = draw.textlength(new_text, font=font)
    text_height = font_size
    x = bbox[0] + (bbox[2] - bbox[0] - text_width) / 2
    y = bbox[1] + (bbox[3] - bbox[1] - text_height) / 2
    
    # Draw a background rectangle matching the original
    draw.rectangle(bbox, fill=(255, 255, 255))
    
    # Draw the new text with original color and size
    draw.text((x, y), new_text, font=font, fill=text_color)
    
    return image

def generate_proxy_number(original):
    """Generate a proxy number that looks similar to the original."""
    digit_map = {'0':'5', '1':'7', '2':'8', '3':'9', '4':'6',
                 '5':'0', '6':'4', '7':'1', '8':'2', '9':'3'}
    
    proxy = []
    for char in original:
        if char in digit_map:
            proxy.append(digit_map[char])
        else:
            proxy.append(char)
    return ''.join(proxy)

def display_image_with_boxes(image, detected_numbers):
    """Display image with bounding boxes around detected numbers."""
    if isinstance(image, np.ndarray):
        img_display = image.copy()
    else:
        img_display = np.array(image)
        if img_display.ndim == 2:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        else:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
    
    for num in detected_numbers:
        bbox = num['bbox']
        cv2.rectangle(img_display, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 255, 0), 2)
        
        cv2.putText(img_display, num['number'], 
                   (int(bbox[0]), int(bbox[1]) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return img_display

# Main app
def main():
    st.title("📷 Image Tracking Number Editor")
    st.markdown("Upload an image to detect and replace phone numbers while maintaining the original formatting.")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.original_image = image
            
            st.session_state.detected_numbers = detect_phone_numbers(image)
            
            for num in st.session_state.detected_numbers:
                original_num = num['number']
                if original_num not in st.session_state.replacement_mapping:
                    st.session_state.replacement_mapping[original_num] = generate_proxy_number(original_num)
            
            st.success(f"Detected {len(st.session_state.detected_numbers)} phone numbers in the image.")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.original_image is not None:
            st.subheader("Original Image")
            annotated_image = display_image_with_boxes(
                st.session_state.original_image, 
                st.session_state.detected_numbers
            )
            st.image(annotated_image, use_container_width=True, caption="Detected phone numbers highlighted")
    
    with col2:
        if st.session_state.original_image is not None:
            st.subheader("Edit Phone Numbers")
            
            if not st.session_state.detected_numbers:
                st.warning("No phone numbers detected in this image.")
            else:
                for num in st.session_state.detected_numbers:
                    original_num = num['number']
                    with st.expander(f"Number: {original_num}"):
                        new_num = st.text_input(
                            f"Replacement for {original_num}",
                            value=st.session_state.replacement_mapping.get(original_num, ""),
                            key=f"replace_{original_num}"
                        )
                        st.session_state.replacement_mapping[original_num] = new_num
                
                if st.button("Apply Changes"):
                    edited_image = st.session_state.original_image.copy()
                    if isinstance(edited_image, np.ndarray):
                        edited_image = Image.fromarray(cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB))
                    
                    for num in st.session_state.detected_numbers:
                        original_num = num['number']
                        new_num = st.session_state.replacement_mapping.get(original_num, original_num)
                        edited_image = replace_text_in_image(
                            edited_image,
                            original_num,
                            new_num,
                            num['bbox'],
                            num['font_size'],
                            num['text_color']
                        )
                    
                    st.session_state.edited_image = edited_image
                    st.success("Changes applied successfully!")
    
    if st.session_state.edited_image is not None:
        st.subheader("Preview")
        preview_col1, preview_col2 = st.columns(2)
        
        with preview_col1:
            st.markdown("**Original Image**")
            st.image(st.session_state.original_image, use_container_width=True)
        
        with preview_col2:
            st.markdown("**Edited Image**")
            st.image(st.session_state.edited_image, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Download Edited Image")
        buf = io.BytesIO()
        if isinstance(st.session_state.edited_image, np.ndarray):
            edited_pil = Image.fromarray(cv2.cvtColor(st.session_state.edited_image, cv2.COLOR_BGR2RGB))
        else:
            edited_pil = st.session_state.edited_image
        edited_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Download Edited Image",
            data=byte_im,
            file_name="edited_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
