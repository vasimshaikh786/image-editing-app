import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io
import re

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'edited_image' not in st.session_state:
    st.session_state.edited_image = None
if 'detected_numbers' not in st.session_state:
    st.session_state.detected_numbers = []
if 'replacement_mapping' not in st.session_state:
    st.session_state.replacement_mapping = {}

def detect_phone_numbers(image):
    """Detect phone numbers in the image using OCR and regex."""
    try:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Use pytesseract to get OCR data
        d = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        phone_numbers = []
        current_number = ""
        current_bbox = None
        text_color = (0, 0, 0)  # Default black color
        phone_pattern = re.compile(r'(\+?\d[\d\s\-\(\)]{7,}\d)')
        
        for i in range(len(d['text'])):
            text = d['text'][i].strip()
            if text:
                # Get text color from the original image
                if current_bbox:
                    region = pil_image.crop(current_bbox)
                    region_np = np.array(region)
                    if region_np.size > 0:
                        avg_color = np.mean(region_np.reshape(-1, 3), axis=0)
                        text_color = tuple(avg_color.astype(int))
                
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
    
    except Exception as e:
        st.error(f"Error in phone number detection: {str(e)}")
        return []

def replace_text_in_image(image, original_text, new_text, bbox, font_size, text_color):
    """Replace text in the image while maintaining formatting."""
    try:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image.copy()
        
        draw = ImageDraw.Draw(pil_image)
        font_size = int(font_size * 0.8)  # Adjusted size
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position
        text_width = draw.textlength(new_text, font=font)
        text_height = font_size
        x = bbox[0] + (bbox[2] - bbox[0] - text_width) / 2
        y = bbox[1] + (bbox[3] - bbox[1] - text_height) / 2
        
        # Draw background and text
        draw.rectangle(bbox, fill=(255, 255, 255))
        draw.text((x, y), new_text, font=font, fill=text_color)
        
        return pil_image
    
    except Exception as e:
        st.error(f"Error in text replacement: {str(e)}")
        return image

def generate_proxy_number(original):
    """Generate a proxy number that looks similar to the original."""
    digit_map = {'0':'5', '1':'7', '2':'8', '3':'9', '4':'6',
                 '5':'0', '6':'4', '7':'1', '8':'2', '9':'3'}
    return ''.join(digit_map.get(c, c) for c in original)

def main():
    st.title("📷 Image Tracking Number Editor")
    st.markdown("Upload an image to detect and replace phone numbers while maintaining the original formatting.")
    
    # Sidebar for upload
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is not None:
                    st.session_state.original_image = image
                    st.session_state.detected_numbers = detect_phone_numbers(image)
                    
                    for num in st.session_state.detected_numbers:
                        original_num = num['number']
                        if original_num not in st.session_state.replacement_mapping:
                            st.session_state.replacement_mapping[original_num] = generate_proxy_number(original_num)
                    
                    if st.session_state.detected_numbers:
                        st.success(f"Detected {len(st.session_state.detected_numbers)} phone numbers")
                    else:
                        st.warning("No phone numbers detected")
                else:
                    st.error("Failed to process the uploaded image")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Main content
    if st.session_state.original_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            try:
                annotated_image = st.session_state.original_image.copy()
                for num in st.session_state.detected_numbers:
                    bbox = num['bbox']
                    cv2.rectangle(annotated_image, 
                                 (int(bbox[0]), int(bbox[1])), 
                                 (int(bbox[2]), int(bbox[3])), 
                                 (0, 255, 0), 2)
                st.image(annotated_image, use_container_width=True, caption="Detected phone numbers")
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        
        with col2:
            st.subheader("Edit Phone Numbers")
            if st.session_state.detected_numbers:
                for num in st.session_state.detected_numbers:
                    with st.expander(f"Number: {num['number']}"):
                        new_num = st.text_input(
                            "Replacement",
                            value=st.session_state.replacement_mapping.get(num['number'], ""),
                            key=f"replace_{num['number']}"
                        )
                        st.session_state.replacement_mapping[num['number']] = new_num
                
                if st.button("Apply Changes"):
                    try:
                        edited_image = st.session_state.original_image.copy()
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
                    except Exception as e:
                        st.error(f"Error applying changes: {str(e)}")
            
            if st.session_state.edited_image is not None:
                st.subheader("Preview")
                st.image(st.session_state.edited_image, use_container_width=True, caption="Edited image")
                
                buf = io.BytesIO()
                edited_pil = Image.fromarray(cv2.cvtColor(st.session_state.edited_image, cv2.COLOR_BGR2RGB))
                edited_pil.save(buf, format="PNG")
                st.download_button(
                    "Download Edited Image",
                    buf.getvalue(),
                    "edited_image.png",
                    "image/png"
                )

if __name__ == "__main__":
    main()
