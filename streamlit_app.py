import streamlit as st
import os
import random
import cv2
import numpy as np
from PIL import Image
import re
import io
import google.generativeai as genai

# === Configuration ===
GENAI_API_KEY = 'AIzaSyAsjqea6U5Ejcu2byxaUsm5YqmawD3AT_Y'
EXTRACTED_IMAGES_FOLDER = "./extracted_images/"
ANNOTATED_FOLDER = "./annotated_images/"
os.makedirs(EXTRACTED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Sketch Annotator", layout="wide")
st.title("🎨 Gemini Sketch Annotator")

# === Utility Functions ===

def adjust_box(box, scale=1.6, shift_x=80, shift_y=80):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    new_w = w * scale
    new_h = h * scale
    return [
        int(cx - new_w / 2 + shift_x),
        int(cy - new_h / 2 + shift_y),
        int(cx + new_w / 2 + shift_x),
        int(cy + new_h / 2 + shift_y)
    ]

def generate_prompt(width, height):
    return f"""
You are analyzing a sketch image that is {width}x{height} pixels.

First, identify the overall object and its spatial layout:
Object - [Brief description of the object and its overall spatial arrangement]

Then, identify 1 to 3 key elements. For each element, provide:
[Element Name] - [x1, y1, x2, y2] - [Brief description of function and location]

Use exact pixel coordinates. The coordinates should outline the actual object and its elements boundaries.
Make sure the coordinates are within the image dimensions (0-{width} for x, 0-{height} for y).

Example format (follow exactly):
Object - A cat sitting upright, facing forward

Head - [80, 40, 160, 120] - Rounded head with pointed ears at the top  
Body - [70, 120, 170, 300] - Oval-shaped torso extending down from the head  
Tail - [160, 250, 220, 350] - Curved tail extending from the lower right side of the body
"""

def annotate_image(pil_img, gemini_text, enhanced=True):
    image = np.array(pil_img.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,165,0)]
    used_positions = []
    elements = []

    for line in gemini_text.strip().split("\n"):
        if not line.lower().startswith("object -"):
            match = re.match(r"(.+?) - \[(\d+), (\d+), (\d+), (\d+)\] - (.+)", line)
            if match:
                name = match.group(1).strip()
                coords = list(map(int, match.group(2,3,4,5)))
                desc = match.group(6).strip()
                elements.append((name, coords, desc))

    for i, (name, coords, desc) in enumerate(elements):
        color = colors[i % len(colors)]
        if enhanced:
            coords = adjust_box(coords)

        x1, y1, x2, y2 = coords
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{name}: {desc}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y2 + text_size[1] + 10

        cv2.rectangle(annotated,
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      color, -1)
        cv2.putText(annotated, text, (text_x, text_y), font, font_scale, (0,0,0), thickness)
        used_positions.append((text_x, text_y))

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated)

def run_analysis(pil_image, use_adjusted=True):
    width, height = pil_image.size
    prompt = generate_prompt(width, height)
    response = model.generate_content([prompt, pil_image])
    gemini_text = response.text

    annotated = annotate_image(pil_image, gemini_text, enhanced=use_adjusted)
    return gemini_text, annotated

def get_random_image(folder):
    all_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(folder)
        for f in filenames if f.lower().endswith(('.jpg','.png','.jpeg'))
    ]
    return random.choice(all_files) if all_files else None

# === UI ===

col1, col2 = st.columns(2)

with col1:
    if st.button("🔁 Analyze Random Image from Folder"):
        random_path = get_random_image(EXTRACTED_IMAGES_FOLDER)
        if random_path:
            pil_img = Image.open(random_path).convert("RGB")
            st.image(pil_img, caption="🎲 Randomly Selected Image", use_column_width=True)
            with st.spinner("Analyzing with Gemini..."):
                text, output = run_analysis(pil_img)
            st.subheader("📌 Gemini Output")
            st.code(text)
            st.image(output, caption="🖍️ Annotated Result")
        else:
            st.warning("No images found in extracted_images/.")

with col2:
    uploaded = st.file_uploader("📥 Drag and drop your own sketch image", type=['jpg','png','jpeg'])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="📂 Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing with Gemini..."):
            text, output = run_analysis(pil_img)
        st.subheader("📌 Gemini Output")
        st.code(text)
        st.image(output, caption="🖍️ Annotated Result")

        img_bytes = io.BytesIO()
        output.save(img_bytes, format="PNG")
        st.download_button("💾 Download Annotated Image", img_bytes.getvalue(), "annotated.png")

        st.download_button("📄 Download Gemini Output", text, "description.txt")
