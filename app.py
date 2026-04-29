import streamlit as st
from openai import OpenAI
import base64
import os

# ==============================
# 🔑 OPENROUTER SETUP
# ==============================
client = OpenAI(
    api_key=st.secrets["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"],
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "openai/gpt-4o-mini"

# ==============================
# 🧠 PROMPTS
# ==============================
def prompt_step1_objects():
    return """
Extract ALL visible objects from the image.

Include:
- furniture, electronics, fixtures, decor
- small items

Return:
OBJECTS:
- item (count)
"""

def prompt_step2_structure(objects_text):
    return f"""
Add missing structural elements:

{objects_text}

Always include:
- Structural walls
- Flooring
- Ceiling
- Electrical points

Return updated list.
"""

def prompt_step3_classify(objects_text):
    return f"""
Classify into STRUCTURE or CONTENT:

{objects_text}

STRUCTURE = fixed
CONTENT = movable

Return:

STRUCTURE:
- item (count)

CONTENT:
- item (count)
"""

# ==============================
# 📸 IMAGE → BASE64
# ==============================
def encode_image(file):
    return base64.b64encode(file.read()).decode("utf-8")

# ==============================
# 🤖 AGENT PIPELINE
# ==============================
def run_agent(uploaded_files):
    results = []

    for file in uploaded_files:
        st.write(f"📸 Processing: {file.name}")

        base64_img = encode_image(file)

        # STEP 1
        step1 = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_step1_objects()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }]
        )
        objects = step1.choices[0].message.content

        # STEP 2
        step2 = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt_step2_structure(objects)}]
        )
        updated = step2.choices[0].message.content

        # STEP 3
        step3 = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt_step3_classify(updated)}]
        )
        final = step3.choices[0].message.content

        results.append((file.name, final))

    return results

# ==============================
# 🖥️ STREAMLIT UI
# ==============================
st.set_page_config(page_title="AI Home Inventory", layout="wide")

st.title("🏠 AI Home Inventory Generator")

st.markdown("Upload images of your home/room to generate structured inventory.")

uploaded_files = st.file_uploader(
    "📂 Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 Run Analysis"):
        with st.spinner("Processing images..."):
            results = run_agent(uploaded_files)

        st.success("✅ Analysis Complete!")

        for name, result in results:
            st.subheader(f"📸 {name}")
            st.text(result)
            st.markdown("---")
