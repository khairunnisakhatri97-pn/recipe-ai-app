from pytube import Search
import cv2
import numpy as np
import joblib
import streamlit as st
import torch
import json
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import os


st.sidebar.title("Navigator Menu")
menu = st.sidebar.radio("Go to", ["Search recipe", "How it make"])


if menu == "Search recipe":

    st.set_page_config(
        page_title="AI Recipe Recommender",
        page_icon="🍳",
        layout="centered"
    )

    
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "food_type" not in st.session_state:
        st.session_state.food_type = None
    if "ingredients" not in st.session_state:
        st.session_state.ingredients = []

    def set_bg_color(color):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-color: {color};
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    
    if st.session_state.step == 1:
        set_bg_color("#f5f5dc")
        st.title("🍽️ Step 1: What kind of food would you like?")

        food_type = st.selectbox(
            "Select food type:",
            [
                "Sweet (Meetha)",
                "Salty (Namkeen)",
                "Spicy (Teez Masalaydar)",
                "Healthy (Sehatmand)",
                "Fast Food"
            ]
        )

        if st.button("➡️ Next Step"):
            st.session_state.food_type = food_type
            st.session_state.step = 2
            st.rerun()

    
    elif st.session_state.step == 2:

        food_type = st.session_state.food_type

        colors = {
            "Sweet": "#ffe4e1",
            "Salty": "#fff8dc",
            "Spicy": "#ffcccb",
            "Healthy": "#e8f5e9",
            "Fast Food": "#e0f7fa"
        }
        for key, color in colors.items():
            if key in food_type:
                set_bg_color(color)

        st.title(f"✍️ Step 2: Write Ingredients for {food_type}")
    

        
        user_input = st.text_input(
            "Ingredients (comma separated):",
            placeholder="e.g. chicken, onion, garlic, tomato"
        )

        ingredients = [
            ing.strip().title()
            for ing in user_input.split(",")
            if ing.strip()
        ]

        
        if "Sweet" in food_type:
            recipe_type = "Sweet Dessert Recipe"
        elif "Salty" in food_type:
            recipe_type = "Namkeen Snack Recipe"
        elif "Spicy" in food_type:
            recipe_type = "Spicy Curry Recipe"
        elif "Healthy" in food_type:
            recipe_type = "Healthy Diet Recipe"
        else:
            recipe_type = "Fast Food Recipe"

        if ingredients:
            st.success("✅ Ingredients you entered:")
            for ing in ingredients:
                st.write(f"• {ing}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅️ Back"):
                st.session_state.step = 1
                st.rerun()

        with col2:
            if st.button("➡️ Show Recipes"):
                if not ingredients:
                    st.warning("⚠️ Please write at least one ingredient")
                else:
                    st.session_state.ingredients = ingredients
                    st.session_state.recipe_type = recipe_type
                    st.session_state.step = 3
                    st.rerun()

    
    elif st.session_state.step == 3:

        food_type = st.session_state.food_type
        recipe_type = st.session_state.recipe_type
        ingredients = st.session_state.ingredients

        st.title(f"🍛 Step 3: Popular {food_type} Recipes")

        query = f"most popular {recipe_type} with {' '.join(ingredients)}"

        st.subheader(f"🔍 Searching YouTube for:")
        st.code(query)

        try:
            search = Search(query)
            results = search.results[:6]

            if results:
                for video in results:
                    st.video(video.watch_url)
                    st.write(video.title)
            else:
                st.error("No recipe found.")
        except Exception as e:
            st.error(str(e))

        if st.button("🔄 Start Over"):
            st.session_state.step = 1
            st.rerun()


elif menu == "How it make":

    MODEL_DIR = "model_artifacts"
    MODEL_PATH = os.path.join(MODEL_DIR, "food_model_final.pth")
    MLB_PATH = os.path.join(MODEL_DIR, "mlb.pkl")

    st.title("🍽️ Food Ingredient Prediction App")
    st.write("Upload food image and AI will predict ingredients")

    mlb = joblib.load(MLB_PATH)
    ingredients = list(mlb.classes_)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ingredients))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    uploaded = st.file_uploader(
        "Upload Food Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        inp = transform(img).unsqueeze(0)

        with torch.no_grad():
            preds = torch.sigmoid(model(inp)).numpy()[0]

        binary = (preds >= 0.5).astype(int)
        predicted = mlb.inverse_transform(np.array([binary]))[0]

        

        st.subheader("Predicted Ingredients")
        if predicted:
            for ing in predicted:
                st.write(f"✔️ {ing}")
        else:
            st.write("No ingredient detected")
            st.info("Run with: streamlit run app.py")

        
    


