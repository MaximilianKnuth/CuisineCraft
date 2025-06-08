print('Importing Start')
import os
import sys
print(sys.version)         # full version string
print(sys.version_info)    # tuple of (major, minor, micro, ...)
import pandas as pd
import numpy as np
import torch
import re
import ast
import json
import warnings
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import faiss
import logging
import joblib

class RecipeEmbedder:
    def __init__(self, logger, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device='cpu')
        self.logger = logger
        
    
   
    
    def create_text_embeddings(self, texts, batch_size=32, mode='train',show_progress_bar=True):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False, # optional: L2-normalize embeddings,
            device=self.device,
            num_workers=0 
        )
        
    def create_ingredient_embeddings(self, ingredients_structured):
        """
        Convert each recipe’s `ingredients_structured` list of dicts into a
        384-dimension MiniLM embedding.

        Parameters
        ----------
        ingredients_structured : Iterable[list[dict]]
            Each element looks like:
            [{'ingredient': 'yogurt, greek, plain, nonfat', ...}, …]

        Returns
        -------
        List[List[float]]
            A 384-dim vector (MiniLM) for every recipe.
        """
        embeddings = []
        for recipe in ingredients_structured:
            # Combine ingredient names into one string
            text = ", ".join(
                ing.get('ingredient', '') for ing in recipe if isinstance(ing, dict)
            )
            #print(text)
            #print('next text:')
            vec = self.create_text_embeddings([text],show_progress_bar=False)[0]   # → shape (384,)
            embeddings.append(vec.tolist())
        return embeddings
    
    def create_nutrition_embeddings(self, nutrition_vectors, mode='train'):
        """Create enhanced nutrition embeddings with derived features"""
        self.logger.info("Creating enhanced nutrition embeddings...")
        
        # Convert list of lists to numpy array
        nutrition_array = np.array(nutrition_vectors)
        
        # Create derived nutritional features
        derived_features = []
        for row in nutrition_array:
            # Get values using standard position in our nutrition keys
            energy = row[0] if len(row) > 0 else 0
            fat = row[1] if len(row) > 1 else 0
            saturates = row[2] if len(row) > 2 else 0
            carbohydrate = row[3] if len(row) > 3 else 0
            sugars = row[4] if len(row) > 4 else 0
            fibre = row[5] if len(row) > 5 else 0
            protein = row[6] if len(row) > 6 else 0
            salt = row[7] if len(row) > 7 else 0
            
            # Avoid division by zero
            protein = max(protein, 0.1)
            carbohydrate = max(carbohydrate, 0.1)
            
            # Calculate nutritional ratios that are meaningful for dietary preferences
            fat_protein_ratio = fat / protein
            carb_protein_ratio = carbohydrate / protein
            sugar_carb_ratio = sugars / carbohydrate
            sat_fat_ratio = saturates / max(fat, 0.1)
            
            # Caloric distribution (percentage of calories from each macronutrient)
            total_calories = (protein * 4) + (carbohydrate * 4) + (fat * 9)
            total_calories = max(total_calories, 1)  # Avoid division by zero
            protein_cal_pct = (protein * 4) / total_calories
            carb_cal_pct = (carbohydrate * 4) / total_calories
            fat_cal_pct = (fat * 9) / total_calories
            
            # Additional nutritional indicators
            fibre_carb_ratio = fibre / max(carbohydrate, 0.1)
            salt_per_100kcal = salt / max(energy/100, 0.1)
            
            # Combine into derived features
            derived = [
                fat_protein_ratio, carb_protein_ratio, sugar_carb_ratio, sat_fat_ratio,
                protein_cal_pct, carb_cal_pct, fat_cal_pct,
                fibre_carb_ratio, salt_per_100kcal
            ]
            
            derived_features.append(derived)
        
        # Combine original and derived features
        combined_nutrition = np.concatenate([nutrition_array, np.array(derived_features)], axis=1)
        
        if mode=='train': 
            self.logger.info("Using new scaler...")
            # Normalize the combined features
            scaler = MinMaxScaler()
            normalized_nutrition = scaler.fit_transform(np.nan_to_num(combined_nutrition))
            self.logger.info("Saving new scaler...")
            joblib.dump(scaler, 'nutrition_scaler.pkl')
        else: 
            scaler = joblib.load('nutrition_scaler.pkl')
            self.logger.info("Using pretrained scaler...")
            # Assume new_nutrition_data is your new data (must have same number of features)
            normalized_nutrition = scaler.transform(combined_nutrition)
        
        return normalized_nutrition.tolist()
    
    def create_fsa_embeddings(self, fsa_vectors):
        """Create enhanced FSA embeddings"""
        self.logger.info("Creating enhanced FSA embeddings...")
        
        # FSA values are already normalized (0=green, 1=amber, 2=red)
        fsa_array = np.array(fsa_vectors)
        
        # Create additional features based on overall healthiness
        enhanced_fsa = []
        for row in fsa_array:
            # Calculate average traffic light value (lower is better/healthier)
            avg_value = np.mean(row)
            
            # Count of each traffic light color
            green_count = sum(1 for x in row if x == 0)
            amber_count = sum(1 for x in row if x == 1)
            red_count = sum(1 for x in row if x == 2)
            
            # Calculate overall health score (0-1, higher is healthier)
            health_score = 1 - (avg_value / 2)
            
            # Add these derived features to the row
            enhanced_row = np.append(row, [avg_value, green_count, amber_count, red_count, health_score])
            enhanced_fsa.append(enhanced_row)
        
        return enhanced_fsa
    
    def combine_embeddings(self, text_embeddings, nutrition_embeddings, fsa_embeddings):
        """Combine all embeddings into a unified representation"""
        self.logger.info("Combining all embeddings...")
        
        combined_embeddings = []
        
        for i in range(len(text_embeddings)):
            text_emb = text_embeddings[i]
            nutrition_emb = nutrition_embeddings[i] if i < len(nutrition_embeddings) else np.zeros(len(nutrition_embeddings[0]) if nutrition_embeddings else 0)
            fsa_emb = fsa_embeddings[i] if i < len(fsa_embeddings) else np.zeros(len(fsa_embeddings[0]) if fsa_embeddings else 0)
            
            # Concatenate all embeddings
            combined = np.concatenate([text_emb, nutrition_emb, fsa_emb])
            combined_embeddings.append(combined)
        
        return combined_embeddings
    
    def create_query_embedding(self, ingredients, dietary_preference, nutrition_goals=None):
        self.logger.info("Creating query embedding...")

        # --- Step 1: Dietary_preference (384)
        #ingredients_string= ", ".join(ingredients)
        #dietpref_emb = self.create_text_embeddings([ingredients_string])[0]
        
        # --- Step 2: Ingredient embedding (384)
        ingredients_string= ", ".join(ingredients)
        ingredient_emb = self.create_text_embeddings([ingredients_string])[0]
        
        # --- Step 3: Nutrition embedding (15)
        #if nutrition_goals is not None:
        #    # Ordered features
        #    keys = ['energy', 'fat', 'saturates', 'sugars', 'protein', 'salt']
        #    raw_nutrition = [nutrition_goals.get(k, 0) for k in keys]
        #    derived_nutrition = self.create_nutrition_embeddings([raw_nutrition], mode='test')[0]
        #else:
        #    derived_nutrition = np.zeros(15)

        # --- Step 3: FSA (keep dummy unless explicitly provided later) (9)
        #dummy_fsa = np.zeros(9)

        

        # --- Combine all
        #query_embedding = np.concatenate([
        #    dietpref_emb,
        #    derived_nutrition,
        #    dummy_fsa,
        #    ingredient_emb
        # )
        # DEBUG: Print shapes of each component
        #print("preference_emb:", dietpref_emb.shape)
        #print("derived_nutrition:", np.array(derived_nutrition).shape)
        #print("dummy_fsa:", dummy_fsa.shape)
        #print("ingredient_emb:", ingredient_emb.shape)
        #print("Total combined:", query_embedding.shape)
        #assert query_embedding.shape[0] == 792, f"Expected embedding size 792 but got {query_embedding.shape[0]}"
        query_embedding = ingredient_emb
        return query_embedding.astype("float32")
    
    def create_query_embedding_new(self, ingredients):
                            #dietary_preference=dietary_preference,
                            #nutrition_goals=nutrition_goals,
                           #w_text=0.4, w_ingredients=0.4,
                           #w_nutrition=0.15, w_fsa=0.05):
        # Ingredient embedding
        ingr_emb = self.create_ingredient_embedding(ingredients)
        return ingr_emb

        """# Dietary preference (text) embedding
        dietary_text = ", ".join(ingredients) + ". " + dietary_preference]
        text_emb = self.create_text_embedding(dietary_text)

        # Nutrition goal embedding
        nutrition_vector = list(nutrition_goals.values())
        nut_emb = self.create_nutrition_embedding([nutrition_vector])[0]

        # FSA dummy zeros for now
        fsa_emb = np.zeros(9)

        # Normalize
        ingr_emb = ingr_emb / np.linalg.norm(ingr_emb)
        text_emb = text_emb / np.linalg.norm(text_emb)
        nut_emb = nut_emb / np.linalg.norm(nut_emb)
        fsa_emb = fsa_emb / np.linalg.norm(fsa_emb + 1e-9)

        # Weighted sum
        fused_emb = (
            w_text * text_emb +
            w_ingredients * ingr_emb +
            w_nutrition * nut_emb +
            w_fsa * fsa_emb
        )
        fused_emb = fused_emb / np.linalg.norm(fused_emb)

        return fused_emb.astype(np.float32)"""