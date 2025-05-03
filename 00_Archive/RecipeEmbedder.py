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

class RecipeEmbedder:
    def __init__(self, logger, model_name="sentence-transformers/all-mpnet-base-v2"):
        """Initialize the embedder with the specified model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device: {self.device}')
        model_path = "sentence-transformers/all-mpnet-base-v2"
        #model_path = "all-mpnet-base-v2"
        
        # Force offline mode
        #os.environ["HF_HUB_OFFLINE"] = "1"
        #os.environ["TRANSFORMERS_OFFLINE"] = "1"
        #os.environ["TRANSFORMERS_NO_ADVISORY_LOCKING"] = "1"
        #self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        #self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.logger = logger

        self.model.eval()
        #print("Loaded completely offline from:", cache_dir)
        
    
    def mean_pooling(model_output, attention_mask):
        
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / input_mask_expanded.sum(1).clamp(min=1e-9)
        return pooled_output
    
    def create_text_embeddings(self, texts, batch_size=32):
        """Create embeddings for text data"""
        self.logger.info(f"Creating embeddings for {len(texts)} texts with batch size {batch_size}...")
        
        all_embeddings = []
        
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            # Prepare inputs
            encoded_input = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Only pass what the model expects
            model_inputs = {
                "input_ids": encoded_input["input_ids"],
                "attention_mask": encoded_input["attention_mask"]
            }

            with torch.no_grad():
                model_output = self.model(**model_inputs)
                
            # Mean pooling
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = embeddings.cpu().numpy()
            
            all_embeddings.extend(embeddings)

        return all_embeddings
    def create_nutrition_embeddings(self, nutrition_vectors):
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
        
        # Normalize the combined features
        scaler = MinMaxScaler()
        normalized_nutrition = scaler.fit_transform(np.nan_to_num(combined_nutrition))
        
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
        """
        Create a query embedding of length 1177 to match the recipe full_embedding.
        Allows optional user-provided nutrition goals.

        Parameters:
        - ingredients: list of strings
        - dietary_preference: string
        - nutrition_goals: dict with optional keys:
            'energy', 'fat', 'saturates', 'carbohydrate', 'sugars',
            'fibre', 'protein', 'salt'

        Returns:
        - query_embedding: np.ndarray of shape (1177,)
        """
        self.logger.info("Creating query embedding...")

        # --- Step 1: Dietary preference embedding (768)
        preference_emb = self.create_text_embeddings([dietary_preference])[0]
        

        # --- Step 2: Nutrition embedding (15)
        if nutrition_goals is not None:
            # Ordered features
            keys = ['energy', 'fat', 'saturates', 'sugars', 'protein', 'salt']
            raw_nutrition = [nutrition_goals.get(k, 0) for k in keys]
            derived_nutrition = self.create_nutrition_embeddings([raw_nutrition])[0]
        else:
            derived_nutrition = np.zeros(15)

        # --- Step 3: FSA (keep dummy unless explicitly provided later) (9)
        dummy_fsa = np.zeros(9)

        # --- Step 4: Ingredient embedding (385)
        ingredient_text = " ".join(ingredients)
        ingredient_emb = self.create_text_embeddings([ingredient_text])[0]
        ingredient_emb = ingredient_emb[:385]  # Truncate
        if len(ingredient_emb) < 385:
            ingredient_emb = np.concatenate([ingredient_emb, np.zeros(385 - len(ingredient_emb))])

        # --- Combine all
        query_embedding = np.concatenate([
            preference_emb,
            derived_nutrition,
            dummy_fsa,
            ingredient_emb
        ])
        # DEBUG: Print shapes of each component
        print("preference_emb:", preference_emb.shape)
        print("derived_nutrition:", np.array(derived_nutrition).shape)
        print("dummy_fsa:", dummy_fsa.shape)
        print("ingredient_emb:", ingredient_emb.shape)
        print("Total combined:", query_embedding.shape)
        assert query_embedding.shape[0] == 1177, f"Expected embedding size 1177 but got {query_embedding.shape[0]}"
        return query_embedding.astype("float32")