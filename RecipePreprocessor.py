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
# Data cleaning and preprocessing
class RecipePreprocessor:
    def __init__(self, logger):
        print('Initializing RecipePreprocessor')
        # Nutrition standard keys
        self.nutrition_keys = ['energy', 'fat', 'saturates', 'carbohydrate', 
                              'sugars', 'fibre', 'protein', 'salt']
        # FSA standard keys
        self.fsa_keys = ['fat', 'saturates', 'sugars', 'salt']
        # Color mapping for FSA traffic lights
        self.color_map = {'green': 0, 'amber': 1, 'red': 2, 'orange': 1}
        print('RecipePreprocessor initialized')
        self.logger = logger
        
    def clean_text(self, text):
        
        """Advanced text cleaning for title and instructions"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        # Replace multiple punctuation with single
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    
    def extract_fsa_features(self, fsa_dict):
        """Extract and normalize FSA traffic light values"""
        result = {}
        
        if not fsa_dict:
            return {key: 0 for key in self.fsa_keys}
        
        for key in self.fsa_keys:
            value = 0  # Default to green (0)
            for possible_key in [key, key.capitalize(), key.upper()]:
                if possible_key in fsa_dict:
                    # Get the color value
                    color = str(fsa_dict[possible_key]).lower()
                    # Map text colors to numeric values using our color map
                    value = self.color_map.get(color, 0)
                    break
            
            result[key] = value
        
        return result
    
    def preprocess_dataframe(self, df):
        """Apply all preprocessing steps to the dataframe"""
        self.logger.info("Starting data preprocessing...")
        
        # Clean text columns
        self.logger.info("Cleaning text data...")
        df['clean_title'] = df['title'].apply(self.clean_text)
        df['clean_instructions'] = df['instructions'].apply(self.clean_text)
        
        
        # Create text for embedding
        df['text_for_embedding'] = df['clean_title'] + " " + df['clean_instructions']
    
        # Parse structured data
        self.logger.info("Parsing nutrition data...")
        df['nutrition_dict'] = df['nutrition_per_100g']
        df['nutrition_features'] = df['nutrition_dict']
        
        self.logger.info("Parsing FSA traffic light data...")
        df['fsa_dict'] = df['fsa_lights_per_100g']
        df['fsa_features'] = df['fsa_dict'].apply(self.extract_fsa_features)
        
        # Convert nutrition features to separate columns
        nutrition_df = pd.DataFrame(df['nutrition_features'].tolist())
        if not nutrition_df.empty:
            # Add nutrition columns to main dataframe
            for col in nutrition_df.columns:
                df[f'nutrition_{col}'] = nutrition_df[col]
            
            # Scale nutrition features
            scaler = StandardScaler()
            nutrition_values = scaler.fit_transform(nutrition_df.fillna(0))
            df['nutrition_vector'] = [row.tolist() for row in nutrition_values]
        else:
            self.logger.warning("No nutrition data found, creating dummy vectors")
            df['nutrition_vector'] = [[0.0] * len(self.nutrition_keys)] * len(df)
        
        # Convert FSA features to separate columns
        fsa_df = pd.DataFrame(df['fsa_features'].tolist())
        if not fsa_df.empty:
            # Add FSA columns to main dataframe
            for col in fsa_df.columns:
                df[f'fsa_{col}'] = fsa_df[col]
            
            # FSA values are already normalized (0,1,2)
            df['fsa_vector'] = fsa_df.values.tolist()
        else:
            self.logger.warning("No FSA data found, creating dummy vectors")
            df['fsa_vector'] = [[0] * len(self.fsa_keys)] * len(df)
        
        return df