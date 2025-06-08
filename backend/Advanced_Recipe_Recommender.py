import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI   # pip install -U langchain-openai
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import os, requests, uuid, pathlib, logging
from RecipeEmbedder_new import RecipeEmbedder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import requests
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(                        # look for backend/.env explicitly
    Path(__file__).resolve().parent / ".env",
    override=False                  # keep any shell‑exported vars
)


DEEPSEEK_IMG_ENDPOINT = "https://api.deepseek.com/v1/images/generations"
DEEPSEEK_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_KEY")
HF_ENDPOINT = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static" / "dishes"
PLACEHOLDER = "/static/plate_placeholder.png"
SERVER_ORIGIN = os.getenv("SERVER_ORIGIN", "http://localhost:8000")
# Absolute dir on disk where files are saved
STATIC_DIR = Path(__file__).resolve().parent / "static" / "dishes"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN = os.getenv("HF_TOKEN")

from pathlib import Path

def generate_dish_image(title: str) -> str:
    if not HF_TOKEN:
        return "/static/plate_placeholder.png"

    payload = {
        "inputs": f"high‑res food photo of {title}, studio lighting, 4k",
        "options": {"wait_for_model": True}
    }
    try:
        r = requests.post(
            HF_ENDPOINT,
            headers={"Authorization": f"Bearer {HF_TOKEN}",
                     "Content-Type": "application/json"},
            json=payload, timeout=60
        )
        r.raise_for_status()
        fname = f"{uuid.uuid4().hex[:12]}.png"
        (STATIC_DIR / fname).write_bytes(r.content)       # ← writes file
        return f"/static/dishes/{fname}"                  # relative path
    except Exception as e:
        logger.error("HF gen failed: %s", e)
        return "/static/plate_placeholder.png"

class AdvancedRecipeRecommender:
    def __init__(self, recipe_data_path: str, faiss_index_path: str):
        """Initialize the recipe recommender system"""
        # Load data and models
        self.df = pd.read_pickle(recipe_data_path)
        self.df = self.df[['id',
                           'title',
                           'clean_instructions',
                           'instructions_full',
                           'ingredients_clean',
                           'ingredients_embedding',
                           'full_embedding',
                           'nutrition_per_100g']]
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.ingredient_weights = self.compute_tfidf_weights(self.df['ingredients_clean'])
        
        # Initialize embedding model
        #self.embedding_model = HuggingFaceEmbeddings(
        #    model_name="sentence-transformers/all-mpnet-base-v2"
        #)
        load_dotenv()  # this loads variables from .env into the environment

        # Initialize LLM with a placeholder API key
        self.api_key = None
        self.llm = None
        
        # Initialize reranker
        self.reranker = pipeline(
            "text-classification", 
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            padding=True,            # pad shorter sequences
            truncation=True,         # truncate longer ones
            max_length=512,  # keep default (usually 512)
            device=0 if faiss.get_num_gpus() > 0 else -1
        )
        self.recipe_embedder = RecipeEmbedder(logger=logger)

    def update_api_key(self, api_key: str):
        """Update the API key and reinitialize the LLM"""
        self.api_key = api_key
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model_name="deepseek-chat",
            temperature=0.2,
        )

    def compute_tfidf_weights(self, ingredients_series: pd.Series) -> Dict[str, float]:
        """Compute TF-IDF weights for all ingredients across recipes"""
        joined = ingredients_series.apply(lambda ings: " ".join(i.lower() for i in ings))
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(joined)
        vocab = vectorizer.get_feature_names_out()
        mean_scores = tfidf_matrix.mean(axis=0).A1
        return dict(zip(vocab, mean_scores))

    def ask_llm(self, prompt: str) -> str:
        if not self.llm:
            raise ValueError("API key not set. Please set the API key before making requests.")
        return self.llm.invoke([{"role": "user", "content": prompt}]).content
        
    def create_query_embedding(self, 
                              ingredients: List[str],
                              dietary_preference: str,
                              nutrition_goals: Optional[Dict[str, float]] = None) -> np.ndarray:
                                  
        return self.recipe_embedder.create_query_embedding(
            ingredients=ingredients,
            dietary_preference=dietary_preference,
            nutrition_goals=nutrition_goals)
        

    def retrieve_candidate_recipes(self, 
                                  query_vector: np.ndarray,
                                  top_k: int = 50) -> pd.DataFrame:
        """Retrieve initial candidates using FAISS"""
        # Perform similarity search
        query_vector = query_vector / np.linalg.norm(query_vector)
        D, I = self.faiss_index.search(query_vector.reshape(1, -1).astype('float32'), top_k)
        
        # Retrieve matching recipes
        results = self.df.iloc[I[0]].copy()
        results["similarity_score"] = D[0]
        
        return results
    
    def filter_by_ingredients(self, 
                             candidates: pd.DataFrame,
                             available_ingredients: List[str],
                             max_missing: int = 2) -> pd.DataFrame:
        """Filter recipes by available ingredients"""
        def ingredient_match_score(recipe_ingredients):
            available = set(i.lower() for i in available_ingredients)
            recipe_set = set(i.lower() for i in recipe_ingredients)
            
            # Count ingredients that can be matched
            matched = sum(1 for r in recipe_set if any(a in r for a in available))
            
            # Calculate match percentage
            match_percent = matched / len(recipe_set) if recipe_set else 0
            missing = len(recipe_set) - matched
            
            return match_percent, missing
        
        # Calculate match scores
        candidates['ingredient_match'], candidates['missing_count'] = zip(
            *candidates['ingredients_clean'].apply(ingredient_match_score)
        )
        
        # Filter by maximum missing ingredients
        filtered = candidates[candidates['missing_count'] <= max_missing].copy()
        
        # Add ingredient match to ranking
        filtered['combined_score'] = (
            filtered['similarity_score'] * 0.6 + 
            filtered['ingredient_match'] * 0.4
        )
        
        return filtered.sort_values('combined_score', ascending=False)
    
    def filter_by_ingredients_new(self, 
                          candidates: pd.DataFrame,
                          available_ingredients: List[str],
                          max_missing: int = 2) -> pd.DataFrame:
        """Improved ingredient filtering with TF-IDF-based weighting"""

        def token_overlap_score(user_ingredients, recipe_ingredients):
            user_tokens = set(word for ing in user_ingredients for word in ing.lower().split())
            recipe_tokens = set(word for ing in recipe_ingredients for word in ing.lower().split())
            intersection = user_tokens & recipe_tokens
            union = recipe_tokens | user_tokens
            return len(intersection) / len(union) if union else 0


        def compute_match(recipe_ingredients):
            jaccard = token_overlap_score(available_ingredients, recipe_ingredients)
            return jaccard

        results = candidates.copy()
        results['token_overlap'] = results['ingredients_clean'].apply(compute_match)
      

       

        # Final combined score
        results['combined_score'] = (
            0.6 * results['similarity_score'] +
            0.4 * results['token_overlap']
        )

        return results.sort_values('combined_score', ascending=False)

    def filter_by_dietary_preferences(self, 
                                   candidates: pd.DataFrame,
                                   preference_text: str) -> pd.DataFrame:
        """Apply dietary filters based on parsed free-text preferences"""

        pref = preference_text.lower()

        # --- Step 1: Parse preferences from text ---
        preferences = {
            'low_carb': any(kw in pref for kw in ['low carb', 'low carb', 'keto']),
            'high_protein': any(kw in pref for kw in ['high protein', 'protein-rich']),
            'low_fat': any(kw in pref for kw in ['low fat', 'low-fat']),
            'dairy_free': any(kw in pref for kw in ['dairy-free', 'no dairy']),
            'vegetarian': 'vegetarian' in pref,
            'vegan': 'vegan' in pref,
            'gluten_free': 'gluten-free' in pref or 'no gluten' in pref,
        }

        # Thresholds
        if preferences['low_carb']:
            preferences['max_energy'] = 1200
        if preferences['high_protein']:
            preferences['min_protein'] = 15
        if preferences['low_fat']:
            preferences['max_fat'] = 20

        # --- Step 2: Apply numeric filters ---
        filtered = candidates.copy()
        print(filtered['nutrition_per_100g'])
        
        if preferences.get('low_carb'):
            filtered = filtered[
                filtered['nutrition_per_100g'].apply(
                    lambda x: x.get('energy') < preferences['max_energy']
                )
            ]

        if preferences.get('high_protein'):
            filtered = filtered[
                filtered['nutrition_per_100g'].apply(
                    lambda x: x.get('protein', 0) > preferences['min_protein']
                )
            ]

        if preferences.get('low_fat'):
            filtered = filtered[
                filtered['nutrition_per_100g'].apply(
                    lambda x: x.get('fat', 100) < preferences['max_fat']
                )
            ]

        # --- Step 3: Return filtered DataFrame ---
        return filtered
    
    def rerank_with_llm(self, 
                       candidates: pd.DataFrame,
                       user_query: str,
                       top_k: int = 10) -> pd.DataFrame:
        """Rerank top candidates using a cross-encoder"""
        if len(candidates) == 0:
            return candidates
            
        if len(candidates) <= top_k:
            return candidates
        
        # Prepare data for reranking
        rerank_pairs = []
        for _, recipe in candidates.head(min(20, len(candidates))).iterrows():
            # Create description for each recipe
            recipe_desc = (
                f"Title: {recipe['title']}\n"
                f"Ingredients: {', '.join(recipe['ingredients_clean'])}\n"
                f"Instructions: {recipe['clean_instructions'][:200]}...\n"
                f"Nutrition: {recipe['nutrition_per_100g']}"
            )
            rerank_pairs.append((user_query, recipe_desc))
        
        # Get reranking scores
        rerank_scores = self.reranker(rerank_pairs)
        
        # Get top recipes based on reranking
        score_dict = {
            i: score['score'] 
            for i, score in enumerate(rerank_scores)
        }
        
        # Get top indices
        sorted_indices = sorted(
            score_dict.keys(), 
            key=lambda i: score_dict[i], 
            reverse=True
        )[:top_k]
        
        # Return reranked results
        return candidates.iloc[sorted_indices].copy()
    
    def generate_recipe_explanation(self, 
                                   recipe: pd.Series,
                                   user_query: str) -> str:
        """Generate an explanation for why this recipe was recommended"""
        prompt = f"""
        User is looking for: {user_query}
        
        You recommended the recipe "{recipe['title']}" which has these characteristics:
        - Ingredients: {', '.join(recipe['ingredients_clean'])}
        - Nutrition: {recipe['nutrition_per_100g']}
        - Instructions: {recipe['clean_instructions'][:100]}...
        
        Write a brief explanation (2-3 sentences) for why this recipe matches their needs:
        """.strip()
        
        return self.ask_llm(prompt)     
    
    def recommend(self, 
                 ingredients: List[str],
                 dietary_preference: str,
                 nutrition_goals: Optional[Dict[str, float]] = None,
                 max_missing: int = 1,
                 top_k: int = 3) -> Dict[str, Any]:
        """End-to-end recommendation pipeline"""
        # 1. Create query embedding
        query_vector = self.create_query_embedding(
            ingredients,dietary_preference,nutrition_goals ) #, dietary_preference, nutrition_goals
        
        
        # 2. Retrieve initial candidates
        candidates = self.retrieve_candidate_recipes(query_vector, top_k=100)
        
        # 3. Apply ingredient filter
        filtered = self.filter_by_ingredients_new(
            candidates, ingredients, max_missing=max_missing
        )
            
        filtered = self.filter_by_dietary_preferences(filtered, dietary_preference)
        
        # 5. Rerank with LLM
        user_query = f"I have {', '.join(ingredients)} and I want {dietary_preference} recipes"
        reranked = self.rerank_with_llm(filtered, user_query, top_k=top_k)
        
        # 6. Add explanations
        if not reranked.empty:
            reranked['explanation'] = reranked.apply(
                lambda row: self.generate_recipe_explanation(row, user_query), axis=1
            )
        
        # ───────── NEW ─────────
        reranked['image_url'] = reranked['title'].apply(generate_dish_image)
        # ───────────────────────
        
        # Return top recommendations with explanations
        result = {
            "query": user_query,
            "total_candidates": len(candidates),
            "filtered_count": len(filtered),
            "recommendations": reranked.to_dict('records') if not reranked.empty else [],
      
        }
        
        return result


# Example usage
if __name__ == "__main__":
    recommender = AdvancedRecipeRecommender(
        recipe_data_path="full_recipe_embeddings.pkl",
        faiss_index_path="recipe_faiss_index_ingredients_embeddings.idx"
    )
    
    user_request = {
    "dietary_preference": 
            "quick, savoury week-night dinner that is high protein, low carb, dairy-free and ready in under 30 minutes,fitness dinner",
            
    # ❶  — INGREDIENTS  ➜ 385-dim ingredient_embedding
    "ingredients": [
        "skinless chicken breast", 
        "salad"
        "broccoli florets", 
        "brown basmati rice", 
        "beef",
        "sphagetti",
        "carrots"
    ],
    
    "nutrition_goals": {
        "energy": 1500, 
        "fat": 30, 
    }
    }
    
    results = recommender.recommend(
        ingredients=user_request["ingredients"],
        dietary_preference=user_request["dietary_preference"],
        nutrition_goals=user_request["nutrition_goals"],
        max_missing=2,
        top_k=3
)
   
    
    print(f"Found {len(results['recommendations'])} recommendations")
    for i, recipe in enumerate(results['recommendations']):      # ← only 3 rows
        # 1) ask DeepSeek for a photo of this dish
        image_url = generate_dish_image(recipe['title'])          # ★ NEW ★

        # 2) (optional) attach it back onto the dict so the UI can use it later
        recipe['image_url'] = image_url

        # 3) logging / debug printout
        print(f"\n{i+1}. {recipe['title']}")
        print(f"Explanation: {recipe['explanation']}")
        print(f"Ingredients: {', '.join(recipe['ingredients_clean'][:5])}…")
        print(f"Instructions: {recipe['clean_instructions'][:100]}…")
        print(f"Nutrition (per 100g): "
            f"protein={recipe['nutrition_per_100g'].get('protein','N/A')} g, "
            f"carbs={recipe['nutrition_per_100g'].get('carbohydrate','N/A')} g, "
            f"image={image_url}")                               # ← view URL
