from flask import Flask, request, jsonify
from flask_cors import CORS
from Advanced_Recipe_Recommender import AdvancedRecipeRecommender
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the recommender
MODEL_PATH = "/Users/maxknuth/6.C511-Project/backend/full_recipe_embeddings.pkl"
FAISS_INDEX_PATH = "/Users/maxknuth/6.C511-Project/backend/recipe_faiss_index_ingredients_embeddings.idx"
recommender = None   

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(FAISS_INDEX_PATH):
    logger.error(f"Model files not found. Please ensure {MODEL_PATH} and {FAISS_INDEX_PATH} exist in the current directory.")
else:
    try:
        recommender = AdvancedRecipeRecommender(
            recipe_data_path=MODEL_PATH,
            faiss_index_path=FAISS_INDEX_PATH
        )
        logger.info("Recipe recommender initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing recommender: {str(e)}")
        recommender = None

def get_api_key():
    """Get API key from request headers"""
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return None
    return api_key

@app.route('/', methods=['GET'])
def index():
    """Root endpoint that provides API information"""
    api_info = {
        "name": "Recipe Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "/api/recommend": {
                "method": "POST",
                "description": "Get recipe recommendations based on ingredients",
                "parameters": {
                    "ingredients": "List of ingredients (required)",
                    "dietary_preference": "Optional dietary preference",
                    "nutrition_goals": "Optional nutrition goals",
                    "max_missing": "Maximum missing ingredients (default: 2)",
                    "top_k": "Number of recommendations to return (default: 3)"
                },
                "headers": {
                    "X-API-Key": "Deepseek API key (required)"
                }
            },
            "/api/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        },
        "status": "online"
    }
    return jsonify(api_info)

@app.route('/api/recommend', methods=['POST'])
def recommend_recipes():
    global recommender         
    if recommender is None:
        return jsonify({"error": "Recipe recommender not initialized"}), 500
    
    # Check for API key
    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "API key is required"}), 401
    
    data = request.json
    
    # Validate input
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    if 'ingredients' not in data or not data['ingredients']:
        return jsonify({"error": "Ingredients are required"}), 400
    
    if 'dietary_preference' not in data:
        data['dietary_preference'] = ""
    
    # Extract parameters
    ingredients = data['ingredients']
    dietary_preference = data['dietary_preference']
    nutrition_goals = data.get('nutrition_goals', {})
    max_missing = data.get('max_missing', 2)
    top_k = data.get('top_k', 3)
    
    logger.info(f"Received recommendation request with {len(ingredients)} ingredients")
    
    try:
        # Update the API key in the recommender
        recommender.update_api_key(api_key)
        
        # Get recommendations
        results = recommender.recommend(
            ingredients=ingredients,
            dietary_preference=dietary_preference,
            nutrition_goals=nutrition_goals,
            max_missing=max_missing,
            top_k=top_k
        )
        
        logger.info(f"Found {len(results['recommendations'])} recommendations")
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({"error": f"Error generating recommendations: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if recommender is None:
        return jsonify({"status": "error", "message": "Recommender not initialized"}), 500
    return jsonify({"status": "ok", "message": "API is running"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Using port 8000 instead of 5000
    app.run(host='0.0.0.0', port=port, debug=False)