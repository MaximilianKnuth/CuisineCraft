import { useState, useEffect } from 'react';
import { Search, X, ChevronRight, Loader2, ChefHat, Info, Star, Clock, Users } from 'lucide-react';

export default function RecipeRecommenderApp() {
  // State variables
  const [ingredients, setIngredients] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [dietaryPreference, setDietaryPreference] = useState('');
  const [nutritionGoals, setNutritionGoals] = useState({
    energy: '',
    protein: '',
    carbohydrate: '',
    fat: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  
  // API endpoint config
  const API_URL = 'http://localhost:8000'; // Update this if your backend is on a different port or host
  
  // Popular ingredients suggestions
  const popularIngredients = [
    'chicken breast', 'ground beef', 'rice', 'pasta', 'potatoes',
    'eggs', 'broccoli', 'spinach', 'onions', 'garlic',
    'tomatoes', 'bell peppers', 'carrots', 'olive oil', 'cheese'
  ];

  // Add ingredient to list
  const addIngredient = () => {
    if (inputValue.trim() && !ingredients.includes(inputValue.trim())) {
      setIngredients([...ingredients, inputValue.trim()]);
      setInputValue('');
    }
  };

  // Remove an ingredient
  const removeIngredient = (index) => {
    setIngredients(ingredients.filter((_, i) => i !== index));
  };

  // Add suggested ingredient
  const addSuggestedIngredient = (ingredient) => {
    if (!ingredients.includes(ingredient)) {
      setIngredients([...ingredients, ingredient]);
    }
  };

  // Handle input change
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  // Handle key press (Enter adds ingredient)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      addIngredient();
    }
  };

  // Handle nutrition goal input changes
  const handleNutritionChange = (e) => {
    const { name, value } = e.target;
    setNutritionGoals({
      ...nutritionGoals,
      [name]: value
    });
  };

  // Find recipes - real API call
  const findRecipes = async () => {
    if (ingredients.length === 0) {
      alert('Please add at least one ingredient');
      return;
    }

    setIsLoading(true);
    setErrorMessage('');
    
    // Prepare request payload
    const payload = {
      ingredients: ingredients,
      dietary_preference: dietaryPreference,
      nutrition_goals: Object.fromEntries(
        Object.entries(nutritionGoals).filter(([_, value]) => value !== '')
      ),
      max_missing: 2, // Default value, could be made configurable
      top_k: 3 // Default value, could be made configurable
    };

    try {
      // Make API call to backend
      const response = await fetch(`${API_URL}/api/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.recommendations && data.recommendations.length > 0) {
        setRecommendations(data.recommendations);
      } else {
        setRecommendations([]);
        setErrorMessage('No recipes found matching your criteria');
      }
    } catch (error) {
      console.error('Error fetching recipes:', error);
      setErrorMessage(`Failed to get recipes: ${error.message}`);
      setRecommendations([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Open recipe details modal
  const openRecipeDetails = (recipe) => {
    setSelectedRecipe(recipe);
    setShowModal(true);
  };

  // Close modal
  const closeModal = () => {
    setShowModal(false);
  };

  // Pre-load some ingredients for demo
  useEffect(() => {
    setIngredients(['chicken breast', 'broccoli', 'brown rice', 'carrots']);
    setDietaryPreference('high protein, low carb, quick dinner');
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-400 to-blue-500 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-center">
            <ChefHat size={36} className="text-white mr-3" />
            <h1 className="text-3xl font-bold text-white">Smart Recipe Recommender</h1>
          </div>
          <p className="text-center text-white mt-2">
            Find delicious recipes based on your ingredients and preferences
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto p-4 md:p-6">
        {/* Input Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Ingredients Section */}
            <div>
              <h2 className="text-2xl font-semibold mb-4">What ingredients do you have?</h2>
              
              {/* Ingredient Input */}
              <div className="flex items-center mb-3">
                <input
                  type="text"
                  value={inputValue}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter an ingredient"
                  className="w-full px-4 py-2 border rounded-l focus:outline-none focus:ring-2 focus:ring-green-500"
                />
                <button 
                  onClick={addIngredient}
                  className="bg-green-500 text-white px-4 py-2 rounded-r hover:bg-green-600 transition duration-200"
                >
                  <Search size={20} />
                </button>
              </div>
              
              {/* Ingredients Tags */}
              <div className="flex flex-wrap gap-2 mb-4">
                {ingredients.map((ingredient, index) => (
                  <div key={index} className="bg-green-100 text-green-800 rounded-full px-3 py-1 flex items-center">
                    <span>{ingredient}</span>
                    <button 
                      onClick={() => removeIngredient(index)}
                      className="ml-2 text-green-600 hover:text-green-800"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
              
              {/* Popular Ingredients */}
              <div>
                <div className="text-sm text-gray-500 mb-2">Popular ingredients:</div>
                <div className="flex flex-wrap gap-2">
                  {popularIngredients.slice(0, 8).map((ingredient) => (
                    <button
                      key={ingredient}
                      onClick={() => addSuggestedIngredient(ingredient)}
                      className={`text-xs px-2 py-1 rounded-full border ${
                        ingredients.includes(ingredient) 
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                        : 'border-blue-300 text-blue-500 hover:bg-blue-50'
                      }`}
                      disabled={ingredients.includes(ingredient)}
                    >
                      {ingredient}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Preferences Section */}
            <div>
              <h2 className="text-2xl font-semibold mb-4">Dietary Preferences</h2>
              
              <textarea
                value={dietaryPreference}
                onChange={(e) => setDietaryPreference(e.target.value)}
                placeholder="Describe your dietary needs (e.g., low carb, high protein, vegetarian, etc.)"
                className="w-full px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-green-500 mb-4"
                rows={3}
              />
              
              <h3 className="text-xl font-semibold mb-3">Nutrition Goals (per 100g)</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-700 mb-1 text-sm">Energy (kcal)</label>
                  <input
                    type="number"
                    name="energy"
                    value={nutritionGoals.energy}
                    onChange={handleNutritionChange}
                    placeholder="e.g., 250"
                    className="w-full px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 mb-1 text-sm">Protein (g)</label>
                  <input
                    type="number"
                    name="protein"
                    value={nutritionGoals.protein}
                    onChange={handleNutritionChange}
                    placeholder="e.g., 20"
                    className="w-full px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 mb-1 text-sm">Carbs (g)</label>
                  <input
                    type="number"
                    name="carbohydrate"
                    value={nutritionGoals.carbohydrate}
                    onChange={handleNutritionChange}
                    placeholder="e.g., 30"
                    className="w-full px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 mb-1 text-sm">Fat (g)</label>
                  <input
                    type="number"
                    name="fat"
                    value={nutritionGoals.fat}
                    onChange={handleNutritionChange}
                    placeholder="e.g., 10"
                    className="w-full px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>
              </div>
            </div>
          </div>
          
          {/* Search Button */}
          <div className="flex justify-center mt-8">
            <button
              onClick={findRecipes}
              disabled={isLoading}
              className="bg-blue-600 text-white px-8 py-3 rounded-full text-lg font-semibold hover:bg-blue-700 transition duration-200 flex items-center"
            >
              {isLoading ? (
                <>
                  <Loader2 size={24} className="mr-2 animate-spin" />
                  Finding Recipes...
                </>
              ) : (
                <>
                  <Search size={24} className="mr-2" />
                  Find Recipes
                </>
              )}
            </button>
          </div>
        </div>
        
        {/* Error Message */}
        {errorMessage && !isLoading && (
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg mb-8 text-center">
            <p>{errorMessage}</p>
          </div>
        )}
        
        {/* Results Section */}
        {recommendations.length > 0 && !isLoading && (
          <div className="mb-8">
            <h2 className="text-2xl font-semibold mb-6">Recommended Recipes</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendations.map((recipe) => (
                <div 
                  key={recipe.id || recipe.title}
                  className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition duration-300"
                >
                  {/* Recipe Image (placeholder) */}
                  <div className="h-48 bg-gradient-to-r from-blue-100 to-green-100 flex items-center justify-center">
                    <ChefHat size={64} className="text-gray-400" />
                  </div>
                  
                  {/* Recipe Content */}
                  <div className="p-4">
                    <h3 className="text-xl font-semibold mb-2">{recipe.title}</h3>
                    <p className="text-gray-600 mb-4">{recipe.explanation || recipe.description || 'A delicious recipe based on your ingredients.'}</p>
                    
                    {/* Nutrition Tags - Handle different API response formats */}
                    {recipe.nutrition_per_100g && (
                      <div className="flex flex-wrap gap-2 mb-4">
                        {recipe.nutrition_per_100g.protein && (
                          <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                            {recipe.nutrition_per_100g.protein}g protein
                          </span>
                        )}
                        {recipe.nutrition_per_100g.carbohydrate && (
                          <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-sm">
                            {recipe.nutrition_per_100g.carbohydrate}g carbs
                          </span>
                        )}
                        {recipe.nutrition_per_100g.fat && (
                          <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm">
                            {recipe.nutrition_per_100g.fat}g fat
                          </span>
                        )}
                        {recipe.nutrition_per_100g.energy && (
                          <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-sm">
                            {recipe.nutrition_per_100g.energy} kcal
                          </span>
                        )}
                      </div>
                    )}
                    
                    {/* Ingredients Preview - Handle different API response formats */}
                    <div className="mb-4">
                      <div className="text-sm text-gray-500">Ingredients:</div>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {(recipe.ingredients_clean || recipe.ingredients || []).slice(0, 3).map((ing, idx) => (
                          <span key={idx} className="text-sm bg-gray-100 px-2 py-1 rounded">
                            {ing}
                          </span>
                        ))}
                        {(recipe.ingredients_clean || recipe.ingredients || []).length > 3 && (
                          <span className="text-sm text-gray-500">
                            +{(recipe.ingredients_clean || recipe.ingredients || []).length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                    
                    {/* View Recipe Button */}
                    <button
                      onClick={() => openRecipeDetails(recipe)}
                      className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 transition duration-200 flex items-center justify-center"
                    >
                      View Recipe
                      <ChevronRight size={16} className="ml-1" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* No Results Message */}
        {recommendations.length === 0 && !isLoading && ingredients.length > 0 && !errorMessage && (
          <div className="text-center py-12 bg-white rounded-lg shadow">
            <Info size={48} className="mx-auto text-gray-400 mb-4" />
            <h3 className="text-2xl font-semibold text-gray-600">No recipes found</h3>
            <p className="text-gray-500 mt-2">Try adding more ingredients or adjusting your preferences</p>
          </div>
        )}
      </div>
      
      {/* Recipe Modal */}
      {showModal && selectedRecipe && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-screen overflow-y-auto">
            <div className="sticky top-0 bg-white p-4 border-b flex justify-between items-center z-10">
              <h3 className="text-2xl font-bold">{selectedRecipe.title}</h3>
              <button onClick={closeModal} className="text-gray-500 hover:text-gray-700">
                <X size={24} />
              </button>
            </div>
            
            <div className="p-6">
              {/* Recipe Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-xl font-semibold mb-3 flex items-center">
                    <Info size={20} className="mr-2 text-blue-500" />
                    Ingredients
                  </h4>
                  <ul className="space-y-2 mb-6">
                    {(selectedRecipe.ingredients_clean || selectedRecipe.ingredients || []).map((ingredient, idx) => (
                      <li key={idx} className="flex items-start">
                        <div className="bg-green-100 text-green-800 rounded-full w-6 h-6 flex items-center justify-center mr-2 mt-0.5">
                          <Check size={14} />
                        </div>
                        <span>{ingredient}</span>
                      </li>
                    ))}
                  </ul>
                  
                  {selectedRecipe.nutrition_per_100g && (
                    <>
                      <h4 className="text-xl font-semibold mb-3 flex items-center">
                        <Star size={20} className="mr-2 text-yellow-500" />
                        Nutrition (per 100g)
                      </h4>
                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(selectedRecipe.nutrition_per_100g).map(([key, value]) => (
                          <div key={key} className="bg-gray-50 p-3 rounded-lg">
                            <div className="text-sm text-gray-500">{key}</div>
                            <div className="text-xl font-bold">{value}{key === 'energy' ? ' kcal' : 'g'}</div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
                
                <div>
                  <h4 className="text-xl font-semibold mb-3 flex items-center">
                    <ClipboardList size={20} className="mr-2 text-blue-500" />
                    Instructions
                  </h4>
                  {(selectedRecipe.clean_instructions || selectedRecipe.instructions || "").split('\n').filter(line => line.trim()).map((step, idx) => (
                    <div key={idx} className="mb-4">
                      <div className="flex items-start">
                        <div className="bg-blue-100 text-blue-800 rounded-full w-6 h-6 flex items-center justify-center mr-3 mt-0.5">
                          {idx + 1}
                        </div>
                        <div>{step.replace(/^\d+\.\s*/, '')}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper components
function Check(props) {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      width={props.size || 24} 
      height={props.size || 24} 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="2" 
      strokeLinecap="round" 
      strokeLinejoin="round"
      className={props.className}
    >
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
  );
}

function ClipboardList(props) {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      width={props.size || 24} 
      height={props.size || 24} 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="2" 
      strokeLinecap="round" 
      strokeLinejoin="round"
      className={props.className}
    >
      <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
      <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
      <path d="M9 12h6"></path>
      <path d="M9 16h6"></path>
      <path d="M9 8h1"></path>
    </svg>
  );
}