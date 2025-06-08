import { useState, useEffect } from 'react';
import {
  Search, X, ChefHat, Loader2, ChevronRight,
  Info, Star, Clock, Users, Check, ClipboardList, Key
} from 'lucide-react';

/* ------------------------------------------------------------------ */
/*  CONFIG  – backend base URL                                         */
/*  Put VITE_API_URL=http://localhost:8000 in frontend/.env.local      */
/*  if you run the API on a different port.                            */
const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
/* ------------------------------------------------------------------ */

interface Nutrition { protein: number; carbohydrate: number; fat: number; energy: number }
interface Recipe {
  id: string|number; title: string; ingredients_clean: string[];
  clean_instructions: string; nutrition_per_100g: Nutrition; explanation: string
}

export default function RecipeRecommenderApp() {
  /* ---------- state ---------- */
  const [ingredients, setIngredients] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [dietaryPreference, setDietaryPreference] = useState('');
  const [nutritionGoals, setNutritionGoals] = useState({ energy:'', protein:'', carbohydrate:'', fat:'' });
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<Recipe[]>([]);
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe|null>(null);
  const [showModal, setShowModal] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [showApiKeyError, setShowApiKeyError] = useState(false);

  /* ---------- helpers ---------- */
  const addIngredient = () => {
    if (inputValue.trim() && !ingredients.includes(inputValue.trim())) {
      setIngredients([...ingredients, inputValue.trim()]);
      setInputValue('');
    }
  };
  const removeIngredient = (i:number) => setIngredients(ingredients.filter((_,idx)=>idx!==i));
  const addSuggested = (ing:string) => !ingredients.includes(ing) && setIngredients([...ingredients, ing]);

  /* ---------- real API call ---------- */
  const findRecipes = async () => {
    if (!ingredients.length) { alert('Please add at least one ingredient'); return; }
    if (!apiKey.trim()) { setShowApiKeyError(true); return; }

    setIsLoading(true); setRecommendations([]);
    try {
      const res = await fetch(`${API_BASE}/api/recommend`, {
        method:'POST',
        headers:{
          'Content-Type':'application/json',
          'X-API-Key': apiKey.trim()
        },
        body:JSON.stringify({
          ingredients,
          dietary_preference: dietaryPreference,
          nutrition_goals: nutritionGoals,
          max_missing:2,
          top_k:3
        })
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      setRecommendations(data.recommendations ?? []);
    } catch (err) {
      console.error(err);
      alert('Backend error – check browser console and Flask log');
    } finally { setIsLoading(false); }
  };

  /* ---------- seed demo ---------- */
  useEffect(() => {
    setIngredients(['chicken breast','broccoli','brown rice','carrots']);
    setDietaryPreference('high protein, low carb, quick dinner');
  }, []);

  /* ---------- UI ---------- */
  const popular = ['chicken breast','ground beef','rice','pasta','eggs','broccoli','spinach','potatoes'];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* header */}
      <header className="bg-gradient-to-r from-green-400 to-blue-500 p-6 text-center text-white">
        <h1 className="text-3xl font-bold inline-flex items-center justify-center">
          <ChefHat className="mr-2"/> Smart Recipe Recommender
        </h1>
        <p className="mt-1">Find delicious recipes based on your ingredients and preferences</p>
      </header>

      <main className="max-w-6xl mx-auto p-4 md:p-6">
        {/* API Key Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Deepseek API Key
          </label>
          <div className="flex items-center">
            <div className="relative flex-grow">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Key className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                  setShowApiKeyError(false);
                }}
                className={`block w-full pl-10 pr-3 py-2 border rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm ${
                  showApiKeyError ? 'border-red-500' : 'border-gray-300'
                }`}
                placeholder="Enter your Deepseek API key"
              />
            </div>
          </div>
          {showApiKeyError && (
            <p className="mt-1 text-sm text-red-600">Please enter your Deepseek API key</p>
          )}
        </div>

        {/* ingredient + preference form (unchanged UI except for handlers) */}
        {/* ... paste your existing JSX for the form here (unchanged) ... */}

        {/* search button */}
        <div className="flex justify-center mt-8">
          <button onClick={findRecipes} disabled={isLoading}
            className="bg-blue-600 text-white px-8 py-3 rounded-full flex items-center hover:bg-blue-700">
            {isLoading ? <Loader2 className="mr-2 animate-spin"/> : <Search className="mr-2"/>}
            {isLoading ? 'Finding Recipes…' : 'Find Recipes'}
          </button>
        </div>

        {/* results */}
        {recommendations.length>0 && !isLoading && (
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-6">Recommended Recipes</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendations.map(r=>(
                <article key={r.id} className="bg-white rounded shadow hover:shadow-lg transition">
                  <div className="h-48 bg-gradient-to-r from-blue-100 to-green-100 flex items-center justify-center">
                    <ChefHat size={56} className="text-gray-400"/>
                  </div>
                  <div className="p-4">
                    <h3 className="text-xl font-semibold mb-1">{r.title}</h3>
                    <p className="text-gray-600 mb-3">{r.explanation}</p>
                    <div className="flex flex-wrap gap-2 text-sm mb-4">
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">{r.nutrition_per_100g.protein} g protein</span>
                      <span className="bg-green-100 text-green-800 px-2 py-1 rounded">{r.nutrition_per_100g.carbohydrate} g carbs</span>
                      <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded">{r.nutrition_per_100g.fat} g fat</span>
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded">{r.nutrition_per_100g.energy} kcal</span>
                    </div>
                    <button onClick={()=>{setSelectedRecipe(r);setShowModal(true);}}
                      className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 flex items-center justify-center">
                      View Recipe <ChevronRight size={16} className="ml-1"/>
                    </button>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        {/* empty state */}
        {recommendations.length===0 && !isLoading && ingredients.length>0 && (
          <p className="text-center text-gray-500 mt-10">No recipes found – try adding more ingredients or loosening constraints.</p>
        )}
      </main>

      {/* modal */}
      {showModal && selectedRecipe && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-3xl w-full overflow-y-auto max-h-[90vh]">
            <header className="p-4 border-b flex justify-between items-center">
              <h3 className="text-xl font-bold">{selectedRecipe.title}</h3>
              <button onClick={()=>setShowModal(false)}><X/></button>
            </header>
            <div className="p-6 space-y-6">
              {/* ... render ingredients / nutrition / instructions … */}
              {/* keep your existing modal details JSX here */}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
