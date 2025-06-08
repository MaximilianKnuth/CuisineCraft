import React from 'react';
import ReactDOM from 'react-dom/client';
import RecipeRecommenderApp from './RecipeRecommenderApp';
import './index.css';            // keep Tailwind / global styles

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RecipeRecommenderApp />
  </React.StrictMode>,
);
