import pandas as pd
import json

def return_df_final():

    # -- df_layer1: from layer1.json --
    with open("recipe1m_downloads/extracted/layer1.json", "r", encoding="utf-8") as f:
        layer1_data = json.load(f)

    df_layer1 = pd.DataFrame([
        {
            "id": r["id"],
            "title": r["title"],
            "partition": r["partition"],
            "url": r.get("url"),
            "ingredients_raw": [i["text"] for i in r.get("ingredients", [])],
            "instructions_full": " ".join([s["text"] for s in r.get("instructions", [])])
        }
        for r in layer1_data
    ])
    #df_layer1.head(1)

    with open("recipe1m_downloads/det_ingrs.json", "r", encoding="utf-8") as f:
        det_ingrs_data = json.load(f)

    df_det_ingrs = pd.DataFrame([
        {
            "id": r["id"],
            "ingredients_clean": [ing["text"] for ing, valid in zip(r["ingredients"], r["valid"]) if valid]
        }
        for r in det_ingrs_data
    ])

    def parse_full_nutrition_entry(entry):
        n = len(entry["ingredients"])

        ingredients_structured = []
        for i in range(n):
            ingredients_structured.append({
                "ingredient": entry["ingredients"][i]["text"] if i < len(entry["ingredients"]) else None,
                "quantity": entry["quantity"][i]["text"] if i < len(entry["quantity"]) else None,
                "unit": entry["unit"][i]["text"] if i < len(entry["unit"]) else None,
                "weight_g": entry["weight_per_ingr"][i] if i < len(entry["weight_per_ingr"]) else None,
                "nutrition": entry["nutr_per_ingredient"][i] if i < len(entry["nutr_per_ingredient"]) else None
            })

        return {
            "id": entry.get("id"),
            "title": entry.get("title"),
            "partition": entry.get("partition"),
            "url": entry.get("url"),
            "instructions": " ".join([step["text"] for step in entry.get("instructions", [])]),
            "ingredients_structured": ingredients_structured,
            "nutrition_per_100g": entry.get("nutr_values_per100g", {}),
            "fsa_lights_per_100g": entry.get("fsa_lights_per100g", {})
        }

    # Load your JSON file or individual object here
    with open("recipe1m_downloads/recipes_with_nutritional_info.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Example: parse just the first entry (you can loop later)
    parsed = [parse_full_nutrition_entry(recipe) for recipe in data]

    # Convert to DataFrame
    df_nutrition_full = pd.DataFrame(parsed)

    # Step 1: merge layer1 and det_ingrs
    df_combined = pd.merge(df_layer1, df_det_ingrs, on="id", how="inner")

    # Step 2: merge with nutrition
    df_final = pd.merge(df_combined, df_nutrition_full, on="id", how="inner", suffixes=("", "_nutrition"))
    drop_columns = ['partition','url', 'partition_nutrition']
    df_final_filtered = df_final.drop(drop_columns, axis = 1)

    return df_final_filtered