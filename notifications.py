import pandas as pd
from sentence_transformers import SentenceTransformer, util, losses, InputExample
from torch.utils.data import DataLoader
import random
import time
import math
import requests
from requests.exceptions import ConnectionError
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# USER NOTIFIC. FUNC
def load_user_data(file_path):
    try:
        user_data_df = pd.read_excel(file_path, header=None).T
        user_data_df.columns = user_data_df.iloc[0]
        user_data_df = user_data_df[1:]
        user_data_df.rename(columns={'UserName:UserID': 'UserID'}, inplace=True)
        user_data_df[['UserName', 'UserID']] = user_data_df['UserID'].str.split(':', expand=True)
        user_data_df.drop('UserID', axis=1, inplace=True)

        geolocator = Nominatim(user_agent="clothing_notification_app", timeout=10)

        def geocode_address(address, max_retries=2, retry_delay=1):
            retries = 0
            while retries < max_retries:
                try:
                    location = geolocator.geocode(address, timeout=10)
                    if location:
                        return location.latitude, location.longitude
                    else:
                        return None, None
                except (GeocoderTimedOut, GeocoderUnavailable, ConnectionError) as e:
                    print(f"Geocoding error: {e}. Retrying in {retry_delay} seconds...")
                    retries += 1
                    time.sleep(retry_delay)
                except requests.exceptions.SSLError as e:
                    print(f"SSL error: {e}. Retrying in {retry_delay} seconds...")
                    retries += 1
                    time.sleep(retry_delay)
                except requests.exceptions.ReadTimeout as e:
                    print(f"Read Timeout error: {e}. Retrying in {retry_delay} seconds...")
                    retries += 1
                    time.sleep(retry_delay)
                print(f"Failed to geocode address after {max_retries} retries.")
                time.sleep(1)
                return None, None

        user_data_df[['Latitude', 'Longitude']] = user_data_df['Address'].apply(geocode_address).apply(pd.Series)
        return user_data_df

    except Exception as e:
        print(f"An error occurred loading user data: {e}")
        return None

def send_category_notifications(category, user_df):
    if user_df is None:
        return
    interested_users = user_df[user_df['Most Visited Category'] == category]
    if not interested_users.empty:
        print(f"\nNotifications for {category} returns:")
        for index, row in interested_users.iterrows():
            print(f"To: {row['UserName']} at {row['Address']}")
            print(f"Subject: New {category} items available!")
            print(f"Message: Exciting news! We have new {category} items back in stock. Check them out!\n")
    else:
        print(f"No users found who frequently visit {category}.\n")

# --- Fine-Tuning Setup ---
clothing_categories = [
    'Outdoor Wear', 'Sportswear', 'Casual Wear', 'Formal Wear', 'Footwear',
    'Activewear', 'Loungewear', 'Swimwear', 'Sleepwear', 'Jeans',
    'T-Shirts', 'Hoodies & Sweatshirts', 'Jackets & Coats', 'Suits & Blazers',
    'Dresses', 'Skirts & Pants', 'Shorts', 'Tops', 'Bags & Backpacks',
    'Hats & Caps', 'Socks & Hosiery', 'Maternity Wear', 'Ethnic Wear',
    'Luxury/Designer Wear', 'Vintage Clothing', 'Rainwear', 'Thermal Wear',
    'Performance Gear', 'Bathrobes', 'Pajamas & Nightwear',
    'Shirts', 'Tunics', 'Blouses', 'Ties & Scarves', 'Sweaters & Cardigans',
    'Vests', 'Chinos & Trousers', 'Overalls & Jumpsuits', 'Suits', 'Blazers & Jackets',
    'Camisoles & Tops', 'Kimonos', 'Sweatpants & Joggers', 'Leggings & Tights',
    'Fleece Jackets', 'Fleece Pants', 'Hiking Gear', 'Raincoats & Ponchos', 'Vest Tops',
    'Underwear', 'Bras & Lingerie', 'Tights & Pantyhose', 'Sleep Pants & Shorts',
    'Teddy Coats', 'Skorts', 'Bermuda Shorts', 'Coveralls', 'Workwear', 'Uniforms'
]

# TAXONOMY PART
def build_category_taxonomy():
    taxonomy = {
        "Tops": ["T-Shirts", "Shirts", "Blouses", "Tunics", "Tops", "Vests", "Vest Tops", "Camisoles & Tops", "Sweaters & Cardigans", "Hoodies & Sweatshirts", "Kimonos"],
        "Bottoms": ["Jeans", "Shorts", "Skirts & Pants", "Chinos & Trousers", "Leggings & Tights", "Sweatpants & Joggers", "Fleece Pants", "Skorts", "Bermuda Shorts", "Tights & Pantyhose"],
        "Full Body": ["Dresses", "Overalls & Jumpsuits", "Suits", "Coveralls"],
        "Outerwear": ["Jackets & Coats", "Blazers & Jackets", "Suits & Blazers", "Fleece Jackets", "Raincoats & Ponchos", "Teddy Coats", "Rainwear", "Vests"],
        "Active & Performance": ["Activewear", "Sportswear", "Performance Gear", "Hiking Gear", "Outdoor Wear", "Vests"],
        "Sleepwear & Lounge": ["Loungewear", "Sleepwear", "Pajamas & Nightwear", "Sleep Pants & Shorts", "Bathrobes"],
        "Swimwear": ["Swimwear"],
        "Underwear & Intimates": ["Underwear", "Bras & Lingerie"],
        "Footwear": ["Footwear"],
        "Accessories": ["Bags & Backpacks", "Hats & Caps", "Ties & Scarves", "Socks & Hosiery"],
        "Style Categories": ["Casual Wear", "Formal Wear", "Luxury/Designer Wear", "Vintage Clothing", "Ethnic Wear", "Maternity Wear", "Thermal Wear"],
        "Professional Wear": ["Workwear", "Uniforms", "Suits", "Suits & Blazers", "Vests"]
    }
    item_to_categories = {}
    for category, items in taxonomy.items():
        for item in items:
            if item not in item_to_categories:
                item_to_categories[item] = []
            item_to_categories[item].append(category)

    related_categories = { 
        "Vests": ["Vest Tops", "Suits", "Blazers & Jackets", "Formal Wear", "Casual Wear", "Outdoor Wear", "Suits & Blazers"],
        "Vest Tops": ["Vests", "Tops", "Camisoles & Tops"],
        "Suits": ["Vests", "Suits & Blazers", "Blazers & Jackets", "Formal Wear"],
        "Suits & Blazers": ["Vests", "Blazers & Jackets", "Formal Wear"],
        "Blazers & Jackets": ["Vests", "Suits", "Suits & Blazers", "Jackets & Coats"],
        "Activewear": ["Sportswear", "Performance Gear", "Hiking Gear", "Leggings & Tights"],
        "Sportswear": ["Activewear", "Performance Gear", "Sweatpants & Joggers"],
        "Outdoor Wear": ["Hiking Gear", "Rainwear", "Raincoats & Ponchos", "Fleece Jackets", "Vests"],
        "Casual Wear": ["T-Shirts", "Jeans", "Shorts", "Hoodies & Sweatshirts", "Vests"],
        "Formal Wear": ["Suits", "Suits & Blazers", "Blazers & Jackets", "Ties & Scarves", "Vests"],
        "Sleepwear": ["Pajamas & Nightwear", "Sleep Pants & Shorts", "Bathrobes"],
        "Workwear": ["Uniforms", "Coveralls"],
        "Loungewear": ["Sweatpants & Joggers", "Pajamas & Nightwear"],
        "Thermal Wear": ["Fleece Jackets", "Fleece Pants"],
        "Fleece Jackets": ["Fleece Pants", "Outdoor Wear"],
        "Rainwear": ["Raincoats & Ponchos"],
        "Uniforms": ["Workwear", "Vests"],
        "T-Shirts": ["Tops", "Casual Wear"],
        "Shirts": ["Tops", "Formal Wear", "Casual Wear"],
        "Tunics": ["Tops", "Ethnic Wear"],
        "Hoodies & Sweatshirts": ["Tops", "Casual Wear", "Loungewear"],
    }

    return taxonomy, item_to_categories, related_categories

def calculate_taxonomy_similarity(cat1, cat2, cache={}):
    if (cat1, cat2) in cache:
        return cache[(cat1, cat2)]

    taxonomy, item_to_categories, related_categories = build_category_taxonomy()
    if cat1 == cat2:
        cache[(cat1, cat2)] = 1.0
        return 1.0
    words1 = set(cat1.lower().split() + cat1.lower().replace('&', '').split())
    words2 = set(cat2.lower().split() + cat2.lower().replace('&', '').split())
    word_overlap = len(words1.intersection(words2))
    if word_overlap > 0:
        cache[(cat1, cat2)] = 0.8
        return 0.8
    parents_cat1 = item_to_categories.get(cat1, [])
    parents_cat2 = item_to_categories.get(cat2, [])
    shared_parents = set(parents_cat1).intersection(set(parents_cat2))
    if shared_parents:
        cache[(cat1, cat2)] = 0.7
        return 0.7
    if cat1 in related_categories and cat2 in related_categories[cat1]:
        cache[(cat1, cat2)] = 0.6
        return 0.6
    if cat2 in related_categories and cat1 in related_categories[cat2]:
        cache[(cat1, cat2)] = 0.6
        return 0.6
    for parent in parents_cat1:
        for related_parent in parents_cat2:
            if parent in related_categories and related_parent in related_categories[parent]:
                cache[(cat1, cat2)] = 0.4
                return 0.4
            if related_parent in related_categories and parent in related_categories[related_parent]:
                cache[(cat1, cat2)] = 0.4
                return 0.4

    unrelated_categories = {
        "Bags & Backpacks": ["Raincoats"],
        "Bras & Lingerie": ["Hiking Gear", "Workwear", "Raincoats", "Bathrobes"],
        "Footwear": ["Blazers", "Kimonos", "Thermal Wear", "Bathrobes"],
        "Suits & Blazers": ["Swimwear", "Socks & Hosiery", "Coveralls"],
        "Sleepwear": ["Hiking Gear", "Performance Gear", "Uniforms", "Bathrobes"],
        "T-Shirts": ["Bathrobes", "Floor Mats", "Bags & Backpacks"],
        "Shorts": ["Luxury/Designer Wear", "Rainwear", "Vintage Clothing", "Bathrobes"],
        "Hats & Caps": ["Jeans", "Leggings & Tights", "Skirts & Pants"],
    }
    if cat1 in unrelated_categories and cat2 in unrelated_categories[cat1]:
        cache[(cat1, cat2)] = 0.1
        return 0.1
    if cat2 in unrelated_categories and cat1 in unrelated_categories[cat2]:
        cache[(cat1, cat2)] = 0.1
        return 0.1
    cache[(cat1, cat2)] = 0.2
    return 0.2

def build_training_examples_with_taxonomy():
    train_examples = []
    similarity_cache = {}
    for i, cat1 in enumerate(clothing_categories):
        for j, cat2 in enumerate(clothing_categories):
            if i != j:
                similarity = calculate_taxonomy_similarity(cat1, cat2, similarity_cache)
                train_examples.append(InputExample(texts=[cat1, cat2], label=similarity))
    return train_examples

# TRAIN THE MACXHINEE
def train_model_with_taxonomy(model_save_path="fine_tuned_clothing_taxonomy_model"):
    print("Training new model with taxonomy-based similarities...")
    train_examples = build_training_examples_with_taxonomy()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100)
    model.save(model_save_path)
    return model

def find_similar_users(model, category, user_df, similarity_cache={}):
    if user_df is None:
        return pd.DataFrame()
    try:
        category_embedding = model.encode(category, convert_to_tensor=True)
        user_categories = user_df['Most Visited Category'].tolist()
        user_embeddings = model.encode(user_categories, convert_to_tensor=True)
        cosine_scores = util.cos_sim(category_embedding, user_embeddings)[0]
        similarity_df = pd.DataFrame({
            'UserName': user_df['UserName'],
            'Address': user_df['Address'],
            'Similarity': cosine_scores.cpu().numpy()
        })
        similar_users = similarity_df.sort_values(by='Similarity', ascending=False)
        return similar_users
    except Exception as e:
        print(f"An error occurred in find_similar_users: {e}")
        return pd.DataFrame()

# --- Warehouse Locations ---
warehouses = {
    "Warehouse A": "St Lawrence-East Bayfront-The Islands, Toronto, ON",
    "Warehouse B": "Harbourfront-CityPlace, Toronto, ON",
    "Warehouse C": "Church-Wellesley, Toronto, ON",
}

warehouse_coords = {}
geolocator = Nominatim(user_agent = "clothing_notification_app")

for name, address in warehouses.items():
    try:
        location = geolocator.geocode(address)
        if location:
            warehouse_coords[name] = (location.latitude, location.longitude)
        else:
            print(f"Could not geocode warehouse {name}")
    except (GeocoderTimedOut, GeocoderUnavailable):
        print(f"Geocoding failed for warehouse {name}")

# --- Distance Calculation ---
def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c
    return distance

def send_ai_notifications(model, category, user_df, max_notifications=5):
    if user_df is None:
        return
    similar_users = find_similar_users(model, category, user_df)
    if not similar_users.empty:
        print(f"\nAI-Enhanced Notifications for {category} returns:")
        direct_users = user_df[user_df['Most Visited Category'] == category]['UserName'].tolist()
        filtered_users = similar_users[~similar_users['UserName'].isin(direct_users)]
        top_users = filtered_users.head(max_notifications)
        if not top_users.empty:
            random_warehouse = random.choice(list(warehouse_coords.items()))
            warehouse_name, warehouse_coord = random_warehouse

            print(f"Randomly selected warehouse: {warehouse_name}")
            for index, row in top_users.iterrows():
                original_category = user_df[user_df['UserName'] == row['UserName']]['Most Visited Category'].values[0]
                user_lat = user_df[user_df['UserName'] == row['UserName']]['Latitude'].values[0]
                user_lon = user_df[user_df['UserName'] == row['UserName']]['Longitude'].values[0]
                user_coord = (user_lat, user_lon)

                if user_lat is None or user_lon is None:
                    print(f"Skipping {row['UserName']}: Could not geocode user address.")
                    continue

                distance = calculate_distance(user_coord, warehouse_coord)

                if distance <= 20:
                    print(f"To: {row['UserName']} ({original_category}) at {row['Address']}")
                    print(f"Subject: Items similar to {category} are back in stock at {warehouse_name}!")
                    print(f"Message: We noticed your interest in {original_category}. You might like our new {category} items available at our {warehouse_name} location, which is only {distance:.2f}km away!\n")
                else:
                    print(f"Skipping {row['UserName']}: Too far from {warehouse_name}.")
        else:
            print("No similar users found.\n") #this is unlikely to happen with the current change.
    else:
        print(f"No users found.\n") #this is unlikely to happen with the current change.

# --- Main Execution ---
if __name__ == "__main__":
    random.seed(int(time.time()))
    user_data_file_path = "user_clothing_data.xlsx"
    user_data_df = load_user_data(user_data_file_path)
    if user_data_df is not None:
        train_ai_model = False
        if train_ai_model:
            model = train_model_with_taxonomy()
        else:
            model = SentenceTransformer("fine_tuned_clothing_taxonomy_model")
        returned_category = random.choice(clothing_categories)
        print(f"\nReturned Category: {returned_category}")
        send_category_notifications(returned_category, user_data_df)
        send_ai_notifications(model, returned_category, user_df=user_data_df, max_notifications=5)