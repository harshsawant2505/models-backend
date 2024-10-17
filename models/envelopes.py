import pandas as pd
df = pd.read_csv('Untitled spreadsheet - Sheet1.csv')
from config import db
import math

# Function to calculate distance using the Haversine formula
def calculate_distance(coord1, coord2):
    R = 6371  # Radius of the Earth in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

# Function to get nearby available quests
def get_nearby_quests(user_id, user_coordinates, radius=10):
    df['distance'] = df['coordinates'].apply(lambda x: calculate_distance(x, user_coordinates))
    nearby_quests = df[df['distance'] <= radius]
    nearby_quests = nearby_quests[~nearby_quests['user_id_completed'].apply(lambda x: user_id in x)]
    nearby_quests.sort_values(by='distance')
    if not nearby_quests.empty:
        first_quest_index = nearby_quests.index[0]
        df.at[first_quest_index, 'user_id_completed'].append(user_id)
        return nearby_quests.iloc[0].to_dict()
    else:
        return None

def get_envelope_state(user_id):
    user_ref = db.collection('users').document(user_id)
    user = user_ref.get()
    if user.exists:
        user_data = user.to_dict()
        return user_data.get('envelope_state', False)
    return False

def set_envelope_active(user_id):
    user_ref = db.collection('users').document(user_id)
    user = user_ref.get()
    if user.exists:
        user_ref.update({'envelope_state': True})

#envelope is clicked on
def send_envelop_quest(location, user_id):
    if not get_envelope_state(user_id):
        quest = get_nearby_quests(user_id, location)
        if quest :
            set_envelope_active(user_id)
        return quest
    return None
    
    