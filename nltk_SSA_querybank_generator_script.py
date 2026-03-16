import pandas as pd
import random
import json
from datetime import timedelta
import datetime
import sys
from mpi4py import MPI
from nltk.corpus import wordnet
from itertools import chain
from sklearn.model_selection import train_test_split

# Helper to find synonyms for variety (Requires: nltk.download('wordnet'))
def get_synonym(word):
    try:
        syns = wordnet.synsets(word)
        if not syns: return word
        lemmas = [l.name().replace('_', ' ') for s in syns for l in s.lemmas()]
        return random.choice(list(set(lemmas))) if lemmas else word
    except:
        return word

def get_matched_entity(df_satcat, sat_entity):
    """Returns (satellite_name, norad_id) tuple for any entity input."""
    if sat_entity in df_satcat['SATNAME'].values:
        row = df_satcat[df_satcat['SATNAME'] == sat_entity].iloc[0]
        return row['SATNAME'], str(row['NORAD_CAT_ID'])
    elif sat_entity in df_satcat['NORAD_CAT_ID'].astype(str).values:
        row = df_satcat[df_satcat['NORAD_CAT_ID'].astype(str) == sat_entity].iloc[0]
        return row['SATNAME'], str(row['NORAD_CAT_ID'])
    else:
        # Fallback: pick random complete pair
        row = df_satcat.sample(1).iloc[0]
        return row['SATNAME'], str(row['NORAD_CAT_ID'])

def generate_multi_labeled_dataset(master_seed=999, step=1, total_target=2000):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    queries_per_core = total_target // size
    random.seed(hash((master_seed, rank, step)))

    # --- 1. Load Entities from SATCAT ---
    try:
        df_satcat = pd.read_csv('satcat04012026.csv')
        sat_names = df_satcat['SATNAME'].unique().tolist()
        norad_ids = df_satcat['NORAD_CAT_ID'].astype(str).unique().tolist()
        all_entities = sat_names + norad_ids  # Combined pool for query variety
    except Exception as e:
        if rank == 0: print(f"Error loading SATCAT: {e}")
        return

    # --- 2. Parameters ---
    now = datetime.datetime.now()
    past_dates = list(chain(
        ((now - timedelta(days=d)).strftime("%B %d, %Y") for d in range(1, 30)),
        ((now + timedelta(days=d)).strftime("%B %d, %Y") for d in range(1, 3))
    ))
    future_dates = ["next week", "next month", "in 2027", "2028"]
    intros = ["I'm looking for help with", "Can you please", "I need to", "Would you mind helping me", "Is it possible to"]
    
    local_data = []

    for _ in range(queries_per_core):
        roll = random.random()
        sat_entity = random.choice(all_entities)  # Select from combined pool
        satellite_name, norad_id = get_matched_entity(df_satcat, sat_entity)
        intro = get_synonym(random.choice(intros))
        
        # Initialize column values
        current_satname = None
        current_norad = None
        current_time = None

        # --- TIER 1: SUFFICIENT ---
        if roll < 0.34:
            dt = random.choice(past_dates)
            current_satname = satellite_name
            current_norad = norad_id
            current_time = dt
            
            # Refined Intent Map (uses sat_entity for query text variety)
            intent_map = {
                "orbital_data_retrieval": [
                    f"{intro} {get_synonym('retrieve')} the orbital data for {sat_entity} on {dt}.",
                    # f"Get the state vector of {sat_entity} for the timestamp {dt}.",
                    # f"Show me the ephemeris for {sat_entity} on {dt}."
                ],
                "conjunction_assessment": [
                    f"For {sat_entity}, {get_synonym('calculate')} the collision probability recorded on {dt}.",
                    f"Can you {get_synonym('check')} the conjunction assessment for {sat_entity} during {dt}?",
                    f"What was the risk of collision for {sat_entity} on {dt}?",
                    f"Assess the conjunction event for {sat_entity} that happened on {dt}."
                ],
                "proximity_search": [
                    f"Find all possible colliding space objects near {sat_entity} on {dt}.",
                    f"What satellites or debris were close to {sat_entity} on {dt}?",
                    # f"Identify any objects that might have threatened {sat_entity} during {dt}.",
                    f"Are there any recorded close approaches for {sat_entity} on {dt}?",
                    # f"Check for any debris in the vicinity of {sat_entity} on {dt}."
                ],
                "tle_extraction": [
                    f"{intro} {get_synonym('extract')} the TLE parameters for {sat_entity} from {dt}.",
                    f"Give me the Two-Line Element set for {sat_entity} for {dt}.",
                    f"Fetch the historical TLE for {sat_entity} on {dt}."
                ]
            }
            intent = random.choice(list(intent_map.keys()))
            query = random.choice(intent_map[intent])
            local_data.append({
                "query": query, 
                "label": "sufficient", 
                "intent": intent,
                "satellite_name": current_satname,
                "norad_id": current_norad,
                "time": current_time,
                "action": "trigger_tool"
            })

        # --- TIER 2: INSUFFICIENT ---
        elif roll < 0.67:
            if random.random() > 0.5:
                # Intent: Future Prediction Attempt
                f_dt = random.choice(future_dates)
                query = f"{intro} calculate a collision risk for {sat_entity} {f_dt}."
                intent = "invalid_temporal_constraint"
                current_satname = satellite_name
                current_norad = norad_id
                current_time = f_dt
            else:
                # Intent: Missing Slots
                dt_temp = random.choice(past_dates)
                cases = [
                    (f"{intro} analyze the orbital apogee for {dt_temp}.", "missing_entity_id", None, None, dt_temp), 
                    (f"I'm trying to fetch the TLE for {sat_entity}.", "missing_temporal_constraint", satellite_name, norad_id, None),
                    (f"Can you verify if there is any risk for the satellite?", "missing_all_metadata", None, None, None),
                    (f"Find objects that were near the satellite on {dt_temp}.", "missing_entity_id", None, None, dt_temp)
                ]
                query, intent, current_satname, current_norad, current_time = random.choice(cases)
            
            local_data.append({
                "query": query, 
                "label": "insufficient", 
                "intent": intent,
                "satellite_name": current_satname,
                "norad_id": current_norad,
                "time": current_time,
                "action": "trigger_probing"
            })

        # --- TIER 3: OUT OF SCOPE ---
        else:
            if random.random() > 0.4:
                # Intent: Entity Factoid
                query = random.choice([
                    f"Tell me about the history of {sat_entity}.",
                    f"Who launched {sat_entity}?",
                    f"Is {sat_entity} still in service?",
                    f"What is the country of origin for {sat_entity}?",
                    f"Which launch site was used for {sat_entity}?"
                ])
                intent = "entity_information_request"
                current_satname = satellite_name
                current_norad = norad_id
                current_time = None
            else:
                # Intent: General SSA/Space Knowledge
                query = random.choice([
                    "What is the Kessler Syndrome?",
                    "How many satellites are in LEO?",
                    "Define the term 'Graveyard Orbit'.",
                    "How often do satellites collide?",
                    "What happens to dead satellites?",
                    "How do we track space junk?",
                    "Explain the difference between LEO and GEO."
                ])
                intent = "general_knowledge_query"
                current_satname = None
                current_norad = None
                current_time = None
            
            local_data.append({
                "query": query, 
                "label": "out_of_scope", 
                "intent": intent,
                "satellite_name": current_satname,
                "norad_id": current_norad,
                "time": current_time,
                "action": "trigger_llm_response"
            })

    # Gather to Rank 0
    all_data = comm.gather(local_data, root=0)

    if rank == 0:
        # 1. Flatten the gathered data
        final_list = [item for sublist in all_data for item in sublist]
        
        # 2. Ensure uniqueness based on the query string
        unique_dict = {v['query']: v for v in final_list}
        unique_list = list(unique_dict.values())
        
        # 3. Create a DataFrame
        df = pd.DataFrame(unique_list)

        # 4. Perform Stratified Split
        train_df, test_df = train_test_split(
            df, 
            test_size=0.20, 
            random_state=master_seed, 
            stratify=df['label']
        )

        # 5. Hard Uniqueness Check
        train_queries = set(train_df['query'])
        test_df = test_df[~test_df['query'].isin(train_queries)]

        # 6. Save separate files
        train_filename = f"SSA_QueryBank_Train_Step_{step}.json"
        test_filename = f"SSA_QueryBank_Test_Step_{step}.json"

        with open(train_filename, "w") as f:
            json.dump(train_df.to_dict(orient="records"), f, indent=4)
            
        with open(test_filename, "w") as f:
            json.dump(test_df.to_dict(orient="records"), f, indent=4)

        print(f"Merge and Split Complete.")
        print(f"Total Unique: {len(df)}")
        print(f"Training Set: {len(train_df)} saved to {train_filename}")
        print(f"Testing Set: {len(test_df)} saved to {test_filename}")
        print(f"Overlap Verification: {len(set(train_df['query']) & set(test_df['query']))} common queries.")

if __name__ == "__main__":
    step_val = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    generate_multi_labeled_dataset(step=step_val, total_target=2000)
