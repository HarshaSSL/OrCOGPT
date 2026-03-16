import json
import glob
import os

file_type = "Test"

def merge_and_deduplicate(pattern=f"SSA_QueryBank_{file_type}_Step_*.json", output_name=f"SSA_QueryBank_new_{file_type}_set.json"):
    files = glob.glob(pattern)
    # print(files)
    print(f"Found {len(files)} files to merge.")
    
    master_list = []
    seen_queries = set()
    duplicate_count = 0

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            for entry in data:
                # Normalize string to catch exact duplicates regardless of whitespace
                query_text = entry['query'].strip().lower()
                
                if query_text not in seen_queries:
                    seen_queries.add(query_text)
                    master_list.append(entry)
                else:
                    duplicate_count += 1

    # Save final cleaned dataset
    with open(output_name, 'w') as f:
        json.dump(master_list, f, indent=4)

    print("\n" + "="*40)
    print(f"MERGE COMPLETE")
    print("="*40)
    print(f"Total Unique Queries: {len(master_list)}")
    print(f"Duplicates Removed:   {duplicate_count}")
    print(f"Uniqueness Rate:      {(len(master_list)/(len(master_list)+duplicate_count))*100:.2f}%")
    print(f"Final Path:           {os.path.abspath(output_name)}")
    print("="*40)

if __name__ == "__main__":
    merge_and_deduplicate()