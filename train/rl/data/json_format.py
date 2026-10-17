import json

path = "filtered_2.json"
output_path = "filtered_2.json"

with open(path) as f:
    data = json.load(f)
    
# order by id
# data = sorted(data, key=lambda x: x["id"])

# with open(output_path, "w") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)
# path = "coin_filtered1.json"
# with open(path) as f:
#     data = json.load(f)
    
for i, item in enumerate(data):
    item["id"] = i
    
data = sorted(data, key=lambda x: x["id"])
with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
    
# output_path = "coin_filtered.json"
# with open(output_path, "r") as f:
#     data = json.load(f)
#     print(len(data))
