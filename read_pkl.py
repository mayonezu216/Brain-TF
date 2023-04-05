import pickle
import collections
with open('/fast/beidi/Com-BrainTF/node_clus_map_HCP_DMON12.pickle', 'rb') as f:
    data = pickle.load(f)

# Create a counter object from the dictionary values
value_counts = collections.Counter(data.values())

# Print the count of each value
counts = 0
counts_list = []
for value, count in value_counts.items():
    counts += count
    print(f"{value}: {count},{counts}")
    counts_list.append(counts)
print(data)
print(counts_list)
print(type(data))