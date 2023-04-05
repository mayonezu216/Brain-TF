import csv
import pickle


class primaryDetails:
    def __init__(self, Node, Network):
        self.Node = Node
        self.Network = Network
        # self.gender = gender
        # self.contactDetails = contactDetails

    def __str__(self):
        return "{} {}".format(int(self.Node), int(self.Network))

    def __iter__(self):
        return iter([self.Node, self.Network])

class contactDetails:
    def __init__(self, roi, cls):
        self.roi = roi 
        self.cls = cls
        # self.Location = Location

    def __str__(self):
        return "{} {} {}".format(self.cellNum, self.phNum, self.Location)

    def __iter__(self):
        return iter([self.cellNum, self.phNum, self.Location])

a_list = []
data = {}
# from csv
# with open("/fast/beidi/Com-BrainTF/shen_268_parcellation_networklabels.csv", "r") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         # a = contactDetails(row[3], row[4], row[5])
#         # a_list.append(row[0])
#         if row[0]=='Node':
#             continue
#         data[int(row[0])]=int(row[1])
#     print(data)

# from DMON
a_list = [7, 7, 7, 4, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7, 4, 7, 7, 7, 7, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 6, 1, 6, 6, 7, 7, 7, 6, 1, 7, 3, 3, 6, 7, 6, 6, 6, 6, 4, 6, 7, 7, 7, 7, 7, 6, 0, 1, 7, 7, 0, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 7, 7, 6, 6, 7, 6, 6, 6, 6, 7, 5, 7, 7, 3, 6, 1, 4, 0, 7, 0, 0, 7, 0, 5, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 7, 7, 6, 7, 4, 7, 0, 5, 0, 7, 7, 2, 7, 0, 7, 3, 7, 0, 7, 7, 7, 6, 7, 4, 7, 7, 7, 7, 7, 7, 4, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 0, 1, 6, 6, 7, 7, 6, 6, 6, 7, 3, 3, 6, 7, 6, 6, 7, 6, 7, 3, 3, 6, 7, 0, 7, 7, 6, 3, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 6, 6, 6, 1, 5, 1, 7, 3, 7, 6, 6, 3, 6, 6, 6, 3, 6, 6, 3, 5, 6, 2, 3, 5, 0, 5, 0, 5, 0, 6, 0, 7, 5, 6, 0, 5, 6, 5, 0, 5, 5, 5, 0, 0, 7, 2, 7, 7, 4, 0, 0, 6, 7, 7, 0, 2]
a_list = [5, 10, 6, 4, 6, 5, 2, 6, 2, 5, 0, 5, 5, 6, 1, 0, 6, 9, 0, 7, 4, 1, 7, 6, 5, 5, 5, 4, 0, 7, 7, 3, 5, 0, 0, 7, 11, 5, 5, 5, 3, 5, 7, 11, 0, 5, 5, 6, 5, 0, 7, 5, 11, 5, 6, 11, 5, 11, 7, 11, 5, 5, 5, 4, 7, 5, 5, 5, 5, 0, 5, 5, 5, 11, 5, 11, 5, 11, 5, 5, 11, 5, 7, 3, 6, 6, 5, 5, 1, 6, 0, 7, 4, 7, 7, 7, 11, 7, 7, 6, 10, 10, 11, 7, 10, 11, 4, 11, 9, 10, 4, 9, 10, 9, 9, 4, 5, 11, 9, 9, 1, 5, 9, 10, 6, 0, 0, 0, 3, 11, 5, 5, 3, 6, 9, 9, 6, 6, 9, 6, 6, 9, 9, 0, 6, 1, 2, 6, 4, 6, 10, 9, 7, 0, 0, 1, 1, 5, 7, 5, 6, 1, 5, 4, 6, 2, 5, 7, 7, 5, 5, 5, 7, 5, 11, 5, 6, 11, 5, 7, 5, 6, 6, 3, 5, 11, 5, 5, 11, 5, 6, 6, 6, 3, 7, 11, 5, 11, 6, 9, 9, 3, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 10, 5, 5, 0, 3, 6, 5, 7, 6, 6, 11, 6, 5, 6, 7, 1, 5, 7, 0, 7, 11, 7, 10, 10, 6, 10, 10, 10, 2, 11, 0, 0, 6, 10, 10, 7, 6, 9, 1, 10, 0, 11, 6, 7, 1, 9, 9, 9, 0, 10, 0, 11, 5, 5, 7]
for i in range(268):
    if a_list[i]>8:
        a=a_list[i]-1
    else:
        a=a_list[i]
    data[i] = a
sorted_data = dict(sorted(data.items(), key=lambda x: x[1]))
print(sorted_data)

file = open('node_clus_map_HCP_DMON12.pickle', 'wb')
with open('node_clus_map_HCP_DMON12.pickle', 'wb') as output_file:
    pickle.dump(sorted_data, output_file)

file.close()