import json
import regex as re
import numpy as np
#import string
import matplotlib.pyplot as plt

def readData(data):
    A = {}
    for key in data.keys():
        print(key)
        name = key
        for key2 in data[key]:
            
            #print(key2)
            print(key2['file_name'])
            url = key2["base_url"]
            name = key2['file_name']
            camNumber = re.search(r'\d+', name).group()
            print(camNumber)
            dataset = url.split("/")[-2]
            myname = dataset + "-" + camNumber
            print(myname)
            A[myname] = [ key2["iterations"], key2["bestCost"],key2["bestCost60"], key2["bestCost30"], key2["kClusters"] ]

            #   print(key2["iterations"])
            #   print(key2["bestCost"])
            #   print(key2["kClusters"])
            #   print(key2["bestIt"])
            #   print(key2["bestCost60"])
            #   print(key2["bestCost30"])

            #   if type(key2) != dict:
            #     print(key2)
            #   else:
            #       for p in data[key][key2]:
            #           print(p)
    return A

# Open and read the JSON file
with open('test.json', 'r') as file:
    data = json.load(file)
with open('test2.json', 'r') as file:
    data2 = json.load(file)

A = readData(data)
B = readData(data2)

A = dict(sorted(A.items()))
B = dict(sorted(B.items()))

# Print the data

#print(data)
#print(data['AAA'])
#print(data.keys()[0])

#print("A", A)
#print("B", B)

# plot 
idx = 0
fig, axs = plt.subplots(4, 8, figsize=(40, 15))
for ds, v in A.items():
    print(ds, " ", v)
    if not ds in B:
        continue
    v2 = B[ds]

    dataset = {}
    dataset['a'] = v[1:4]
    dataset['b'] = v2[1:4]

    index = range(len(dataset['a']))
    bar_width = 0.4

    ax = axs.flatten()[idx]
    idx = idx + 1
    # Bar plots for 'a' and 'b'
    bars_a = ax.bar(index, dataset['a'], bar_width, label='a')
    bars_b = ax.bar([i + bar_width for i in index], dataset['b'], bar_width, label='b')
    print("v", dataset['a'], " vs ", dataset['b'])
    # Add labels and title
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(ds)
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(['90', '60', '30'])
    # ax.legend() # not per image . better not as subplots / no boundary

plt.subplots_adjust(wspace=0.1)
plt.tight_layout()
plt.savefig("mygraph.png")
#plt.show()

############## new plot: all in one ?!
# Flatten data for combined bar plot
values_a = []
values_b = []
names = []
for ds, v in A.items():
    print(ds, " ", v)
    if not ds in B:
        continue
    v2 = B[ds]
    values_a.extend(v[1:4])
    values_b.extend(v2[1:4])
    names.append(ds)
#values_a.flatten()
#values_b.flatten()

# Create an array of x indices for each dataset
x_indices = np.arange(len(values_a))
#x_indices = np.linspace(0, len(values_a) - 1, len(values_a))

print(len(values_a), x_indices.shape)
its = [90,60,30]
# Bar width
bar_width = 0.4
spacing = 0.00

# Create the bar plot
fig, ax = plt.subplots(figsize=(30, 10))

#bars_a = ax.bar(x_indices - bar_width / 2, values_a, bar_width, label='a')
#bars_b = ax.bar(x_indices + bar_width / 2, values_b, bar_width, label='b')
bars_a = ax.bar(x_indices * (1 + spacing) - bar_width / 2, values_a, bar_width, label='a')
bars_b = ax.bar(x_indices * (1 + spacing) + bar_width / 2, values_b, bar_width, label='b')

# Add labels and title
ax.set_xlabel('Datasets')
ax.set_ylabel('Values')
ax.set_title('Combined Bar Plot for 30 Datasets')


ax.set_xticks(x_indices)
#ax.set_xticklabels([f'Dataset {i//3 + 1} Bar {i%3 + 1}' for i in range(len(values_a))], rotation=90)
ax.set_xticklabels([f"{names[i//3]} {its[i%3]}" for i in range(len(values_a))], rotation=90)
#ax.set_xticks(x_indices)
#ax.set_xticklabels([f"{names[i]} {i%3 + 1}" for i in range(len(values_a)//3)], rotation=90)

ax.set_xticks(x_indices * (1 + spacing))
ax.set_xticklabels([f"{names[i//3]} {its[i%3]}" for i in range(len(values_a))], rotation=90)

ax.legend()

# Adjust layout to prevent overlap and remove extra space
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
                    
# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("mygraph2.png")

# Show the plot
