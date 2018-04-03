import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import warnings
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

def k_nearest_neighbors(data,predict,k=3):
    if len(data)>=k:
        warnings.warn('The given k value is less than the no. of groups')
    distances = []
    for group in data:
        for feature in data[group]:
            euclidian_dist = np.linalg.norm(np.array(feature)-np.array(predict))
            distances.append([euclidian_dist,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result        

result = k_nearest_neighbors(dataset,new_features)
[[plt.scatter(j[0],j[1],c=i) for j in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],c=result)
plt.show()
print(result)