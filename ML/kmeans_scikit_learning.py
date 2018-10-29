# Result by using library scikit-learning
# To check more, we compare result before with result that we get by using library scikit-learning
from sklearn.cluster import KMeans
kmeans = KMeans(n_cluster = 3, random_state = 0).fix(X)
print('Centers found by kmeans_scikit:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label) 