from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_models(X_train, y_train, sample_size=10000):
    X_tr = X_train[:sample_size]
    y_tr = y_train[:sample_size]

    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_tr, y_tr)

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_tr, y_tr)

    return svm, knn