from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os

classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test[:2000])
    acc = accuracy_score(y_test[:2000], y_pred)
    print(f"{name} Accuracy: {acc:.3f}")

    cm = confusion_matrix(y_test[:2000], y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)

    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/confusion_matrix_{name.lower()}.png")
    plt.close()

    return acc