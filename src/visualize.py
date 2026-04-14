import matplotlib.pyplot as plt
import numpy as np
import os

classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

def show_predictions(model, X_test_raw, X_test, y_test):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    indices = np.random.choice(len(y_test[:2000]), 10, replace=False)

    for ax, i in zip(axes.flat, indices):
        pred = model.predict([X_test[i]])[0]
        true = y_test[i]

        ax.imshow(X_test_raw[i])
        color = 'green' if pred == true else 'red'
        ax.set_title(f"P:{classes[pred]}\nT:{classes[true]}", color=color)
        ax.axis('off')

    plt.suptitle("Predictions (Green = Correct, Red = Wrong)")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/predictions.png")
    plt.close()