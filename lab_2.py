import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.image import imread


def eigenvalues_and_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    for i in range(len(eigenvalues)):
        Av = np.dot(matrix, eigenvectors[:, i])
        lv = eigenvalues[i] * eigenvectors[:, i]

        if np.allclose(Av, lv):
            print(f"Eigenvalue {eigenvalues[i]} and eigenvector {eigenvectors[:, i]} are correct")
        else:
            print(f"Eigenvalue {eigenvalues[i]} and eigenvector {eigenvectors[:, i]} are incorrect")
    return eigenvalues, eigenvectors


my_matrix = np.array([[1, 2], [2, 1]])
eigenvalues_and_eigenvectors(my_matrix)


def pca_image_compression(image_path):
    image_raw = imread(image_path)
    print(f"Image size: {image_raw.shape}")

    plt.figure(figsize=(6, 6))
    plt.imshow(image_raw)
    plt.title("Color image")
    plt.show()

    image_bw = np.dot(image_raw[..., :3], [0.2989, 0.5870, 0.1140])
    print(f"Black and white image size: {image_bw.shape}")

    plt.figure(figsize=(6, 6))
    plt.imshow(image_bw, cmap='gray')
    plt.title("Black and white image")
    plt.show()

    pca = PCA()
    pca.fit(image_bw)  # встановлюємо головні компоненти

    var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100  # визначаємо кумулятивну дисперсію (дозволяє візуалізувати, яка частка загальної варіації у даних була вже пояснена за допомогою перших i головних компонент)

    k = np.argmax(var_cumu > 95) # визначаємо кількість компонент, які пояснюють 95% дисперсії
    print("Number of components explaining 95% variance: " + str(k))

    plt.title('Cumulative Explained Variance explained by the components')
    plt.ylabel('Cumulative Explained variance')
    plt.xlabel('Principal components')
    plt.axvline(x=k, color="k", linestyle="--")
    plt.axhline(y=95, color="r", linestyle="--")
    plt.plot(var_cumu)
    plt.show()

    ipca = PCA(n_components=k)  # встановлюємо кількість головних компонент
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))  # виконуємо зворотне перетворення після зменшення розмірності
    plt.imshow(image_recon, cmap=plt.cm.gray)  # відображаємо зображення
    plt.show()

    def plot_at_k(k, ax): # відображаємо зображення для різної кількості компонент
        ipca = PCA(n_components=k)
        image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
        ax.imshow(image_recon, cmap=plt.cm.gray)
        ax.set_title("Components: " + str(k))

    ks = [5, 15, 25, 75, 100, 170]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, ax in enumerate(axes.flat):
        plot_at_k(ks[i], ax)

    plt.show()


image_path = '/Users/olenapopova/Documents/GitHub/Lab_1/guinea pig копія.jpg'
pca_image_compression(image_path)
