import numpy as np
import matplotlib.pyplot as plt
import zipfile
from PIL import Image, ImageOps
def loadimg(filezip):
    imgs = []
    with zipfile.ZipFile(filezip, 'r') as archive:
        for filename in archive.namelist():
            with archive.open(filename) as img_file:
                img = Image.open(img_file)
                img = img.resize((64, 64), resample=Image.BILINEAR)
                imgdata = np.array(img, dtype=np.float32)
                imgdata = imgdata.flatten()
                imgs.append(imgdata)
    return np.array(imgs)


filezip = "afhq_cat.zip"
X = loadimg(filezip)

Xred = X[:, :4096]
Xgreen = X[:, 4096:8192]
Xblue = X[:, 8192:]

def calcpca(data, n_components):
    meanpc = np.mean(data, axis=0)
    centerdata = data - meanpc
    covmat = np.cov(centerdata.T)
    eigenvalue, eigenvector = np.linalg.eig(covmat)
    sorted_indices = np.argsort(-eigenvalue)[:n_components]
    seleigenvalues = eigenvalue[sorted_indices]
    seleigenvectors = eigenvector[:, sorted_indices]
    return meanpc, seleigenvalues, seleigenvectors

def calcpve(eigenvalues):
    return eigenvalues / np.sum(eigenvalues)

def cumulative_pve(eigenvalues, threshold=0.7):
    calcpvevalues = calcpve(eigenvalues)
    cumpve = np.cumsum(calcpvevalues)
    return np.argmax(cumpve >= threshold) + 1

ncomponent = 10

meanred, eigenvaluesred, eigenvectorsred = calcpca(Xred, ncomponent)
meangreen, eigenvaluesgreen, eigenvectorsgreen = calcpca(Xgreen, ncomponent)
meanblue, eigenvaluesblue, eigenvectorsblue = calcpca(Xblue, ncomponent)

print("PVE for Red channel: ", calcpve(eigenvaluesred))
print("PVE for Green channel: ", calcpve(eigenvaluesgreen))
print("PVE for Blue channel: ", calcpve(eigenvaluesblue))

#FIND MIN NU OF PRINCIPAL COMP
minpcred = cumulative_pve(eigenvaluesred)
minpcgreen = cumulative_pve(eigenvaluesgreen)
minpcblue = cumulative_pve(eigenvaluesblue)

print("Minimum number of PCs for Red: ", minpcred)
print("Minimum number of PCs for Green: ", minpcgreen)
print("Minimum number of PCs for Blue:", minpcblue)

def disppc(eigenvectors, channel_name):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axes):
        pc = eigenvectors[:, i].reshape((64, 64))
        pc_normalized = (pc - np.min(pc)) / (np.max(pc) - np.min(pc))
        ax.imshow(pc_normalized, cmap='gray')
        ax.set_title(f"PC {i+1}")
        ax.axis('off')
    plt.suptitle(f"{channel_name} Channel PC")
    plt.show()

disppc(eigenvectorsred, "Red")
disppc(eigenvectorsgreen, "Green")
disppc(eigenvectorsblue, "Blue")

def reconstimg(mean, eigenvectors, originalimg, k):
    centereddata = originalimg - mean
    projection = np.dot(centereddata, eigenvectors[:, :k])
    reconsdata = np.dot(projection, eigenvectors[:, :k].T) + mean
    return reconsdata

imgindex = 1
kvalues = [1, 50, 250, 500, 1000, 4096]

reconsimages = []
for k in kvalues:
    reconsred = reconstimg(meanred, eigenvectorsred, Xred[imgindex], k)
    reconsgreen = reconstimg(meangreen, eigenvectorsgreen, Xgreen[imgindex], k)
    reconsblue = reconstimg(meanblue, eigenvectorsblue, Xblue[imgindex], k)
    reconsimage = np.stack([reconsred, reconsgreen, reconsblue], axis=-1).reshape(64, 64, 3)
    reconsimages.append(reconsimage)

fig, axes = plt.subplots(1, len(kvalues), figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(reconsimages[i].astype(np.uint8))
    ax.set_title(f"k = {kvalues[i]}")
    ax.axis('off')
plt.suptitle("Reconstructed Image")
plt.show()

