"""

 similar_images_TL.py  (author: Anson Wong / git: ankonzoid)

 We find similar images in a database by using transfer learning
 via a pre-trained VGG image classifier. We plot the 5 most similar
 images for each image in the database, and plot the tSNE for all
 our image feature vectors.

"""
import sys, os
import numpy as np
import caffe

from model import Model
from imagenet_utils import get_transformer
from plot_utils import plot_query_answer
from sort_utils import find_topk_unique
from kNN import kNN
from tSNE import plot_tsne

caffe_root = '/home/wanggh/caffe/'


def main():
    # ================================================
    # Load pre-trained model and remove higher level layers
    # ================================================
    print("Loading pre-trained model...")
    caffe.set_mode_cpu()
    model = Model(caffe_root)
    model.model_init()

    # ================================================
    # Read images and convert them to feature vectors
    # ================================================
    inputdata_shape = model.get_input_datashape()
    transformer = get_transformer(inputdata_shape, caffe_root)
    # print('inputdata shape',inputdata_shape)
    imgs, filename_heads, X = [], [], []
    path = "../db"
    print("Reading images from '{}' directory...\n".format(path))
    for f in os.listdir(path):
        # Process filename
        filename = os.path.splitext(f)  # filename in directory
        filename_full = os.path.join(path,f)  # full path filename
        head, ext = filename[0], filename[1]
        if ext.lower() not in [".jpg", ".jpeg"]:
            continue
        print('read', head)
        # Read image file
        image = caffe.io.load_image(filename_full)
        transformed_image = transformer.preprocess('data', image)

        imgs.append(np.array(image))  # image
        filename_heads.append(head)  # filename head

        # Pre-process for model input
        features = model.get_feature(transformed_image)  # features
        X.append(features.copy())  # append feature extractor

    # all X are the same....
    X = np.array(X)  # feature vectors
    imgs = np.array(imgs)  # images
    print("imgs.shape = {}".format(imgs.shape))
    print("X_features.shape = {}\n".format(X.shape))

    # ===========================
    # Find k-nearest images to each image
    # ===========================
    n_neighbours = 5 + 1  # +1 as itself is most similar
    knn = kNN()  # kNN model
    knn.compile(n_neighbors=n_neighbours, algorithm="brute", metric="cosine")
    knn.fit(X)

    # ==================================================
    # Plot recommendations for each image in database
    # ==================================================
    output_rec_dir = os.path.join("../output", "rec")
    if not os.path.exists(output_rec_dir):
        os.makedirs(output_rec_dir)
    n_imgs = len(imgs)
    ypixels, xpixels = imgs[0].shape[0], imgs[0].shape[1]
    for ind_query in range(n_imgs):

        # Find top-k closest image feature vectors to each vector
        print("[{}/{}] Plotting similar image recommendations for: {}".format(ind_query+1, n_imgs, filename_heads[ind_query]))
        distances, indices = knn.predict(np.array([X[ind_query]]))
        distances = distances.flatten()
        indices = indices.flatten()
        indices, distances = find_topk_unique(indices, distances, n_neighbours)

        # Plot recommendations
        rec_filename = os.path.join(output_rec_dir, "{}_rec.png".format(filename_heads[ind_query]))
        x_query_plot = imgs[ind_query].reshape((-1, ypixels, xpixels, 3))
        x_answer_plot = imgs[indices].reshape((-1, ypixels, xpixels, 3))
        plot_query_answer(x_query=x_query_plot,
                          x_answer=x_answer_plot[1:],  # remove itself
                          filename=rec_filename)

    # ===========================
    # Plot tSNE
    # ===========================
    output_tsne_dir = os.path.join("../output")
    if not os.path.exists(output_tsne_dir):
        os.makedirs(output_tsne_dir)
    tsne_filename = os.path.join(output_tsne_dir, "tsne.png")
    print("Plotting tSNE to {}...".format(tsne_filename))
    plot_tsne(imgs, X, tsne_filename)

# Driver
if __name__ == "__main__":
    main()