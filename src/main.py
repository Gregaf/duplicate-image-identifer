import logging
from typing import List

import numpy as np
import tensorflow as tf
import networkx as nx
from keras.applications import ResNet152
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image
from utils.args import parse_args
from utils.logger import config_logger
from matplotlib import pyplot as plt
from scipy.spatial import distance, KDTree
import os


class ImageProcessor:
    def __init__(self) -> None:
        base_model = ResNet152(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        self.model = base_model
        # self.model = tf.keras.Model(
        #     inputs=base_model.input,
        #     outputs=base_model.get_layer("conv5_block3_out").output,
        # )
        pass

    def extract_features(self, img_path: str) -> List[float]:
        """
        Extract features from an image using a pre-trained MobileNetV2 model.

        Parameters:
        img_path (str): Path to the image file.

        Returns:
        List[float]: The extracted features as a 1D array.
        """
        if not os.path.exists(img_path):
            logging.error(f"Image file not found: {img_path}")
            return []

        try:
            img = image.load_img(img_path, target_size=(224, 224))
            # plt.imshow(img)
            # plt.show()
        except Exception as e:
            logging.error(f"Failed to load image: {img_path}. Error: {e}")
            return []

        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        features = self.model.predict(img_array)
        return features.flatten()


def cosine_similarity(featuresA, featuresB):
    return distance.cosine(featuresA, featuresB)


def get_image_paths(dir_path: str) -> List[str]:
    image_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def find_similar_images(
    features, image_file_paths, use_cosine_similarity=False, k=5, threshold=0.2
):
    G = nx.Graph()

    for i, feature in enumerate(features):
        # Create a new list of features and file paths excluding the current image
        other_features = features[:i] + features[i + 1 :]
        other_image_file_paths = image_file_paths[:i] + image_file_paths[i + 1 :]

        if use_cosine_similarity:
            # Calculate cosine distances to all other features
            distances = [
                distance.cosine(feature, other_feature)
                for other_feature in other_features
            ]

            # Get the indices of the k smallest distances
            nearest_indices = np.argsort(distances)[:k]

            # Filter out distances greater than the threshold
            similar_image_paths = [
                other_image_file_paths[j]
                for j in nearest_indices
                if distances[j] <= threshold
            ]
            similar_distances = [
                round(distances[j], 2)
                for j in nearest_indices
                if distances[j] <= threshold
            ]

            if similar_image_paths:
                G.add_node(image_file_paths[i])
                for similar_image_path, similar_distance in zip(
                    similar_image_paths, similar_distances
                ):
                    G.add_edge(
                        os.path.basename(image_file_paths[i]),
                        os.path.basename(similar_image_path),
                        weight=similar_distance,
                    )

                logging.info(
                    f"Image {image_file_paths[i]} is most similar to images {similar_image_paths} with distances {similar_distances}"
                )
        else:
            tree = KDTree(other_features)

            k = min(k, len(other_features))
            dist, ind = tree.query([feature], k=k)
            # Exclude neighbors with distance greater than the threshold
            similar_image_paths = [
                other_image_file_paths[j] for j in ind[0] if dist[0][j] <= threshold
            ]
            similar_distances = [dist[0][j] for j in ind[0] if dist[0][j] <= threshold]

            if similar_image_paths:
                # Add the current image as a node in the graph
                G.add_node(image_file_paths[i])

                # Add an edge between the current image and each similar image
                for similar_image_path, similar_distance in zip(
                    similar_image_paths, similar_distances
                ):
                    G.add_edge(
                        image_file_paths[i], similar_image_path, weight=similar_distance
                    )

                logging.info(
                    f"Image {image_file_paths[i]} is most similar to images {similar_image_paths} with distances {similar_distances}"
                )

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, "weight")
    # Create a list to store the colors of the nodes
    colors = []

    # Iterate over the nodes in the graph
    for node in G.nodes():
        # If the node has more than one edge, color it red. Otherwise, color it blue.
        if len(G.edges(node)) > 1:
            colors.append("red")
        else:
            colors.append("blue")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def main():
    config_logger()

    args = parse_args()
    logging.debug(f"Target directory was: '{args.dir}'")

    image_file_paths = get_image_paths(args.dir)
    logging.info(f"Found '{len(image_file_paths)}' image files in the directory")
    logging.debug(f"Image file paths: {image_file_paths}")

    image_processor = ImageProcessor()
    logging.info("Initialized the image processor")

    logging.info("Extracting features from images...")

    features = []
    for img_path in image_file_paths:
        features.append(image_processor.extract_features(img_path))

    find_similar_images(features, image_file_paths, use_cosine_similarity=True, k=2)

    # tree = KDTree(features)

    # threshold = 0.5
    # k = min(5, len(features))
    # for i, feature in enumerate(features):
    #     dist, ind = tree.query([feature], k=k)
    #     similar_image_paths = [image_file_paths[j] for j in ind[0]]
    #     logging.info(
    #         f"Image {image_file_paths[i]} is most similar to images {similar_image_paths} with distances {dist}"
    #     )

    # features = image_processor.extract_features(image_file_paths[0])
    # features2 = image_processor.extract_features(image_file_paths[1])

    # similarity = cosine_similarity(features, features2)
    # logging.info(f"Similarity between images: {similarity}")


if __name__ == "__main__":
    main()
