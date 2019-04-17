import sys
import sqlite3
import numpy as np
from PIL import Image
from setup_database import adapt_array, convert_array
from face_embed import Embedder
from pose_estimator import PoseEstimator


def find_n_closest(options, target, n):
    options = options.reshape(options.shape[0], -1)
    repeated_target = np.repeat(target, options.shape[0], 0)
    difference = options - repeated_target
    norms = np.linalg.norm(difference, axis=1)
    closest = np.argpartition(norms, 1)
    print("closest", closest)
    return closest[-n:]


def get_best_match(conn, embedder, pose_estimator, image):
    input_embedding = embedder.embed(input_image)
    input_pose = pose_estimator.estimate_pose(input_image)

    c = conn.cursor()
    c.execute('SELECT embedding FROM videos')
    embeddings = c.fetchall()
    embeddings = np.array(embeddings)
    num_people = 5
    candidates = find_n_closest(embeddings, input_embedding, num_people)
    print("Candidates:", candidates)

    query = "SELECT image_path, pose FROM frames WHERE video_id IN " + \
            str(tuple(candidates))

    c.execute(query)
    paths, poses = zip(*c.fetchall())
    best_frame_index = find_n_closest(np.array(poses), input_pose, 1)
    return paths[best_frame_index[0]]


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit('Requires a database, an embedder weights file,' +
                 'and a pose estimator weights file')

    database_file = sys.argv[1]
    facenet_protobuf = sys.argv[2]
    pose_weights = sys.argv[3]
    input_image_path = sys.argv[4]

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    conn = sqlite3.connect(database_file, detect_types=sqlite3.PARSE_DECLTYPES)

    embedder = Embedder(facenet_protobuf)
    pose_estimator = PoseEstimator(pose_weights)
    input_image = Image.open(input_image_path)
    image_dimension = 160
    embedding_image = input_image.resize((image_dimension, image_dimension))
    embedding_image = np.array(embedding_image)
    embedding_image = np.expand_dims(embedding_image, 0)
    image_array = np.array(embedding_image)
    #input_image = np.random.rand(1, 160, 160, 3)
    if len(image_array.shape) != 4:
        print("image less than 4d")
        image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], image_array.shape[2])
    print("shape:", image_array.shape)
    input_image = image_array

    print(get_best_match(conn, embedder, pose_estimator, image_array))
