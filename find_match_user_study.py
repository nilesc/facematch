import sys
import sqlite3
import random
import numpy as np
from PIL import Image
from setup_database import adapt_array, convert_array
from face_embed import Embedder
from pose_estimator import PoseEstimator
import face_recognition
from helpers import resize_image


def find_n_closest(options, target, n):
    options = options.reshape(options.shape[0], -1)
    repeated_target = np.repeat(target, options.shape[0], 0)
    difference = options - repeated_target
    norms = np.linalg.norm(difference, axis=1)
    closest = np.argpartition(norms, 1)
    return closest[:n]


# Math based on code from:
# https://stackoverflow.com/questions/2827393/
# angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def find_smallest_angle_difference(options, target):
    # Conver target to a unit vector
    target = target / np.linalg.norm(target)

    # Change options into unit vectors
    options = options.reshape(options.shape[0], -1)
    lengths = np.linalg.norm(options, axis=1)
    lengths = np.reshape(lengths, (-1, 1))
    lengths = np.repeat(lengths, 3, 1)
    options = options / lengths

    dots = np.tensordot(options, target, axes=([1], [1]))
    clipped = np.clip(dots, -1.0, 1.0)
    differences = np.arccos(clipped)
    results = []
    min_ = np.argmin(differences)
    results.append(min_)
    for x in range(3):
        rand_num = random.random(0,len(differences))
        while rand_num == min_ or rand_num in results:
            rand_num = random.random(0,len(differences))
        results.append(rand_num)



def get_best_match(conn, embedder, pose_estimator, image):
    image_to_embed = np.expand_dims(np.array(image), 0)
    input_embedding = embedder.embed(image_to_embed)
    input_pose = pose_estimator.estimate_pose(image)

    c = conn.cursor()
    c.execute('SELECT embedding FROM videos')
    embeddings = c.fetchall()
    embeddings = np.array(embeddings)
    num_people = 5
    candidates = find_n_closest(embeddings, input_embedding, num_people)

    query = "SELECT image_path, pose FROM frames WHERE video_id IN " + \
            str(tuple(candidates))

    c.execute(query)
    paths, poses = zip(*c.fetchall())
    best_frame_index = find_smallest_angle_difference(np.array(poses),
                                                      input_pose)
    return paths[best_frame_index]


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
    as_array = np.array(input_image)
    possible_bounds = face_recognition.api.face_locations(as_array)
    input_image_bounds = list(possible_bounds[0])
    rotated = input_image_bounds[-1:] + input_image_bounds[:-1]
    input_image = input_image.crop(rotated)
    embedding_image = resize_image(input_image, 160)
    embedding_image = Image.fromarray(embedding_image[0].astype('uint8'),
                                      'RGB')

    results = get_best_match(conn, embedder, pose_estimator, image_array)
    for item in results:
        print(item)
