import sys
import sqlite3
import random
import numpy as np
from PIL import Image
from setup_database import adapt_array, convert_array
from face_embed import Embedder
from pose_estimator import PoseEstimator
import face_recognition
from helpers import crop_to_face, get_normalized_landmarks


def find_n_closest(options, target, n):
    if not n < options.shape[0]:
        return np.arange(options.shape[0])
    norms = get_euclidean_distances(options, target)
    closest = np.argpartition(norms, n)
    return closest[:n]


def get_euclidean_distances(options, target):
    options = options.reshape(options.shape[0], -1)
    target = target.flatten()
    target = np.expand_dims(target, 0)
    repeated_target = np.repeat(target, options.shape[0], 0)
    difference = options - repeated_target
    return np.linalg.norm(difference, axis=1)


# Math based on code from:
# https://stackoverflow.com/questions/2827393/
# angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def get_angle_differences(options, target):
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
    return differences


def get_best_match(conn, embedder, pose_estimator, image):
    input_embedding = embedder.embed(image)
    input_pose = pose_estimator.estimate_pose(image)
    input_landmarks = get_normalized_landmarks(image)

    c = conn.cursor()
    c.execute('SELECT embedding FROM videos')
    embeddings = c.fetchall()
    embeddings = np.array(embeddings)
    num_people = 5
    candidates = find_n_closest(embeddings, input_embedding, num_people)

    query = 'SELECT image_path, pose, landmarks FROM frames WHERE video_id IN ' + \
            str(tuple(candidates))

    c.execute(query)
    paths, poses, landmarks = zip(*c.fetchall())
    landmarks = np.array(landmarks)
    poses = np.array(poses)
    # best_frame_index = find_smallest_angle_difference(np.array(poses),
    #                                                   input_pose)
    # return paths[best_frame_index]
    pose_differences = get_angle_differences(poses, input_pose).flatten()
    #print(f'Pose differences: {pose_differences}')
    landmark_differences = get_euclidean_distances(landmarks, input_landmarks).flatten()
    #print(f'Landmark differences: {landmark_differences}')
    combination_ratio = 0.90
    combined = combination_ratio * pose_differences + \
               (1 - combination_ratio) * landmark_differences
    #print(f'Combined: {combined}')

    best_option = np.argmin(combined)
    results = []
    results.append(paths[best_option])
    print(results[0])
    idx_list = []
    for x in range(3):
        rand_num = random.randint(0,len(combined)-1)
        while rand_num == best_option or rand_num in idx_list:
            rand_num = random.randint(0,len(combined)-1)
        rand_path = paths[rand_num]
        rand_path = rand_path.split('/')
        rand_celeb = rand_path[1]
        for names in results:
            while rand_celeb in names or rand_num in idx_list:
                rand_num = random.randint(0,len(combined)-1)
                rand_path = paths[rand_num]
                rand_path = rand_path.split('/')
                rand_celeb = rand_path[1]
        idx_list.append(rand_num)
        results.append('/'.join(rand_path))

    random.shuffle(results)
    return results


    return paths[best_option]


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
    input_image = crop_to_face(input_image)

    results = get_best_match(conn, embedder, pose_estimator, input_image)

    for item in results:
        print(item)



