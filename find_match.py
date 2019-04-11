import sqlite3
import numpy as np
from setup_database import adapt_array, convert_array

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
database_file = 'video_database.db'
conn = sqlite3.connect(database_file, detect_types=sqlite3.PARSE_DECLTYPES)

c = conn.cursor()
c.execute('SELECT embedding FROM videos')
embeddings = c.fetchall()
embeddings = np.array(embeddings)
embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])
input_embedding = np.random.rand(1, 128)
input_repeated = np.repeat(input_embedding, embeddings.shape[0], 0)
difference = embeddings - input_repeated
norms = np.linalg.norm(difference, axis=1)
