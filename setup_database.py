import sqlite3
import numpy as np
import io


# Based on: https://www.pythonforthelab.com/blog/storing-data-with-sqlite/
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
database_file = 'video_database.db'
conn = sqlite3.connect(database_file, detect_types=sqlite3.PARSE_DECLTYPES)

c = conn.cursor()
c.execute('DROP TABLE IF EXISTS videos')
c.execute('DROP TABLE IF EXISTS frames')

c.execute('''CREATE TABLE videos
        (id INTEGER PRIMARY KEY,
         embedding array)''')
c.execute('''CREATE TABLE frames
        (video_id INTEGER,
         pose array,
         FOREIGN KEY(video_id) REFERENCES videos(id))''')

c.execute('INSERT INTO videos (embedding) values (?)', (np.random.rand(128),))
c.execute('INSERT INTO videos (embedding) values (?)', (np.random.rand(128),))
c.execute('INSERT INTO frames (video_id) values (?)', (1,))
c.execute('INSERT INTO frames (video_id) values (?)', (1,))
c.execute('SELECT video_id FROM frames')
data = c.fetchall()
print(data)
