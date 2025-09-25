# Steel Defect Classification

## Installation

Clone the repository

```bash
git clone https://github.com/Theorist-Git/steel_defect_clf.git
```

## Requirements
The required Python libraries are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```
You will also need the NEU dataset, you can find pre-processed tar ball in the repo itself


## Usage
You have the option to either execute the notebooks to train new models and potentially fine-tune their hyperparameters, or utilize the pre-trained `.keras` models provided.

```python
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os

import pathlib

data_dir = pathlib.Path("~/")
data_dir = data_dir / "train" / "images"

batch_size = 32
img_height = 200
img_width = 200

model = tf.keras.models.load_model("choose_your_model.keras")
crazing_path = pathlib.Path("~/crazing_19.jpg")

img = tf.keras.utils.load_img(
    crazing_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
```

## Author
* Mayank vats : [Theorist-git](https://github.com/Theorist-Git)
  * Email: dev-theorist.e5xna@simplelogin.com


## License

[MIT](https://choosealicense.com/licenses/mit/)
