import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1 - dice_coef(y_true, y_pred)
    return bce_loss + dice_loss


def predict_mask(image_path, model, image_size=512):
    img = Image.open(image_path)
    original_size = img.size

    img = img.resize((image_size, image_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    mask = (prediction[0] > 0.5).astype(np.uint8)
    mask = np.squeeze(mask)


    mask = Image.fromarray(mask * 255)
    mask = mask.resize(original_size)
    return mask


if __name__ == "__main__":
    model = load_model('Unet.h5', custom_objects={'bce_dice_loss': bce_dice_loss})
    image_path = "test.jpg"

    predicted_mask = predict_mask(image_path, model)


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.show()