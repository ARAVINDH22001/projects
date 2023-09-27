from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Concatenate every two images horizontally
new_imgs = []
new_nums = []
count = 0
for i in range(0, len(x_train) - 1, 2):

    count = count + 1
    img1, img2 = x_train[i], x_train[i+1]
    img3, img4 = x_train[i+1], x_train[i]
    concat_img1 = np.concatenate([img1, img2], axis=1)
    concat_img2 = np.concatenate([img3, img4], axis=1)
    resized_img1 = Image.fromarray(concat_img1).resize((32, 32))
    resized_img2 = Image.fromarray(concat_img2).resize((32, 32))
    new_imgs.append(np.asarray(resized_img1))
    new_imgs.append(np.asarray(resized_img2))

    count = count + 1
    num1, num2 = y_train[i], y_train[i+1]
    num3, num4 = y_train[i+1], y_train[i]
    N1 = num1 * 10 + num2
    N2 = num3 * 10 + num4
    new_nums.append(N1)
    new_nums.append(N2)

# Plot the concatenated and resized images
# fig, axs = plt.subplots(5, 2, figsize=(10, 20))
# axs = axs.flatten()
# print(len(new_imgs))
# for i in range(len(new_imgs)):
#     axs[i].imshow(new_imgs[i], cmap='gray')
# plt.show()
print(count)

plt.imshow(new_imgs[1345], cmap='gray')
plt.show()
print(new_nums[1345])

print(count)

len(new_nums)

len(new_imgs)

plt.imshow(new_imgs[59999], cmap='gray')
plt.show()
print(new_nums[59999])

def CountFrequency(new_nums):
    freq = {}
    for item in new_nums:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
 
    sorted_freq = sorted(freq.items())
 
    for key, value in sorted_freq:
        print("%d: %d" % (key, value))

CountFrequency(new_nums)

# Unique data
print(np.unique(new_nums))

# Concatenation of TEST dataset
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Concatenate every two images horizontally
new_test_imgs = []
new_test_nums = []
count = 0
for i in range(0, len(x_test) - 1, 2):

    count = count + 1
    img1, img2 = x_test[i], x_test[i+1]
    img3, img4 = x_test[i+1], x_test[i]
    concat_test_img1 = np.concatenate([img1, img2], axis=1)
    concat_test_img2 = np.concatenate([img3, img4], axis=1)
    resized_test_img1 = Image.fromarray(concat_test_img1).resize((32, 32))
    resized_test_img2 = Image.fromarray(concat_test_img2).resize((32, 32))
    new_test_imgs.append(np.asarray(resized_test_img1))
    new_test_imgs.append(np.asarray(resized_test_img2))

    count = count + 1
    num1, num2 = y_test[i], y_test[i+1]
    num3, num4 = y_test[i+1], y_test[i]
    N1 = num1 * 10 + num2
    N2 = num3 * 10 + num4
    new_test_nums.append(N1)
    new_test_nums.append(N2)

# Plot the concatenated and resized images
# fig, axs = plt.subplots(5, 2, figsize=(10, 20))
# axs = axs.flatten()
# print(len(new_imgs))
# for i in range(len(new_imgs)):
#     axs[i].imshow(new_imgs[i], cmap='gray')
# plt.show()
print(count)

len(new_test_imgs)

len(new_test_nums)

plt.imshow(new_test_imgs[10], cmap='gray')
plt.show()
print(new_test_nums[10])

plt.imshow(new_test_imgs[9999], cmap='gray')
plt.show()
print(new_test_nums[9999])

print(new_test_imgs[1])

type(new_imgs)

new_imgs = np.array(new_imgs)
new_test_imgs = np.array(new_test_imgs)

new_imgs = new_imgs / 255
new_test_imgs = new_test_imgs / 255

print(new_imgs[1])

print(new_imgs.shape)

# Displaying the image 
for i in range(10):
  plt.imshow(new_imgs[i])
  plt.show()
  print('------')
  # Print the label
  print(new_nums[i])
