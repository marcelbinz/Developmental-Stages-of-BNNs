import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from scipy.ndimage import convolve


def create_sample(condition):
    img = torch.zeros((2, 24, 24))

    # zero: left is larger, one: right is larger
    label = int(np.random.choice([0, 1]))

    objects_left = condition if label else condition + 1
    objects_right = condition + 1 if label else condition

    def place_objects(num_objects, sizes, probs):
        channel = torch.zeros((24, 24))

        for i in range(num_objects):
            # sample from free_space
            object_placed = False
            size = np.random.choice(sizes, p=probs)
            while not object_placed:
                pos_x = np.random.randint(1, 23-size)
                pos_y = np.random.randint(1, 23-size)

                if not np.any(channel[pos_x-1:pos_x+size+1, pos_y-1:pos_y+size+1]):
                    # update img
                    channel[pos_x:pos_x+size, pos_y:pos_y+size] = 1

                    object_placed = True

        return channel

    min_size = 1
    max_size = 3
    sizes = np.arange(min_size, max_size + 1)

    # right is larger
    if label:
        solution_found = False
        while not solution_found:
            ratio = objects_right / objects_left
            probs_right = np.random.uniform(size=3)
            probs_right = probs_right / np.sum(probs_right)

            avg_area_right = probs_right.dot(sizes ** 2)
            a = np.random.uniform()

            bc = np.linalg.solve(np.array([[4, 9], [1, 1]]) , np.array([avg_area_right * ratio - a, 1 - a]))
            probs_left = np.insert(bc, 0, a)
            if np.all(probs_left > 0):
                solution_found = True
    else:
        solution_found = False
        while not solution_found:
            ratio = objects_left / objects_right
            probs_left = np.random.uniform(size=3)
            probs_left = probs_left / np.sum(probs_left)

            avg_area_left = probs_left.dot(sizes ** 2)
            a = np.random.uniform()

            bc = np.linalg.solve(np.array([[4, 9], [1, 1]]) , np.array([avg_area_left * ratio - a, 1 - a]))
            probs_right = np.insert(bc, 0, a)
            if np.all(probs_right > 0):
                solution_found = True

    img[0, :] = place_objects(objects_left, sizes, probs_left)
    img[1, :] = place_objects(objects_right, sizes, probs_right)

    return img, label, condition - 1, objects_left - 1, objects_right - 1

points_per_condition = 6000

samples_per_type = list(zip(list(range(1, 10)), [points_per_condition] * 9)) + list(zip(list(range(1, 10)), [points_per_condition] * 9))
total_samples = sum(list(x[1] for x in samples_per_type))
print(total_samples)
data = torch.zeros((total_samples, 2, 24, 24))
labels = torch.zeros(total_samples)
conds = torch.zeros(total_samples)
objects_left = torch.zeros(total_samples)
objects_right = torch.zeros(total_samples)
i = 0
for (condition, samples) in samples_per_type:
    for _ in range(samples):
        img, label, cond, object_left, object_right = create_sample(condition)
        data[i, :, :, :] = img
        labels[i] = label
        conds[i] = cond
        objects_left[i] = object_left
        objects_right[i] = object_right
        i += 1

torch.save([data[:total_samples // 2], labels[:total_samples // 2], conds[:total_samples // 2], objects_left[:total_samples // 2], objects_right[:total_samples // 2]], 'train.pt')
torch.save([data[(total_samples // 2):], labels[(total_samples // 2):], conds[(total_samples // 2):], objects_left[(total_samples // 2):], objects_right[(total_samples // 2):]], 'test.pt')

images = 8

fig = plt.figure()
axs = []
gs = gridspec.GridSpec(images, images)

for i in range(images ** 2):
    index = np.random.randint(total_samples)
    img = data[index].reshape(-1, 24).t()
    axs.append(fig.add_subplot(gs[i]))
    axs[-1].set_title(objects_right[index] + 1)
    axs[-1].imshow(img, origin='lower')
    plt.setp(axs[-1].get_xticklabels(), visible=False)
    plt.setp(axs[-1].get_yticklabels(), visible=False)
    axs[-1].tick_params(axis='both', which='both', length=0)

plt.show()
