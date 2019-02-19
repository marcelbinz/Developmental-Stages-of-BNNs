import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

import torch

PLATFORM = 0
BOX = 1
GROUND = 2

def create_sample(condition, stable):
    img = torch.zeros((3, 24, 24))

    # place ground
    ground_level = np.random.choice([1, 2, 3, 4, 5])
    img[GROUND, :ground_level, :] = 1

    # place platform
    base_height = np.random.choice([7, 8, 9])
    base_width = np.random.choice([7, 8, 9])
    base_x = np.random.choice(np.arange(5, 19-base_width))
    img[PLATFORM, ground_level:ground_level + base_height, base_x:base_x+base_width] = 1

    # place box
    if condition == 'full':
        cond = 0
        box_y_offset = ground_level + base_height
        if stable:
            # randomize size and offset
            box_size = np.random.choice([3, 4, 5])
            box_x_offset = np.random.choice(np.arange(base_x, base_x + base_width - box_size + 1))

            img[BOX, box_y_offset:box_y_offset + box_size, box_x_offset:box_x_offset + box_size] = 1
            label = 0
        else:
            box_size = np.random.choice([3, 4, 5])
            box_x_offset = np.random.choice(np.concatenate((np.arange(0, base_x - box_size), np.arange(base_x + base_width + 1, img.shape[1] - box_size + 1))))

            img[BOX, box_y_offset:box_y_offset + box_size, box_x_offset:box_x_offset + box_size] = 1
            label = 1
    elif condition == 'side':
        cond = 1
        if stable:
            box_size = np.random.choice([3, 4, 5])
            box_x_offset = np.random.choice(np.arange(base_x, base_x + base_width - box_size + 1))
            box_y_offset = ground_level + base_height

            img[BOX, box_y_offset:box_y_offset + box_size, box_x_offset:box_x_offset + box_size] = 1
            label = 0
        else:
            box_size = np.random.choice([3, 4, 5])
            box_y_offset = np.random.randint(ground_level + 1, ground_level + base_height)
            box_x_offsets = []
            if base_x - box_size > 0:
                box_x_offsets.append(base_x-box_size)
            if (base_x + base_width + box_size) < img.shape[1]:
                box_x_offsets.append(base_x+base_width)
            box_x_offset = np.random.choice(box_x_offsets)

            img[BOX, box_y_offset:box_y_offset + box_size, box_x_offset:box_x_offset + box_size] = 1
            label = 1
    elif condition == 'amount':
        cond = 2
        box_y_offset = ground_level + base_height
        if stable:
            box_size = np.random.choice([3, 4, 5])
            box_x_offsets = np.concatenate((np.arange(math.ceil(-box_size/2), 0) + base_x, np.arange(base_x + base_width - box_size + 1, math.ceil(-box_size/2) + base_x + base_width)))
            box_x_offset = np.random.choice(box_x_offsets)

            img[BOX, box_y_offset:box_y_offset + box_size, box_x_offset:box_x_offset + box_size] = 1
            label = 0
        else:
            box_size = np.random.choice([3, 4, 5])
            box_x_offsets = np.concatenate((np.arange(max(1, -box_size + base_x + 1), math.ceil(-box_size/2) + base_x), np.arange(math.floor(-box_size/2) + base_x + base_width + 1, base_x + base_width)))
            box_x_offset = np.random.choice(box_x_offsets)

            img[BOX, box_y_offset:box_y_offset + box_size, box_x_offset:box_x_offset + box_size] = 1
            label = 1
    elif condition == 'proportional':
        cond = 3

        il = np.random.choice([3, 4, 5])
        ih = np.random.choice([2])
        jl = il - 2 #np.random.choice([2, 3, 4])
        jh = np.random.choice([2])
        k = np.random.choice(['left', 'right'])

        a1 = il * ih
        x1 = il / 2.0
        y1 = ih / 2.0

        a2 = jl * jh
        x2 = (jh / 2.0) if (k == 'left') else il - (jh / 2.0)
        y2 = (jl / 2.0) + ih

        x = (a1 * x1 + a2 * x2) / (a1 + a2)
        y = (a1 * y1 + a2 * y2) / (a1 + a2)

        if stable:
            list1 = list(range(-base_width + math.ceil(x), math.floor(x) + 1))
            list1 = [i for i in list1 if (i > 0) or (-i > base_width-il)]
            shift = np.random.choice(list1)
            label = 0
        else:
            list1 = list(range(math.floor(x) + 1, min(il, base_x)))
            list2 = list(range(-base_width + 1, -base_width + math.floor(x) + 1))
            shift = np.random.choice(list1 + list2)
            label = 1

        box_x1_offset = base_x - shift
        box_y1_offset = ground_level + base_height

        box_y2_offset = ground_level + base_height + ih

        img[BOX, box_y1_offset:box_y1_offset + ih, box_x1_offset:box_x1_offset + il] = 1

        if k == 'left':
            box_x2_offset = base_x - shift
            img[BOX, box_y2_offset:box_y2_offset + jl, box_x2_offset: box_x2_offset + ih] = 1
        else:
            box_x2_offset = base_x + il - shift
            img[BOX, box_y2_offset:box_y2_offset + jl, box_x2_offset-jh: box_x2_offset] = 1

    return img, label, cond

samples_per_type = [[1250, 'full', True], [1250, 'full', False], [1250, 'side', True], [1250, 'side', False], [1250, 'amount', True], [1250, 'amount', False], [1250, 'proportional', True], [1250, 'proportional', False], \
                    [1250, 'full', True], [1250, 'full', False], [1250, 'side', True], [1250, 'side', False], [1250, 'amount', True], [1250, 'amount', False], [1250, 'proportional', True], [1250, 'proportional', False]]
                    
'''samples_per_type = [[1250, 'proportional', True], [1250, 'proportional', True], \
                    [1250, 'proportional', True], [1250, 'proportional', True]]'''

total_samples = sum(list(x[0] for x in samples_per_type))
data = torch.zeros((total_samples, 3, 24, 24))
labels = torch.zeros(total_samples)
conds = torch.zeros(total_samples)

i = 0
for (samples, condition, stable) in samples_per_type:
    for _ in range(samples):
        img, label, cond = create_sample(condition, stable)
        data[i, :, :, :] = img
        labels[i] = label
        conds[i] = cond
        i += 1

torch.save([data[:total_samples // 2], labels[:total_samples // 2], conds[:total_samples // 2]], 'train.pt')
torch.save([data[(total_samples // 2):], labels[(total_samples // 2):], conds[(total_samples // 2):]], 'test.pt')

images = 8

fig = plt.figure()
axs = []
gs = gridspec.GridSpec(images, images)

for i in range(images ** 2):
    index = np.random.randint(total_samples)
    img = data[index].sum(0)
    axs.append(fig.add_subplot(gs[i]))
    axs[-1].set_title(labels[index])
    axs[-1].imshow(img, origin='lower')
    plt.setp(axs[-1].get_xticklabels(), visible=False)
    plt.setp(axs[-1].get_yticklabels(), visible=False)
    axs[-1].tick_params(axis='both', which='both', length=0)

plt.show()
