import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

CYLINDER = 0
SCREEN = 1
GROUND = 2


def create_sample(condition):
    img = torch.zeros((3, 24, 24))

    # place ground
    ground_level = np.random.choice([1, 2, 3, 4, 5])
    img[GROUND, :ground_level, :] = 1

    # place cylinder
    cylinder_height = np.random.choice([4, 5, 6, 7, 8])
    cylinder_offset_x = np.random.choice([1, 2, 3, 4, 5, 6])
    img[CYLINDER, ground_level:ground_level + cylinder_height, cylinder_offset_x] = 1

    # place screen
    screen_width = np.random.choice([5, 6, 7, 8, 9, 10])
    screen_height = np.random.choice([3, 4, 5, 6, 7, 8]) + cylinder_height
    screen_offset_x = np.random.randint(cylinder_offset_x + 2, 24 - screen_width - 4)

    # draw outer walls
    img[SCREEN, ground_level:ground_level + screen_height, screen_offset_x] = 1
    img[SCREEN, ground_level:ground_level + screen_height, screen_offset_x+screen_width] = 1

    # create conditions, labels: 0 (not visible), 1 (visible)
    if condition == 'full':
        img[SCREEN, ground_level:ground_level + screen_height, screen_offset_x:screen_offset_x+screen_width+1] = 1
        label = 0
        cond = 0
    elif condition == 'bottom1':
        fill_height = cylinder_height + np.random.choice([1, 2])  # 0
        img[SCREEN, ground_level:ground_level + fill_height, screen_offset_x + 1:screen_offset_x+screen_width] = 1
        label = 0
        cond = 1
    elif condition == 'bottom2':
        fill_height = cylinder_height + np.random.choice([-2, -3]) # -1
        img[SCREEN, ground_level:ground_level + fill_height, screen_offset_x + 1:screen_offset_x+screen_width] = 1
        label = 1
        cond = 1
    elif condition == 'top':
        fill_height = np.random.randint(2, screen_height - 2)
        img[SCREEN, ground_level+fill_height:ground_level + screen_height, screen_offset_x + 1:screen_offset_x+screen_width] = 1
        label = 1
        cond = 2
    elif condition == 'none':
        label = 1
        cond = 3
    return img, label, cond

samples_per_type = [['full', 4000], ['bottom1', 1000], ['bottom2', 1000], ['top', 2000], ['none', 2000], ['full', 4000], ['bottom1', 1000], ['bottom2', 1000], ['top', 2000], ['none', 2000]]

total_samples = sum(list(x[1] for x in samples_per_type))
data = torch.zeros((total_samples, 3, 24, 24))
labels = torch.zeros(total_samples)
conds = torch.zeros(total_samples)
i = 0
for (condition, samples) in samples_per_type:
    for _ in range(samples):
        img, label, cond = create_sample(condition)
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
    img = data[index, 0]
    axs.append(fig.add_subplot(gs[i]))
    axs[-1].set_title(labels[index])
    axs[-1].imshow(img, origin='lower')
    plt.setp(axs[-1].get_xticklabels(), visible=False)
    plt.setp(axs[-1].get_yticklabels(), visible=False)
    axs[-1].tick_params(axis='both', which='both', length=0)

plt.show()
