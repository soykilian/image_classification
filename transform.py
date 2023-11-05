import random
available_transforms = [
    RandomVerticalFlip(1),
    RandomHorizontalFlip(1),
    lambda x: rotate(x, int(random.choice([90,180, 270])))
]
selected_transform = random.choice(available_transforms)