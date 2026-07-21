import random
from collections import defaultdict
from torch.utils.data import Dataset
from utils.utils import read_image
from utils.utils import get_images_and_labels

class ClassificationDataset(Dataset):
    def __init__(self, images, labels, image_size, db_root, class_map, N: int, mode: str = "default", augments=None, transforms=None, seed: int = 42):
        '''
        mode="default"
            → No oversampling
            → Uses images & labels as provided
        mode="all_to_N"
            → Every class ends up with exactly N samples
            → Classes > N are truncated
            → Classes < N are oversampled (with replacement)

        mode="under_to_N"
            → Only classes with < N samples are oversampled to N
            → Classes with ≥ N samples are left unchanged
        '''
        super().__init__()
        assert mode in {"default", "all_to_N", "under_to_N"}, "Invalid oversampling mode"

        self.image_size = image_size
        self.db_root = db_root
        self.class_map = class_map
        self.augments = augments
        self.transforms = transforms

        random.seed(seed)

        # --------------------------------------------------
        # DEFAULT MODE (no oversampling)
        # --------------------------------------------------
        if mode == "default":
            self.images = images
            self.labels = labels
            return

        if N is None:
            raise ValueError("N must be provided for oversampling modes")

        # --------------------------------------------------
        # OVERSAMPLING MODES
        # --------------------------------------------------
        # Group class indices
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_indices[label].append(idx)

        new_indices = []

        for label, idxs in class_to_indices.items():
            count = len(idxs)

            if mode == "all_to_N":
                if count >= N:
                    sampled = random.sample(idxs, N)
                else:
                    sampled = idxs + random.choices(idxs, k=N - count)

            elif mode == "under_to_N":
                if count >= N:
                    sampled = idxs
                else:
                    sampled = idxs + random.choices(idxs, k=N - count)

            new_indices.extend(sampled)

        random.shuffle(new_indices)

        # Store final samples
        self.images = [images[i] for i in new_indices]
        self.labels = [labels[i] for i in new_indices]


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image = read_image(
            f"{self.db_root}/{self.images[index]}",
            None
        )
        label = self.labels[index]

        if self.augments is not None:
            image = self.augments(image=image)["image"]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label
