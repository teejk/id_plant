import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import cv2 as cv
from sklearn.model_selection import train_test_split

from pathlib import Path
import multiprocessing
import time
from abc import ABC, abstractmethod
from datetime import datetime

from utils.training_helper_funcs import time_elapsed_remaining

n_cores = multiprocessing.cpu_count() - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlantIDDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        df_id_images: pd.DataFrame = None,
        return_target: bool = True,
        transform=None,
    ):
        """Dataset that is used to train a plant identifier model.

        Parameters
        ----------
        dataset_path : Path
            Location of dataset. The dataset should contain a folder with cropped images
            at ("/cropped_images") and a feather file (named "df_id_images.feather) with
            each row composed of a pair of images and a target column encoding if the
            images are of the same plant or not.

            The columns of the feather file should be:
            - pot_name_1: name of the potted plant in the first image
            - pot_name_2: name of the potted plant in the second image
            - img_name_1: name of image containing pot_name_1. The image should be in the
            cropped images folder
            - img_name_2: name of image containing pot_name_2. The image should be in the
            cropped images folder
            - SAME: target column encoding if the images are of the same plant or not.
            (Not necessary if return_target is False)

        df_id_images : pd.DataFrame, optional
            DataFrame containing the image pairs and target column, by default None.
            If None, the dataset will look for the feather file in the dataset_path.
        return_target : bool, optional
            Whether to return the target column, by default True
        transform : torchvision.transforms, optional
            Transform to apply to the images, by default None
        """
        self.dataset_path = Path(dataset_path)
        self.cropped_images_path = dataset_path / "cropped_images"
        self.return_target = return_target

        self.transforms = [torchvision.transforms.ToTensor()]
        if transform is not None:
            self.transforms.append(transform)
        self.df_id_images = df_id_images

        if self.df_id_images is None:
            self.df_id_images_path = dataset_path / "df_id_images.feather"
            if self.df_id_images_path.exists():
                self.df_id_images = pd.read_feather(self.df_id_images_path)

    def __len__(self):
        return len(self.df_id_images)

    def __getitem__(self, i):
        def return_item(i):
            img1 = cv.imread(
                self.cropped_images_path / self.df_id_images["img_name_1"].iloc[i]
            )
            img2 = cv.imread(
                self.cropped_images_path / self.df_id_images["img_name_2"].iloc[i]
            )

            img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

            transforms = torchvision.transforms.Compose(self.transforms)

            img1 = transforms(img1)
            img2 = transforms(img2)

            if self.return_target:
                return (
                    img1,
                    img2,
                    torch.tensor(
                        self.df_id_images["SAME"].iloc[i], dtype=torch.float32
                    ),
                )
            else:
                return img1, img2

        if isinstance(i, slice):
            start = i.start
            stop = i.stop
            if start is None:
                start = 0
            if i.stop is None:
                stop = len(self._groups)
            items = [return_item(j) for j in range(start, stop)]
            return tuple(map(torch.stack, zip(*items)))

        elif isinstance(i, (list, np.ndarray)):
            items = [return_item(j) for j in i]
            return tuple(map(torch.stack, zip(*items)))

        else:
            return return_item(i)

    def add_transform(self, transform):
        self.transforms.append(transform)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1 - target) * torch.pow(euclidean_distance, 2)
        neg = (target) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive


class PlantIdentifier(ABC):
    @abstractmethod
    def predict(self, image1, image2) -> float:
        pass


class SiamesePlantIdentifier(PlantIdentifier, nn.Module):
    """
    Siamese network for image similarity estimation.
    The network is composed of two identical networks, one for each input.
    The output of each network is concatenated and passed to a linear layer.
    The output of the linear layer passed through a sigmoid function.
    `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese
    network.
    This implementation varies from FaceNet as we use the `ResNet-18` model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    as our feature extractor.
    """

    def __init__(self):
        super(SiamesePlantIdentifier, self).__init__()
        resnet_model = torchvision.models.resnet18
        pretrained_weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.preprocess = pretrained_weights.transforms()

        # get resnet model
        self.resnet = resnet_model(weights=pretrained_weights)

        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet
        # (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
        )

        # initialize the weights of the linear layer
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2) -> float:
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def train_model(
        self,
        train_dataset: PlantIDDataset,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        batch_size: int = 720,
        thresholds: float = [1.0],
        margin: float = 2.0,
        lr: float = 0.001,
        n_epochs: int = 1000,
        continue_training: bool = True,
        val_dataset: PlantIDDataset = None,
        dont_save: bool = False,
        model_save_folder: Path = None,
    ):
        if model_save_folder is None:
            model_save_folder = Path(__file__).parent / "models" / "siamese_net"

        if continue_training:
            existing_weights_path = model_save_folder / "latest.pt"
            if existing_weights_path.exists():
                self.load_state_dict(
                    torch.load(existing_weights_path, weights_only=True)
                )
                print(f"Model loaded from {existing_weights_path}")

        if val_dataset is None:
            df_id_images = train_dataset.df_id_images
            dataset_path = train_dataset.dataset_path
            train_idx, val_idx = train_test_split(
                df_id_images.index,
                test_size=0.2,
                random_state=42,
                stratify=df_id_images["SAME"],
            )
            train_dataset = PlantIDDataset(dataset_path, df_id_images.iloc[train_idx])
            val_dataset = PlantIDDataset(dataset_path, df_id_images.iloc[val_idx])

        train_dataset.return_target = True
        train_dataset.add_transform(self.preprocess)
        val_dataset.return_target = True
        val_dataset.add_transform(self.preprocess)

        optimizer = optimizer(self.parameters(), lr=lr)
        criterion = ContrastiveLoss(margin=margin)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.train()
        self.to(device)

        start = time.time()
        print_loss_total = 0

        for epoch in range(n_epochs):
            total_loss = 0
            for batch_idx, (imgs1, imgs2, targets) in enumerate(train_loader):
                imgs1, imgs2, targets = (
                    imgs1.to(device),
                    imgs2.to(device),
                    targets.to(device),
                )
                optimizer.zero_grad()
                outputs1, outputs2 = self(imgs1, imgs2)
                outputs1, outputs2 = outputs1.squeeze(), outputs2.squeeze()
                loss = criterion(outputs1, outputs2, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.sum().item()

            total_loss /= len(train_loader)
            print_loss_total += total_loss

            if epoch % 100 == 0:
                print_loss_avg = print_loss_total / 100
                print(
                    f"{time_elapsed_remaining(start, (epoch + 1) / n_epochs)} "
                    f"| Epoch: {epoch + 1}, {epoch / n_epochs * 100:.0f}% | "
                    f"Avg Loss: {print_loss_avg:.4f}"
                )
                self.evaluate_model(val_dataset, thresholds)
                print_loss_total = 0

                if not dont_save and epoch > 0:
                    model_save_folder.mkdir(parents=True, exist_ok=True)
                    torch.save(self.state_dict(), model_save_folder / "latest.pt")

                    # Save a copy of the model in the archive
                    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                    archive_folder = model_save_folder / "archive"
                    archive_folder.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        self.state_dict(),
                        archive_folder / f"{now}_e{epoch}.pt",
                    )

        print(
            f"Training complete. Model has been saved to "
            f"{model_save_folder / 'latest.pt'}. Past models have also been "
            f"saved to {model_save_folder / 'archive'}."
        )

    def evaluate_pair(
        self,
        outputs1: torch.Tensor,
        outputs2: torch.Tensor,
        targets: torch.Tensor,
        thresholds: list[float],
    ):
        euclidean_distance = F.pairwise_distance(outputs1, outputs2)
        return_dict = {}
        for threshold in thresholds:
            preds = euclidean_distance > threshold

            total_similar = (targets == 0).sum()
            total_dissimilar = (targets == 1).sum()
            similar_correct = 0
            dissimilar_correct = 0

            for i in range(len(preds)):
                if targets[i] == preds[i] == 0:
                    similar_correct += 1
                if targets[i] == preds[i] == 1:
                    dissimilar_correct += 1

            return_dict[threshold] = (
                similar_correct,
                dissimilar_correct,
                total_similar,
                total_dissimilar,
            )
        return return_dict

    def evaluate_model(
        self,
        test_dataset: PlantIDDataset,
        thresholds: list[float],
        margin: float = 1.0,
        batch_size: int = 1000,
    ):
        test_dataset.return_target = True
        test_dataset.add_transform(self.preprocess)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.to(device)
        self.eval()

        total_loss = 0

        accuracy_dict = {
            threshold: {
                "total_similar": 0,
                "total_dissimilar": 0,
                "similar_correct": 0,
                "dissimilar_correct": 0,
            }
            for threshold in thresholds
        }

        criterion = ContrastiveLoss(margin=margin)

        with torch.no_grad():
            for imgs1, imgs2, targets in test_loader:
                imgs1, imgs2, targets = (
                    imgs1.to(device),
                    imgs2.to(device),
                    targets.to(device),
                )
                outputs1, outputs2 = self(imgs1, imgs2)
                outputs1, outputs2 = outputs1.squeeze(), outputs2.squeeze()
                loss = criterion(outputs1, outputs2, targets)
                total_loss += loss.sum().item()
                return_dict = self.evaluate_pair(
                    outputs1, outputs2, targets, thresholds
                )
                for threshold in thresholds:
                    (
                        similar_correct_b,
                        dissimilar_correct_b,
                        total_similar_b,
                        total_dissimilar_b,
                    ) = return_dict[threshold]

                    accuracy_dict[threshold]["total_similar"] += total_similar_b
                    accuracy_dict[threshold]["total_dissimilar"] += total_dissimilar_b
                    accuracy_dict[threshold]["similar_correct"] += similar_correct_b
                    accuracy_dict[threshold][
                        "dissimilar_correct"
                    ] += dissimilar_correct_b

        print(f"Average Validation Loss: {total_loss / len(test_loader)}")
        for threshold, data in accuracy_dict.items():
            print(f"Threshold: {threshold}")

            similar_acc = data["similar_correct"] / data["total_similar"]
            dissimilar_acc = data["dissimilar_correct"] / data["total_dissimilar"]

            print(
                f"Similar Accuracy: {similar_acc:.4f}, Dissimilar Accuracy: "
                f"{dissimilar_acc:.4f}"
            )
        self.train()

    def predict(self, infer_dataset: PlantIDDataset, threshold) -> list:
        infer_dataset.return_target = False
        infer_dataset.add_transform(self.preprocess)

        train_kwargs = {"batch_size": 1000, "shuffle": False}
        if device == "cuda":
            train_kwargs["num_workers"] = 1
            train_kwargs["pin_memory"] = True
        infer_loader = DataLoader(infer_dataset, **train_kwargs)

        self.to(device)
        self.eval()

        preds = []

        with torch.no_grad():
            for imgs1, imgs2 in infer_loader:
                imgs1, imgs2 = imgs1.to(device), imgs2.to(device)
                outputs1, outputs2 = self(imgs1, imgs2)
                outputs1, outputs2 = outputs1.squeeze(), outputs2.squeeze()
                euclidean_distance = F.pairwise_distance(outputs1, outputs2)
                preds.extend((euclidean_distance > threshold).cpu().tolist())

        return preds
