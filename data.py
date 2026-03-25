"""Data loader"""
import torch
import torchvision
from torchvision import datasets, transforms
from ops import ChunkSampler
from dataloader import get_config, load_dataset, real_dataset_list


def normalize_dataset_name(dataset):
    """Normalize dataset names from CLI/input to canonical internal codes."""
    if dataset is None:
        raise ValueError("dataset must not be None")

    key = str(dataset).strip().lower()
    key = key.replace('_', '-')

    aliases = {
        'cifar-10': 'cifar10',
        'connect-4': 'connect',
        'connect4': 'connect',
        'satimage': 'satimages',
        'sat-images': 'satimages',
    }

    return aliases.get(key, key)


def _split_train_valid(train_loader, valid_ratio, cuda=False, num_workers=0):
    """Create a validation split when an external loader does not provide one.
    
    Properly handles dataset splits while maintaining deterministic seeding.
    """
    dataset = train_loader.dataset
    total_num = len(dataset)
    num_valid = int(round(total_num * valid_ratio))
    num_valid = max(1, min(num_valid, total_num - 1))
    num_train = total_num - num_valid

    generator = torch.Generator().manual_seed(0)
    train_set, valid_set = torch.utils.data.random_split(
        dataset,
        [num_train, num_valid],
        generator=generator,
    )

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    } if cuda else {'num_workers': num_workers}

    train_loader_new = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_loader.batch_size,
        shuffle=True,
        **kwargs
    )
    valid_loader_new = torch.utils.data.DataLoader(
        valid_set,
        batch_size=train_loader.batch_size,
        shuffle=False,
        **kwargs
    )
    return train_loader_new, valid_loader_new


def get_dataloaders(
        dataset='mnist',
        batch_size=128,
        augmentation_on=False,
        cuda=False, num_workers=0,
        valid_ratio=0.1,
):
    """Load dataloaders for a given dataset.
    
    For tabular datasets, StandardScaler normalization is automatically applied
    to all features (fit on training set, applied to train/valid/test).
    
    Args:
        dataset: Dataset name (e.g., 'mnist', 'letter', 'connect', etc.)
        batch_size: Batch size for data loaders
        augmentation_on: Apply data augmentation (only for image datasets)
        cuda: Use GPU-compatible kwargs
        num_workers: Number of workers for data loading
        valid_ratio: Validation split ratio (default 0.1)
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, NUM_TRAIN, NUM_VALID)
    """
    dataset = normalize_dataset_name(dataset)

    kwargs = {
        'num_workers': num_workers, 'pin_memory': True,
    } if cuda else {}

    if dataset in real_dataset_list:
        config = dict(get_config(dataset))
        config['batch_size'] = batch_size

        # Keep using the dataloader.py path for MNIST as requested.
        if dataset == 'mnist' and augmentation_on:
            print('augmentation_on is ignored for dataloader.py MNIST path.')

        train_loader, valid_loader, test_loader = load_dataset(config)

        if valid_loader is None:
            train_loader, valid_loader = _split_train_valid(
                train_loader,
                valid_ratio,
                cuda=cuda,
                num_workers=num_workers,
            )

        NUM_TRAIN = len(train_loader.dataset)
        NUM_VALID = len(valid_loader.dataset)

    elif dataset == 'cifar10':
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )

        cifar10_train = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True,
            transform=transform_train,
        )
        cifar10_valid = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform_test,
        )
        cifar10_test = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True,
            transform=transform_test,
        )

        TOTAL_NUM = 50000
        NUM_VALID = int(round(TOTAL_NUM * 0.02))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        train_loader = torch.utils.data.DataLoader(
            cifar10_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            cifar10_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            cifar10_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)
    else:
        raise NotImplementedError("Specified data set is not available.")

    return train_loader, valid_loader, test_loader, NUM_TRAIN, NUM_VALID


def get_dataset_details(dataset):
    dataset = normalize_dataset_name(dataset)

    if dataset == 'mnist':
        input_nc, input_width, input_height = 1, 28, 28
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    elif dataset == 'cifar10':
        input_nc, input_width, input_height = 3, 32, 32
        classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck',
        )
    elif dataset in real_dataset_list:
        config = get_config(dataset)
        input_nc, input_width, input_height = 1, config['n_attributes'], 1
        if config['classes']:
            classes = tuple(config['classes'])
        else:
            classes = tuple(range(config['n_classes']))
    else:
        raise NotImplementedError("Specified data set is not available.")

    return input_nc, input_width, input_height, classes
