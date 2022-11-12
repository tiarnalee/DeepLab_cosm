from torch.utils.data import DataLoader
from data_generators.deepfashion import DeepFashionSegmentation
from data_generators.NUMPY_loader1 import np_loader

def initialize_data_loader(config):


    if config['dataset']['dataset_name'] == 'NUMPY_dataloader1':
        train_set = np_loader(config, split='train')
        val_set = np_loader(config, split='val')
        test_set = np_loader(config, split='test')

    else:
        raise Exception('dataset not implemented yet!')

    num_classes = train_set.num_classes
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['workers'], pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['workers'], pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['workers'], pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes

