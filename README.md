## Deeplab_cosm
## Relevant files
- configs: file with configurations of the model

### Data generators
  - data_generators: loads the datasets, called by the trainer
  - NUMPY_loader1: dataloader for this dataset, calls the data augmentation. The more complicated DA methods are currently commented out
  - (deepfashion: dataloader for the original implementation, ignore)
### Datasets
  - Figaro1k:
    - GT: original images and labels. The labels are in both .npy and .pbm format (.pbm not used)
    - training, validation, test: contains all of the images for the respective training stages. All of the images are in the training folder but only the training images are used in training (not sure why this is necessary but it throws an error if only the training images are in the folder)
    - train_val_test.json: list of the images used in each stage
 
### Preprocessing
- preprocessing: data augmentation I've implemented. Colour based DA is applied to the image only, spatial transformations are applied to the image and mask

### Training 
- models: files for the model inc. backbones and Deeplab. Pretrained weights are used by default, this can be turned off the in the backbone files
- trainers: main trainer which trains model. The loss function here has been changed from Cross Entropy loss to Binary Cross Entropy from the original implementation. Training and validation loss, and accuracy metrics are saved to tensorboard
- main.py: file to run the training. To train the model run, `cd` into the deeplab_cosm file then run :

```
python main.py -c configs/config.yml --train
```

### Inference
- predictors: predictor script used for inference, called by `predict.py`. The loss here is changed from CE to BCE.
- predict.py: my script for inference on the test set. The accuracy is measured using the Dice Similarity Coefficient (1 is the best score)

### Visualise training
- tensorboard: the training metrics can be visualised by running (in another terminal):
```
tensorboard --logdir '/path/to/deeplab_cosm/tensorboard'
```
### Generating new training sets
The data in `datasets > Figaro1k > training/validation/test` has already been split for convenience. To create a new split of training sets, delete the following folders:
- `datasets > Figaro1k > training`
- `datasets > Figaro1k > validation`
- `datasets > Figaro1k > test`

Then run `save_images.py`
