from imports import random, COCO, COCOeval, Precision, Recall, AUC
import augmentation as aug
import helpers as tools

data_folder = 'data'
annFile='{}/annotations/instances_{}2017.json'.format(data_folder, 'train')

coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
desired_classes = random.sample(nms, 5)  # nb of class we take
# desired_classes = ['laptop', 'tv', 'cell phone']

print("### Loading data###")
train_img, train_size, coco_train, train_img_ids = tools.filterDataset(data_folder, desired_classes, 'train')
val_img, val_size, coco_val, _ = tools.filterDataset(data_folder, desired_classes, 'val')

nb_epochs = 10
batch_size = 5
input_image_size = (224, 224)  # arbitrary, downsize every img
mask_type = 'normal'  # normal or binary

print("### Creating data generators###")
train_gen = tools.dataGeneratorCoco(train_img, desired_classes, coco_train, data_folder,
                                    mode='train', batch_size=batch_size, mask_type=mask_type)

val_gen = tools.dataGeneratorCoco(val_img, desired_classes, coco_val, data_folder,
                                  mode='val', batch_size=batch_size, mask_type=mask_type)

epoch_steps = train_size // batch_size
val_steps = val_size // batch_size

augGeneratorArgs = dict(featurewise_center=False,
                        samplewise_center=False,
                        rotation_range=5,
                        width_shift_range=0.01,
                        height_shift_range=0.01,
                        brightness_range=(0.8, 1.2),
                        shear_range=0.01,
                        zoom_range=[1, 1.25],
                        horizontal_flip=True,
                        vertical_flip=False,
                        fill_mode='reflect',
                        data_format='channels_last')

print("### Data augmentation ###")
train_aug = aug.augmentationsGenerator(train_gen, augGeneratorArgs)
val_aug = aug.augmentationsGenerator(val_gen, augGeneratorArgs)

aug.visualizeGenerator(train_aug)

print("### Model compiling ###")
m = <your model>
opt = <your optimizer>
lossFn = <your loss function>

# Compile your model first
m.compile(loss = lossFn, optimizer = opt, metrics=['accuracy', 'categorical_accuracy',
                                                   Precision(), Recall(), AUC()])

print("### Model training ###")
# Start the training process
history = m.fit(x = train_aug,
                validation_data = val_aug,
                steps_per_epoch = epoch_steps,
                validation_steps = val_steps,
                epochs = nb_epochs,
                verbose = True)

# running evaluation
print("\n### Evaluation process ###")
cocoEval = COCOeval(cocoGt=coco_train,cocoDt=None)  # iouType is "segm" by default
cocoEval.params.imgIds = train_img_ids

print("### Evaluates detections on every image and every category  ###\n")
cocoEval.evaluate()
print("### Accumulates the per-image, per-category evaluation ###\n")
cocoEval.accumulate()
print("### Display summary metrics of results ###\n")
cocoEval.summarize()

print("\n### End of program ###\n")
