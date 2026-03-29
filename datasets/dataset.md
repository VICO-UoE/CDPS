# Install datasets
We follow the same datasets installation and the same split files in [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md).
The json files in meta data can be found in [SWAT](https://github.com/tian1327/SWAT/tree/master/data).
To generate your own fewshot samples, make sure each dataset has the following structure:

### ImageNet:
```
imagenet
|-- images
|-- LOC_synset_mapping.txt
|-- val_labels.txt
```

### Caltech101:
- download the split file
```
caltech101
|-- 101_ObjectCategories
|-- split.json
```

### OxfordPets
- download the split file
```
oxford_pets
|-- images
|-- split.json
```

### StanfordCars
- download the split file
```
stanford_cars
|-- cars_test
|-- cars_train
|-- split.json
```

### Flowers102
- download the split file
```
flowers102
|-- jpg
|-- split.json
```

### Food101
- download the split file
```
food101
|-- images
|-- split.json
```

### FGVCAircraft
```
fgvc-aircraft
|-- images
|-- images_variant_test.txt
|-- images_variant_train.txt
|-- variants.txt
```

### SUN397
- download the split file
```
sun397
|-- images
|-- split.json
```

### DTD
- download the split file
```
dtd
|-- images
|-- split.json
```

### EuroSAT
- download the split file
```
eurosat
|-- EuroSAT_RGB
|-- split.json
```

### UCF101
- download the split file
```
ucf101
|-- ucf-101-midframes
|-- split.json
```

### SemiAves
- download the dataset
```
semi-aves
|-- test
|-- trainval_images
|-- solution.csv
```

