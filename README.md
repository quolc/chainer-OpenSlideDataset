# OpenSlideDataset for Chainer


## What's this?

This dataset enables training DNN (e.g., convnets) with very large images such as pathological slides with polygonal region annotation.
It generates many randomly cropped image patches from any images that [OpenSlide](http://openslide.org/api/python/) can handle.


## Dependency

- Chainer
- OpenCV3
- tripy
- OpenSlide (can be installed by `pip install openslide-python`)


## Usage example

Usage of `LabeledOpenSlideDataset` is very similar to that of official `LabeledImageDataset`, except for that you should prepare custom input file that contains the names of slide images, polygonal regions in them and their labels.
Please see the demo files and sample scripts for the detailed usage.

You can try training of convnet for trivial discrimination task (sand vs. sky in desert images) as follows.

```bash
# prepare input slide (tiled tif) from jpg
$ cd test_slides
$ for i in 1 2 3 4; do ./convert_to_tiled_tif.sh desert${i}.jpg desert${i}.tif; done

# run example script
# if you do not have GPU, remove --gpu option.
$ cd ..
$ python train_example.py train.txt val.txt --root ./test_slides --test --gpu 0

```
