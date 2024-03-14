### Extra information on datasets.json

In UKBB: Different train/val/test csvs for different purposes (different binary labels and sometimes different images):
- ckd: ckd disease predicition (subset of images with labels)
- sex: sex prediction (high quality images). Also csvs with smaller proportion of images ranging form 1 to 100%
- BMI: bmi prediction (high quality images)
- BP: BP prediction:
    - all: low and high quality images
    - standard: high quality
    - filt: high quality with some images w/o ethnicity label removed

MIMIC CXR:
Need to remember that 'No Finding' == 1 and else == 0. But in analysis I switch it so that disease appears as "positive" class

### Sensitive attributes
Added 'Ethnicity' as a sensitive attribute with 4 classes (int from 0 to 3)