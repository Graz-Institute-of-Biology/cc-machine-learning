import albumentations as albu
import cv2


def get_training_augmentation(min_height=1024, min_width=1024,
                              photometric=False, scale_limit=0.1):
    """Training augmentation pipeline.

    Defaults reproduce the long-standing hardcoded pipeline used for both the
    ATTO and Großglockner projects (geometry-only, scale_limit=0.1). The
    resolved parameters are recorded in each run's code_snapshot/run_config.json
    so past and future runs stay attributable.

    Args:
        photometric: add mild brightness/contrast + gentle hue/sat jitter.
            Kept mild on purpose — color is a discriminating feature between
            the dark cyano classes; this only hedges against lighting variation.
        scale_limit: ShiftScaleRotate scale range (0.1 = legacy default).
    """
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),

        # border_mode=REFLECT_101: mirrors image/mask content at edges instead of filling
        # with black (0), which would create spurious background labels at patch borders
        albu.ShiftScaleRotate(scale_limit=scale_limit, rotate_limit=45, shift_limit=0.1, p=1,
                              border_mode=cv2.BORDER_REFLECT_101),

        # Reduced from 0.5: nadir drone images have minimal real perspective distortion
        albu.Perspective(p=0.3),

        # Blur and MotionBlur removed: fine structure is a key discriminating feature
        # between organism classes — destroying it hurts more than it helps
        albu.Sharpen(p=0.2),
    ]

    if photometric:
        train_transform += [
            albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            albu.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        ]

    return albu.Compose(train_transform)


def get_validation_augmentation(min_height=1024, min_width=1024):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.RandomCrop(height=min_height, width=min_width, always_apply=True),
        # albu.PadIfNeeded(min_height=min_height, min_width=min_width)
    ]
    return albu.Compose(test_transform)


def img_to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def mask_to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('long')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=img_to_tensor, mask=mask_to_tensor),
    ]
    return albu.Compose(_transform)