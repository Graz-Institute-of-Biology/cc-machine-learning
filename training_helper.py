import albumentations as albu


def get_training_augmentation(min_height=1024, min_width=1024):
    train_transform = [

        albu.RandomCrop(height=min_height, width=min_width, always_apply=True),
        albu.PadIfNeeded(min_height=min_height, min_width=min_width, always_apply=True, border_mode=0),
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),


        # albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.2,
        ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(min_height=1024, min_width=1024):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.RandomCrop(height=min_height, width=min_width, always_apply=True),
        albu.PadIfNeeded(min_height=min_height, min_width=min_width)
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