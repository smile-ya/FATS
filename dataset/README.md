## Generating Adversarial Examples

We used the framework by [Ma et al.](https://github.com/xingjunm/lid_adversarial_subspace_detection) to generate various adversarial examples (FGSM, BIM-A, BIM-B, JSMA, and C&W). Please refer to [craft_adv_samples.py](https://github.com/xingjunm/lid_adversarial_subspace_detection/blob/master/craft_adv_examples.py) in the above repository of Ma et al., and put them in the mutation/adv

## Simulated test sets
Run create_dataset.py to generate the simulated test sets.
The part of the simulated test sets of mnist and cifar10 are stored in mnist/adv and cifar10/adv.

## Pre-trained models
Run train_model.py to generate the trained model using mnist, cifar10 and svhn.
