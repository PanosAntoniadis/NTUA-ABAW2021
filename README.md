# An audiovisual and contextual approach for categorical and continuous emotion recognition in-the-wild

This repository contains the source code of the team NTUA-CVSP for the 2nd Affective Behavior Analysis in-the-wild (ABAW2) Competition organized at ICCV 2021. Please read the accompanied paper for more details ([arxiv](https://arxiv.org/abs/2107.03465)).



## Requiremnets

- PyTorch
- NumPy
- OpenCV
- sklearn
- pandas
- matplotlib

## Preparation

- Download the [Aff-Wild2 database](https://ibug.doc.ic.ac.uk/resources/aff-wild2/).
- Run ```scripts/pickle_data.py``` to generate training and validation set.


## Training


To train the proposed model for VA estimation (track 1) run:

> python train_affwild2.py -c config.json --track 1 --pretrained_affectnet single --optimizer SGD --lr 0.001 --context --body --face


To train the proposed model for basic expression classifcation (track 2) run:

> python train_affwild2.py -c config.json --track 2 --pretrained_affectnet single --optimizer SGD --lr 0.001 --context --body --face


## Pretrained Models






## Citation

If you use this code for your research, consider citing our paper.

```
@article{NTUA_ABAW2,
    title={An audiovisual and contextual approach for categorical and continuous emotion recognition in-the-wild},
    author={Panagiotis Antoniadis and Ioannis Pikoulis and Panagiotis P. Filntisis and Petros Maragos},
    journal={arXiv preprint arXiv:2107.03465}
    year={2021}
}
```


## Acknowlegements

- [https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)

## Contact

For questions feel free to open an issue.
