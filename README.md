# An audiovisual and contextual approach for categorical and continuous emotion recognition in-the-wild

This repository hosts the official PyTorch source code of the team NTUA-CVSP for the 2nd Affective Behavior Analysis in-the-wild (ABAW2) Competition organized at ICCV 2021. 

Abstract:
> In this work we tackle the task of video-based audio-visual emotion recognition, within the premises of the 2nd Workshop and Competition on Affective Behavior Analysis in-the-wild (ABAW2). Poor illumination conditions, head/body orientation and low image resolution constitute factors that can potentially hinder performance in case of methodologies that solely rely on the extraction and analysis of facial features. In order to alleviate this problem, we leverage both bodily and contextual features, as part of a broader emotion recognition framework. We choose to use a standard CNN-RNN cascade as the backbone of our proposed model for sequence-to-sequence (seq2seq) learning. Apart from learning through the RGB input modality, we construct an aural stream which operates on sequences of extracted mel-spectrograms. Our extensive experiments on the challenging and newly assembled Aff-Wild2 dataset verify the validity of our intuitive multi-stream and multi-modal approach towards emotion recognition in-the-wild. Emphasis is being laid on the the beneficial influence of the human body and scene context, as aspects of the emotion recognition process that have been left relatively unexplored up to this point. All the code was implemented using PyTorch and is publicly available.

You can read the full paper [here](https://arxiv.org/abs/2107.03465).



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
@InProceedings{Antoniadis_2021_ICCV,
    author    = {Antoniadis, Panagiotis and Pikoulis, Ioannis and Filntisis, Panagiotis P. and Maragos, Petros},
    title     = {An Audiovisual and Contextual Approach for Categorical and Continuous Emotion Recognition In-the-Wild},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {3645-3651}
}
```


## Acknowlegements

- [https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)

## Contact

For questions feel free to open an issue.
