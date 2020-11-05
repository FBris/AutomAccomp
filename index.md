# Real-time Music Accompaniment System based on Tempo Models

This page contains datasets for article "Real-time Music Accompaniment System based on Tempo Model" submitted to ____________.

Authors:    , F.-J. Bris-Peñalver,  

![Foto escuela](/Other/PORTADA_8230_33.jpg)

This work presents a methodology based on spectral factorization and online dynamic time warping (DTW). We propose a novel probabilistic model to improve the accuracy of the algorithm through the characterization of the performance tempo. Unlike conventional DTW where the costs are fixed, in this work we define the costs in terms of an occupancy function which model the probability of remaining in a certain state of the composition using information from the immediate past. This contribution drives into an anticipatory behavior of the system and it produce a more accurate detection of performed notes by soloist. Finally, it entails a better adaptation to tempo changes of the musician.

### Data sets for optimization

Three pieces from the classical music database MusicNet are selected. MusicNet consists of classical music recordings with their corresponding ground-truth data and the MIDI repre- sentation of the scores, resulting in a wide variety of music performances under various studio and microphone conditions.

### Datasets for evaluation

Evaluation files are separated in function of tempo deviation, from 0% to 5%, from 5% to 15% and above 15%. Then, they are subdivided in soloist and multi instrument compositions. They are obtained from MusicNet dataset and transformed to Ascograph compatible score files.

- *.asco.txt -> Ascograph score file
- *.mid -> Score file
- *.txt -> Antescofo output
- *.mid -> Ground truth file
- *.wav -> Real audio performance

### Datasets for subjective evaluation of admissible misalingment 

- *The Entertainer (S.Joplin)*
- *Trio for piano, violin & cello in E major K542, second movement (W. A. Mozart)*

Both scores are artificially modified, misaligning the notes of one of the instruments with four different values of delay: 0 to 50 ms, 50 to 100 ms, 100 to 150 ms and 150 to 200 ms, produced throw continuous uniform distribution. 

###  Sofware release provided

The code of this experimental study is provided as reproducible research for testing in this project website including demo files, publicly available. 
