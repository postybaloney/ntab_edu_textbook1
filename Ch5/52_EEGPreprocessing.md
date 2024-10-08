---
title: EEG Preprocessing
layout: page
under_construction: false
parent: Chapter 5
nav_order: 1
released: true
---

# Preprocessing
Pre-processing involves the removal of noise and other outliers in the data set. There are several reasons why preprocessing is necessary for EEG data. First of all, the signals that are picked up from the scalp are not necessarily an accurate representation of the signals originating from the brain, as the spatial information gets lost. Secondly, EEG data tends to contain a lot of noise which can obscure weaker EEG signals. Artifacts such as blinking or muscle movement can contaminate the data and distort the picture. Finally, we want to separate the relevant neural signals from random neural activity that occurs during EEG recordings. Before you start analyzing the data, you should ask yourself what exactly are you looking for? You only need to keep the features that are relevant to the specific experiment that you are doing to decrease the computational complexity and time it takes to compute. Most data that is taken is in the form of Functional Imaging File (.fif) format. It is also common to use .mat or .csv files as imports for data.

## Data Cleaning
Once you import the data, the key thing is to then plot it and ensure that any bad channels are removed and replace them with splines. Then filter the frequencies into the bands which are relevant to the project. To decrease even further the computational complexity, it is important to downsample, or decrease the number of samples used, the data. This is done by selecting an nth sample (every 2nd, 3rd, 5th, 10th, . . . etc.). The first thing important to consider when it comes to sampling is what is known as the Nyquist–Shannon sampling theorem. Despite its fancy name, it’s really just a rule relating the information you can get out of a sampled signal. Put simply: if you are sampling at a rate of R Hz, then any signal of frequency above half of that (i.e. R/2 Hz) will be mistaken for a lower frequency. This process is also known as ‘Aliasing’, as the higher frequency is aliased, or renamed, to the lower one.

## Noise
In EEG data, the voltage for each electrode is recorded relative to other electrodes. The ‘reference’, which can be one or a combination of electrodes, is what the voltage will be relative to. This means that neural activity at the reference electrode will also be reflected in all the other electrodes, which could contaminate your signal.

## Artifacts
Artifacts are signals that are picked up by the EEG system but do not actually originate from the brain. There are many different sources of artifacts for EEG data, which will manifest themselves differently. EEG artifacts can be roughly classified as biological or environmental. Environmental artifacts originate from outside-world interference - for example, power lines, electrodes losing contact or other people’s movement during the experiment. The easiest way to minimize the effect of those artifacts is by adjusting the environment (e.g shielding the room, properly securing the electrodes). Biological artifacts originate from sources in the body. Some of the most common biological artifacts are blinks, eye movements, head movements, heart beats and muscular noise. It is possible to detect those artifacts if you have access to other biometric data, for example, accelerometer, electrooculogram (EOG) or eye tracking data for eye movement artifacts, accelerometer data for head movement artifacts and electrocardiogram (ECG) data for heartbeat artifacts.