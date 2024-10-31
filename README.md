# Doorbell Detector

This repository is my playground on creating a sound event detection model for my doorbell.

# Use Case

During summer I reguarly don't hear the doorbell while I am in the garden, because I am too far away from the door or because the door to the garden is closed to keep the hot air outside of our flat.
Especially, when expecting a delivery this a quite annoying.
Additionally, I am hearing music during work from home and I just realized, that my new headphones do also surpress the sound of the doorbell.
So I would like to get a notification to my smartphone.

Existing solutions are quite invasive by modifing the electrics or using a new smart doorbell.
Since I don't own the flat I'm living in, I cannot do such modifications easy.
So I decided to try to detect the sound of the door bell by microphone.

# Hardware Setup

I got a [Rapsberry PI Zero W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/) with a [ReSpeaker 2 Mics PI Hat](https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT/).

# Plan

I will try to leverage my ML skills to solve this problem and to enhance them along the way.
Especially, data management and reproducibility are points I would like to dive deeper.

## Data gathering script
Everything starts with sufficent data. I already created a simple bash script for recording data using `arecord`.
This script places a `.wav` file on a cifs network storage as soon as it finished the recording.
With [Termux](https://f-droid.org/de/packages/com.termux/) and [Termux:Widget](https://f-droid.org/de/packages/com.termux.widget/) I created a simple way to start short recording from my smartphone, by invoking the recording script on the Raspberry PI Zero W via SSH.
This is handy for starting, but does not scale.
I want to have some (semi-)automatic data gathering.
This is going to be at the first run some rule or DSP based algorithm, which triggers the recording of potential interesting data.

## Labeling
Recorded data is placed on a network drive, which will be fetched from a data labeling system.
I use an internal instance of [Label Studio](https://labelstud.io/), mainly because it was more or less the best tool I found for audio sequence labeling.

