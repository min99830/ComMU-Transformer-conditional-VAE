# ComMU with Transformer based Conditional VAE

Forked from [link](https://github.com/POZAlabs/ComMU-code)

## Getting Started
- Note : This Project requires python version `3.8.12`. Set the virtual environment if needed.
### Setup
1. Clone this repository
2. Install required packages
    ```
    pip install -r requirements.txt
    ```
### Download the Preprocessed Data

All you have to do is just

```
cd dataset && ./download.sh && cd ..
```

If you have some problems downloading the dataset, try

```
chmod u+x ./dataset/download.sh
```

## Training
```
$ python3 -m torch.distributed.launch --nproc_per_node=4 ./trainCVAE.py --data_dir ./dataset/commu_midi/output_npy --work_dir {./working_direcoty}
```

## Generating - this will be implemented
- generation involves choice of metadata, regarding which type of music(midi file) we intend to generate. the example of command is showed below.
    ```
    $ python3 generateCVAE.py \
    --checkpoint_dir {./working_directory/checkpoint_best.pt} \
    --output_dir {./output_dir} \
    --bpm 70 \
    --audio_key aminor \
    --time_signature 4/4 \
    --pitch_range mid_high \
    --num_measures 8 \
    --inst acoustic_piano \
    --genre newage \
    --min_velocity 60 \
    --max_velocity 80 \
    --track_role main_melody \
    --rhythm standard \
    --chord_progression Am-Am-Am-Am-Am-Am-Am-Am-G-G-G-G-G-G-G-G-F-F-F-F-F-F-F-F-E-E-E-E-E-E-E-E-Am-Am-Am-Am-Am-Am-Am-Am-G-G-G-G-G-G-G-G-F-F-F-F-F-F-F-F-E-E-E-E-E-E-E-E \
    --num_generate 3
    ```
