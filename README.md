# Online Identification of New Speakers

This repository is the update of: https://github.com/julik43/Generic-Speaker-Verificator

In this repository an approach to speaker identification through verification is presented.

There are three posibilities with the code given:
1. Train your own models.
2. Built an Identification system with the weights of your trained model.
3. Use an identification system with weights given.


# 1. Train your own models

Update the correct path on train_speakers.txt, valid_speakers.txt and test_speakers.txt to the files you are going to use.

Use a configuration like one in the configurations folder and run the models like this:

bash run.sh config.json 1

This run.sh script recieves the desired configuration of the model and the amount of processes to generate data. The basic case for the processes is 1, it is recommended with the configurations given to use as processes 1, 2, 4, 8 or 16.

For this project, VoxCeleb database was changed from m4a format to flac format using the bash code "change_m4a_to_flac.sh".

Note: it is important to mention that "change_m4a_to_flac.sh" localize the audios of the third level of folders from the path given.

Note 2: config.json has the configuration of configurations/config_VGG11_EmphSpec.json . In the folder configurations you can find the rest of configurations used during this project.


# 2. Built an Identification system with the weights of your trained model.

Update the path of the weights in identification_system.json and all data needed.

Run the identification model like this:

python identification_system.py identification_system.json


# 3. Use an identification system with weights given.

Download "2weights.ckpt.data-00000-of-00001" from: http://calebrascon.info/oneshotid/2weights.ckpt.data-00000-of-00001 and move it to VGG11_EmphSpec folder

Run the identification model like this:

python identification_system.py identification_system.json