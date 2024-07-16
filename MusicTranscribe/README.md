**MusicTranscribe** will split your audio file into its various sound sources and then convert each one to midi. The output of MusicTranscribe is a .zip of each sound sources midi file as well as a chord_chart.txt, documenting each chord in the input file. MusicTranscribe is compatible with .wav, .mp3, .mp4, and .FLAC audio files. Other than source separation and midi conversion, MusicTranscribe has many other built in signal processing methods. A few examples are chromagram to audio file conversion, harmonic and percussive extraction and stereo to mono conversion.

**------Installation:-------**
Download the whole MusicTranscribe repository. 
Then, in your terminal, cd to the MusicTranscribe folder with 'cd /path/to/MusicTranscribe'.
Make sure you have poetry and flask installed on your hard drive. If you don't have poetry installed, you can do it in you terminal with 'pip install poetry' and 'pip install flask'
Now run 'poetry install' to install MusicTranscribe dependencies from the pyproject.toml. 

**------Instructions:-------**
To use MusicTranscribe from the shell, first cd to the MusicTranscribe folder in your filesystem with 'cd downloads' or 'cd desktop' and then 'cd MusicTranscribe'. poetry ensures MusicTranscribes poetry dependencies are active, so append 'poetry run' to the beginning of your command. Here is an example of a command line argument to use MusicTranscribe: 

'(base) YourName MusicTranscribe % poetry run python3 music_transcriber/MusicAssist.py /valid/path/to/audio/file.mp3


OR, you can run MusicAssist from the webpage with the following command: 'poetry run python3 music_transcriber/app.py' and opening the the IP address.

This would look like the following in your terminal: 

(base) YourName MusicTranscribe % 'poetry run python3 music_transcriber/app.py'. 

Then you can go to the webpage at the given IP address and upload your file there.

**------Citations:------**
aholman and Daudzarif created the front end of the user friendly webpage bundled in with MusicTranscribe.

seperate_foreground_background() and  chroma_enhance() are by Brian McAfee from Librosa:
Code source: Brian McFee
License: ISC
Code source: Brian McFee
License: ISC
sphinx_gallery_thumbnail_number = 5


This project also uses spleeter, a tensorflow powerered open-source audio recognition program by deezer.
@article{spleeter2020,
  doi = {10.21105/joss.02154},
  url = {https://doi.org/10.21105/joss.02154},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {50},
  pages = {2154},
  author = {Romain Hennequin and Anis Khlif and Felix Voituret and Manuel Moussallam},
  title = {Spleeter: a fast and efficient music source separation tool with pre-trained models},
  journal = {Journal of Open Source Software},
  note = {Deezer Research}
}
@misc{musdb18,
  author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
  title        = {The {MUSDB18} corpus for music separation},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1117372},
  url          = {https://doi.org/10.5281/zenodo.1117372}
}


As well as basic-pitch, a tensorflow powered open-source midi recognition engine.
@inproceedings{2022_BittnerBRME_LightweightNoteTranscription_ICASSP,
  author= {Bittner, Rachel M. and Bosch, Juan Jos\'e and Rubinstein, David and Meseguer-Brocal, Gabriel and Ewert, Sebastian},
  title= {A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation},
  booktitle= {Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  address= {Singapore},
  year= 2022,
}
