# Deep visual real estate geolocalisation

![Geolocalisation](img.jpg){width=200px}

## Context

The objective of this project is to find the address of a real estate property photo by comparing it to street and satellite view photos of known addresses.
There are different methods for achieving this which can be split up between students:
* Street to street view (visual intra-modality, see EigenPlaces)
* Street to satellite view (visual cross-modality, see Sample4Geo)
* Street to street & satellite view (visual multi-modality, no articles found yet)
* Street & satellite to street, garden, and inside-out images (visual multi-modality, definitely no articles, *data coming soon*)

It is up to you to explore the scientific literature around this subject in order to find and compare the different existing methods after fine-tuning (beyond the ones already identified within the bibliography document), or conversely identify an opportunity to contribute to the scientific literature with an innovative model and approach (the objective being to publish such paper eventually).

## Getting started

To set up the environment, use the ```bash env_setup.sh``` command.

The jupyter notebook ```geolocalisation.ipynb``` contains code for inference on single image couples with the EigenPlaces and Sample4Geo models.

The data is available on demand to Matthieu (matthieus@homiwoo.com) or Simon (simon@homiwoo.com) at https://drive.google.com/drive/folders/1e9wMZF9R0aGwxvMaCHqtHgws7Jh_zB_A
The ```data.zip``` should be unzipped in this parent folder.

The bibliography document is in the same google drive.