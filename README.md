# rice_atlas
 
#Installation V1 (mais faire plutot la V2)
python3.10 -m venv napari-env-py310
source napari-env-py310/activate
pip install napari[all]
cd chemin/vers/le/repo
pip install -e .

#Installation V2
micromamba create -n napari310 -c conda-forge napari pyqt pybind11 connected-components-3d python=3.10 
eval "$(micromamba shell hook --shell bash)"
micromamba activate
micromamba activate napari310
cd ~/Python_prog/rice_atlas_stage_thomas
pip install -e . #This should last
micromamba install -c conda-forge pyqt

cd ~/Python_prog/rice_atlas_stage_thomas/src/rice_atlas/preproc
python setup.py build_ext --inplace

napari

#After this, you should be able to find the Segmentation 3D plugin by clicking in Plugins Menuitem
#And this should work. If not, contact romain.fernandez@cirad.fr



#Usage
Click on Charger un volume, select a Xray 3D volume in grayscale. Then set the slicer at a place where we see well the roottap (we should see one single inside roottap volume, and the roots around it).
Then click on Chemin du modeles pour racines, to load the neural network. Navigate to the ressources dir in the parent path of the git repo on you computer. Go to root_model, and select segformer3d_epoch_10.pth

The prediction starts when clicking run. Take care of the batch size, done for a 32 GB RAM GPU. Instead reduce it
In the shell there will be some things happening.

