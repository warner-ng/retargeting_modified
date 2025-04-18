# Humanoid-retarget
Humanoid retargeting tool using Full-body IK solver

## How to install

First of all, please install Pinocchio using Conda. Make sure the version of **pinocchio >= 3.0.0**.

```
conda install pinocchio -c conda-forge
```

Then, you can install all the dependencies using pip.

```
pip install -r requirements.txt
```

## How to retarget

First, you need to put all the **.tar.bz2** files from [AMASS](https://amass.is.tue.mpg.de/) into **data** folder.

Then run ``bash pre.sh <EXTRACT_FLAG>`` to do the preprocessing. For example:
```
bash pre.sh extract
```

You can then run ``bash run.sh <NUM_PROCESS>``. For example:
```
bash run.sh 4
```

You can also run ``bash vis.sh <ROBOT_NAME>`` to visualize the retargeting results.

* NOTE: Make sure you have more than 24GB of RAM for retargeting.


## Support List

$\bullet$ H1

$\bullet$ H1_2

$\bullet$ G1

$\bullet$ GR1T2

$\bullet$ ORCA
