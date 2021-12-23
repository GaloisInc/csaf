#!/bin/bash
set -e
FILES=`find .  -wholename "./notebooks/*.ipynb" ! -name '*checkpoint*'`
export PYTHONPATH=${PYTHONPATH}:${PWD}/
echo "PYTHONPATH=${PYTHONPATH}"
for NOTEBOOK in $FILES
do
    echo ">>> Processing ${NOTEBOOK}"
    BASENAME=${NOTEBOOK%.*}
    PYTHONFILE=${BASENAME}".py"

    # if [[ $NOTEBOOK =~ .*f16.* ]] || [[ $NOTEBOOK =~ .*csaf_env_example.* ]]; then
    #   EXAMPLE="f16"
    # else
    # if [[ $NOTEBOOK =~ .*cansat.* ]]; then
    #   EXAMPLE="cansat"
    # else
    # if [[ $NOTEBOOK =~ .*dubins.* ]]; then
    #   EXAMPLE="rejoin"
    # else
    #   echo "Unknown example!"
    #   exit 1
    # fi
    # fi
    # fi
    # # Now actually execute the commands
    # echo ">>> Adjust example paths: ln -s ${PWD}/examples/${EXAMPLE} /csaf-system"
    # sudo ln -s ${PWD}/examples/${EXAMPLE} /csaf-system
    echo ">>> Converting ${NOTEBOOK}: jupyter nbconvert --to python $NOTEBOOK"
    jupyter nbconvert --to python $NOTEBOOK
    echo ">>> Testing ${NOTEBOOK}: ipython $PYTHONFILE"
    ipython $PYTHONFILE
    # echo ">>> Testing ${NOTEBOOK} complete! Removing symlink: rm /csaf-system"
    # sudo rm /csaf-system
done
