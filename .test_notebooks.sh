#!/bin/bash
set -e
FILES=`find .  -name "*.ipynb" ! -name '*checkpoint*'`
for NOTEBOOK in $FILES
do
    echo ">>> Processing ${NOTEBOOK}"
    BASENAME=${NOTEBOOK%.*}
    PYTHONFILE=${BASENAME}".py"
    JOBNAME=`basename ${BASENAME}`
    if [[ $NOTEBOOK =~ .*f16.* ]] || [[ $NOTEBOOK =~ .*csaf_env_example.* ]]; then
      EXAMPLE="f16"
    else
    if [[ $NOTEBOOK =~ .*cansat.* ]]; then
      EXAMPLE="cansat"
    else
    if [[ $NOTEBOOK =~ .*dubins.* ]]; then
      EXAMPLE="rejoin"
    else
      echo "Unknown example!"
      exit 1
    fi
    fi
    fi
    # Now actually execute the commands
    echo ">>> Adjust example paths"
    ln -s ${PWD}/examples/${EXAMPLE} ./csaf-system
    echo ">>> Testing ${NOTEBOOK}"
    jupyter nbconvert --to python $NOTEBOOK
    ipython $PYTHONFILE
    echo ">>> Testing ${NOTEBOOK} complete!"
    rm -rf ./csaf-system
done