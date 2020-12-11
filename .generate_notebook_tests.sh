#!/bin/bash
CONFIG_FILE="generated-notebooks-config.yml"
echo "image: galoisinc/csaf:latest

before_script:
  - 'which ssh-agent || ( apt-get update -y && apt-get install openssh-client -y )'
  - eval \$(ssh-agent -s)
  - mkdir -p ~/.ssh
  - chmod 700 ~/.ssh
  - export PYTHONPATH=\${PYTHONPATH}:\${PWD}/src:\${PWD}/examples/f16:\${PWD}/examples/inverted-pendulum:\${PWD}/examples/rejoin:\${PWD}/examples/cansat:/csaf-system" > $CONFIG_FILE

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
    echo "
notebook_${JOBNAME}:
  stage: test
  script:
    - echo \">>> Adjust example paths\"
    - ln -s \${PWD}/examples/${EXAMPLE} /csaf-system
    - echo \">>> Testing ${NOTEBOOK}\"
    - jupyter nbconvert --to python $NOTEBOOK
    - ipython $PYTHONFILE
    - echo \">>> Testing ${NOTEBOOK} complete!\"
    - rm /csaf-system" >> $CONFIG_FILE
done
