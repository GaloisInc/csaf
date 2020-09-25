#!/bin/bash

# To be run from f16 root directory via ./scripts/install_fg.sh
FG_ROOT=/usr/share/games/flightgear
BASENAME=http://ns334561.ip-5-196-65.eu/~fgscenery/WS2.0

# Install flightgear
add-apt-repository ppa:saiarcot895/flightgear
sudo apt-get update
apt-get -y install flightgear

# Install desktop recorder
apt-get -y install recordmydesktop

# Install additional scenery data
for LON in 110 120
do
    for LAT in 20 30 40
    do
        FILENAME='w'$LON'n'$LAT'.zip'
        wget $BASENAME/$FILENAME -P /tmp/
        sudo unzip /tmp/$FILENAME -d $FG_ROOT/Scenery/
    done
done

# Install custom protocol for manipulating with F16's control surfaces
sudo ln -s f16.xml $FG_ROOT/Protocol/f16.xml
