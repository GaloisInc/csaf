#!/bin/bash
fgfs --fdm=null --native-fdm=socket,in,30,,5505,udp --disable-sound --aircraft=f16-block-20 \
    --fg-aircraft=/home/mpodhradsky/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020 \
    --generic=socket,in,30,,5506,udp,f16 --multiplay=out,10,127.0.0.1,5000 --multiplay=in,10,127.0.0.1,5001 --callsign=Csaf1
