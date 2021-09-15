#!/bin/bash
fgfs --fdm=null --native-fdm=socket,in,30,,5605,udp --disable-sound --aircraft=org.flightgear.fgaddon.stable_2020.ZF_Navy_free_balloon \
    --generic=socket,in,30,,5507,udp,f16 --multiplay=out,10,127.0.0.1,5001 --multiplay=in,10,127.0.0.1,5000 --callsign=Csaf2 --enable-terrasync
