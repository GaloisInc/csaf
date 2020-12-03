""" Control System Analysis Framework (CSAF)

CSAF is a middleware environment for describing, simulating, and analyzing closed loop dynamical
systems. A CSAF component's primary interface is ZeroMQ subscribe and publish system
components, which communicate input/output of a lump abstracted component as ROSmsgs. It
allows component creation using common system representations, especially for specifying controllers. The
architecture is supportive of systems to be simulated with diverse components, agnostic to the languages
and platforms from which they were implemented.
"""

import logging

csaf_logger = logging.getLogger("csaf")

has_global_config = False
