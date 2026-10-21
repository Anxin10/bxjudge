"""Local Label Studio Community settings: no optional Redis queue connections."""

# Upstream reads Django settings while importing its edition settings; populate
# the base first so those reads see the normal configuration.
from label_studio.core.settings.base import *
from label_studio.core.settings.label_studio import *

# Community edition runs synchronously; upstream still probes its default queue
# during import. An empty queue registry disables that unused connection.
RQ_QUEUES = {}
