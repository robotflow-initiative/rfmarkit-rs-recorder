from .calibrate import entry_point as calibrate
from .serve import entry_point as serve
from .post_process import entry_point as post_process

_EXPORTS = {
    "serve": serve,
    "calibrate": calibrate,
}
