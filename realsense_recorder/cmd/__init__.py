from .local_record_seq import entry_point as LocalRecordSeq
from .local_capture_static_interactive import entry_point as LocalCaptureStaticInteractive
from .remote_record_seq import entry_point as RemoteRecordSeq

_EXPORTS = {
    "local_record_seq": LocalRecordSeq,
    "local_capture_static_interactive": LocalCaptureStaticInteractive,
    "remote_record_seq": RemoteRecordSeq,
}
