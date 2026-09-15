from graft.models import SdxlIpGenerator


class _FakePipe:
    def __init__(self):
        self.scale = None

    def set_ip_adapter_scale(self, s):
        self.scale = s


def test_set_scale_before_and_after_pipe_load():
    g = SdxlIpGenerator.__new__(SdxlIpGenerator)  # bypass heavy __init__
    g.ip_scale = 0.6
    g._pipe_ip = None
    g.set_scale(0.4)  # before load: just records
    assert g.ip_scale == 0.4
    g._pipe_ip = _FakePipe()
    g.set_scale(0.8)  # after load: pushes to the pipe
    assert g.ip_scale == 0.8
    assert g._pipe_ip.scale == 0.8
