"""RN-146 tester-only helpers: file-like sources with fault injection."""
import io


class TrackingBytesIO(io.BytesIO):
    def __init__(self, data, max_chunk_size=None):
        super().__init__(data)
        self.max_chunk_size = max_chunk_size
        self.read_sizes = []
        self.close_calls = 0

    def read(self, size=-1):
        assert size >= 0, "TsFileReader must use explicitly sized reads"
        self.read_sizes.append(size)
        if self.max_chunk_size is not None:
            size = min(size, self.max_chunk_size)
        return super().read(size)

    def close(self):
        self.close_calls += 1
        super().close()


class FailingBytesIO(io.BytesIO):
    """fail_at: 第 N 次 read 抛错；fail_after_bytes: 累计消费 N 字节后抛错。"""

    def __init__(self, data, fail_at=None, fail_after_bytes=None, exc=OSError):
        super().__init__(data)
        self.fail_at = fail_at
        self.fail_after_bytes = fail_after_bytes
        self.exc = exc
        self.read_calls = 0
        self.consumed = 0
        self.close_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        if self.fail_at is not None and self.read_calls == self.fail_at:
            raise self.exc("injected failure at read call %d" % self.fail_at)
        if self.fail_after_bytes is not None and self.consumed >= self.fail_after_bytes:
            raise self.exc("injected stream cut after %d bytes" % self.consumed)
        chunk = super().read(size)
        self.consumed += len(chunk)
        return chunk

    def close(self):
        self.close_calls += 1
        super().close()


class NoSizeRead:
    """seek/tell 可用，但 read() 不接受 size 参数。"""

    def __init__(self, data):
        self._buf = io.BytesIO(data)

    def seek(self, *a):
        return self._buf.seek(*a)

    def tell(self):
        return self._buf.tell()

    def read(self):
        return self._buf.read()


class TextSource(NoSizeRead):
    """read(size) 返回 str 而非 bytes。"""

    def read(self, size=-1):
        return "not-bytes"


class ReadOnlySource:
    """无 seek/tell。"""

    def __init__(self, data):
        self._buf = io.BytesIO(data)

    def read(self, size=-1):
        return self._buf.read(size)


class OverReturnSource(io.BytesIO):
    """违约：返回超过请求长度的数据。"""

    def read(self, size=-1):
        if size > 0:
            return super().read(size + 16)
        return super().read(size)


class NegativeLength(io.BytesIO):
    """违约：seek(0,2) 与 tell() 均报告负长度（用例 54）。"""

    def __init__(self, data):
        super().__init__(data)
        self._neg = False

    def seek(self, offset, whence=0):
        self._neg = whence == 2
        super().seek(offset, whence)
        return -1 if self._neg else super().tell()

    def tell(self):
        return -1 if self._neg else super().tell()


class BytesPathLike:
    """os.PathLike 实现，__fspath__ 返回 bytes（用例 3）。"""

    def __init__(self, path):
        self._path = path

    def __fspath__(self):
        return self._path
