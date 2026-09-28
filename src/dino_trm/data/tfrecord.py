"""Minimal TFRecord + tf.train.Example reader.

MOVi ships only as TFDS TFRecords. Both formats are simple enough that parsing them
directly beats adding a ~600MB TensorFlow dependency to read image bytes out of a
container:

    TFRecord   uint64 len | uint32 crc(len) | len bytes payload | uint32 crc(payload)
    Example    Features -> map<string, Feature> -> bytes_list | float_list | int64_list

CRCs are skipped — they are Castagnoli with a mask, and a truncated download fails at
PNG decode anyway, which is the error we actually care about.
"""

from __future__ import annotations

import struct
from typing import Iterator


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        b = buf[i]; i += 1
        v |= (b & 0x7F) << s
        if not b & 0x80:
            return v, i
        s += 7


def _fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield (field_number, wire_type, payload) for one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i); yield fn, wt, v
        elif wt == 2:
            n, i = _varint(buf, i); yield fn, wt, buf[i:i + n]; i += n
        elif wt == 5:
            yield fn, wt, buf[i:i + 4]; i += 4
        elif wt == 1:
            yield fn, wt, buf[i:i + 8]; i += 8
        else:
            raise ValueError(f"unsupported wire type {wt}")


def _packed_varints(buf: bytes) -> list[int]:
    out, i = [], 0
    while i < len(buf):
        v, i = _varint(buf, i)
        out.append(v)
    return out


def parse_example(raw: bytes) -> dict[str, object]:
    """tf.train.Example -> {name: [bytes] | [int]}.

    bytes_list stays a list of bytes; int64_list becomes a list of ints. float_list is
    skipped — nothing here needs it, and decoding it would cost a pass over every
    camera/pose feature in the record.
    """
    out: dict[str, object] = {}
    for fn, _, feats in _fields(raw):
        if fn != 1:                                   # Example.features
            continue
        for efn, _, entry in _fields(feats):          # Features.feature map entry
            if efn != 1:
                continue
            key, vals = None, None
            for kfn, _, payload in _fields(entry):
                if kfn == 1:
                    key = payload.decode()
                elif kfn == 2:                        # Feature
                    for lfn, _, lst in _fields(payload):
                        if lfn == 1:                  # BytesList
                            vals = [v for _, _, v in _fields(lst)]
                        elif lfn == 3:                # Int64List
                            # repeated int64 value = 1, which protobuf may emit packed
                            # (wire type 2) or one varint per tag (wire type 0)
                            vs: list[int] = []
                            for vfn, vwt, vp in _fields(lst):
                                if vfn != 1:
                                    continue
                                if vwt == 0:
                                    vs.append(vp)
                                elif vwt == 2:
                                    vs += _packed_varints(vp)
                            vals = vs
            if key is not None and vals is not None:
                out[key] = vals
    return out


def records(path: str, limit: int | None = None) -> Iterator[bytes]:
    """Yield raw Example payloads from a .tfrecord file."""
    with open(path, "rb") as f:
        n = 0
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return
            ln = struct.unpack("<Q", hdr)[0]
            f.read(4)                                  # crc(len)
            payload = f.read(ln)
            if len(payload) < ln:
                return                                 # truncated tail: stop cleanly
            f.read(4)                                  # crc(payload)
            yield payload
            n += 1
            if limit and n >= limit:
                return
