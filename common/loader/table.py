#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Union, Dict
import os, sys, io
from abc import ABCMeta, abstractmethod


from .text import  AbstractLoader


class AnnotatedTextLoader(AbstractLoader):

    def __init__(self, file_path, text_a: int, text_b: Optional[int], label: int, label_type: type, header: Optional[int] = None, sep: str = " "):
        super(AnnotatedTextLoader, self).__init__(file_path, n_minibatch=0)
        self._sep = sep
        self._cfg_column_no = {
            "text_a":text_a,
            "text_b":text_b,
            "label":label
        }
        self._label_type = label_type
        self._header = header
    
    @property
    def columns(self):
        return self._cfg_column_no

    def __iter__(self) -> Dict[str, Union[str, int, float]]:
        with io.open(self._path, mode="r") as ifs:
            # skip first n lines
            if self._header is not None:
                for _ in range(self._header):
                    next(ifs)
            # raad each line
            for line in ifs:
                lst_fields = line.strip().split(self._sep)
                payload = {
                    "text_a": lst_fields[self._cfg_column_no["text_a"]],
                    "text_b": "" if self._cfg_column_no["text_b"] is None else lst_fields[self._cfg_column_no["text_b"]],
                    "label": self._label_type(lst_fields[self._cfg_column_no["label"]])
                }
                yield payload


class MinibatchAnnotatedTextLoader(AnnotatedTextLoader):

    def __init__(self, file_path, n_minibatch: int, text_a: int, text_b: Optional[int], label: int, label_type: type, header: Optional[int] = None, sep: str = " "):
        super(MinibatchAnnotatedTextLoader, self).__init__(file_path, text_a, text_b, label, label_type, header, sep)
        self._n_mb = n_minibatch

    def __iter__(self):

        iter_parent = super(MinibatchAnnotatedTextLoader, self).__iter__()
        lst_ret = []
        for payload in iter_parent:
            lst_ret.append(payload)
            if len(lst_ret) >= self._n_mb:
                yield lst_ret
                lst_ret = []
        if len(lst_ret) > 0:
            yield lst_ret


class PayloadContainer(object):

    def __init__(self, container: AnnotatedTextLoader, payload_key: str):
        self._container = container
        self._payload_key = payload_key

        assert payload_key in container.columns, f"invalid payload key was specified: {payload_key}"

    def __iter__(self):

        for payload in self._container:
            yield payload[self._payload_key]