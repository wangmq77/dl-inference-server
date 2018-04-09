# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from builtins import range
from future.utils import iteritems
from ctypes import *
import numpy as np
import pkg_resources
import inference_server.api.model_config_pb2
from inference_server.api.server_status_pb2 import ServerStatus

_crequest_path = pkg_resources.resource_filename('inference_server.api', 'libcrequest.so')
_crequest = cdll.LoadLibrary(_crequest_path)

_crequest_error_new = _crequest.ErrorNew
_crequest_error_new.restype = c_void_p
_crequest_error_new.argtypes = [c_char_p]
_crequest_error_del = _crequest.ErrorDelete
_crequest_error_del.argtypes = [c_void_p]
_crequest_error_isok = _crequest.ErrorIsOk
_crequest_error_isok.restype = c_bool
_crequest_error_isok.argtypes = [c_void_p]
_crequest_error_msg = _crequest.ErrorMessage
_crequest_error_msg.restype = c_char_p
_crequest_error_msg.argtypes = [c_void_p]
_crequest_error_serverid = _crequest.ErrorServerId
_crequest_error_serverid.restype = c_char_p
_crequest_error_serverid.argtypes = [c_void_p]

_crequest_status_ctx_new = _crequest.ServerStatusContextNew
_crequest_status_ctx_new.restype = c_void_p
_crequest_status_ctx_new.argtypes = [POINTER(c_void_p), c_char_p, c_char_p, c_bool]
_crequest_status_ctx_del = _crequest.ServerStatusContextDelete
_crequest_status_ctx_del.argtypes = [c_void_p]
_crequest_status_ctx_get = _crequest.ServerStatusContextGetServerStatus
_crequest_status_ctx_get.restype = c_void_p
_crequest_status_ctx_get.argtypes = [c_void_p, POINTER(c_char_p), POINTER(c_uint32)]

_crequest_infer_ctx_new = _crequest.InferContextNew
_crequest_infer_ctx_new.restype = c_void_p
_crequest_infer_ctx_new.argtypes = [POINTER(c_void_p), c_char_p, c_char_p, c_bool]
_crequest_infer_ctx_del = _crequest.InferContextDelete
_crequest_infer_ctx_del.argtypes = [c_void_p]
_crequest_infer_ctx_set_options = _crequest.InferContextSetOptions
_crequest_infer_ctx_set_options.restype = c_void_p
_crequest_infer_ctx_set_options.argtypes = [c_void_p, c_void_p]
_crequest_infer_ctx_run = _crequest.InferContextRun
_crequest_infer_ctx_run.restype = c_void_p
_crequest_infer_ctx_run.argtypes = [c_void_p]

_crequest_infer_ctx_options_new = _crequest.InferContextOptionsNew
_crequest_infer_ctx_options_new.restype = c_void_p
_crequest_infer_ctx_options_new.argtypes = [POINTER(c_void_p), c_uint64]
_crequest_infer_ctx_options_del = _crequest.InferContextOptionsDelete
_crequest_infer_ctx_options_del.argtypes = [c_void_p]
_crequest_infer_ctx_options_add_raw = _crequest.InferContextOptionsAddRaw
_crequest_infer_ctx_options_add_raw.restype = c_void_p
_crequest_infer_ctx_options_add_raw.argtypes = [c_void_p, c_void_p, c_char_p]
_crequest_infer_ctx_options_add_class = _crequest.InferContextOptionsAddClass
_crequest_infer_ctx_options_add_class.restype = c_void_p
_crequest_infer_ctx_options_add_class.argtypes = [c_void_p, c_void_p, c_char_p, c_uint64]

_crequest_infer_ctx_input_new = _crequest.InferContextInputNew
_crequest_infer_ctx_input_new.restype = c_void_p
_crequest_infer_ctx_input_new.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
_crequest_infer_ctx_input_del = _crequest.InferContextInputDelete
_crequest_infer_ctx_input_del.argtypes = [c_void_p]
_crequest_infer_ctx_input_set_raw = _crequest.InferContextInputSetRaw
_crequest_infer_ctx_input_set_raw.restype = c_void_p
_crequest_infer_ctx_input_set_raw.argtypes = [c_void_p, c_void_p, c_uint64]

_crequest_infer_ctx_result_new = _crequest.InferContextResultNew
_crequest_infer_ctx_result_new.restype = c_void_p
_crequest_infer_ctx_result_new.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
_crequest_infer_ctx_result_del = _crequest.InferContextResultDelete
_crequest_infer_ctx_result_del.argtypes = [c_void_p]
_crequest_infer_ctx_result_dtype = _crequest.InferContextResultDataType
_crequest_infer_ctx_result_dtype.restype = c_void_p
_crequest_infer_ctx_result_dtype.argtypes = [c_void_p, POINTER(c_uint32)]
_crequest_infer_ctx_result_next_raw = _crequest.InferContextResultNextRaw
_crequest_infer_ctx_result_next_raw.restype = c_void_p
_crequest_infer_ctx_result_next_raw.argtypes = [c_void_p, c_uint64, POINTER(c_char_p),
                                                POINTER(c_uint64)]
_crequest_infer_ctx_result_class_cnt = _crequest.InferContextResultClassCount
_crequest_infer_ctx_result_class_cnt.restype = c_void_p
_crequest_infer_ctx_result_class_cnt.argtypes = [c_void_p, c_uint64, POINTER(c_uint64)]
_crequest_infer_ctx_result_next_class = _crequest.InferContextResultNextClass
_crequest_infer_ctx_result_next_class.restype = c_void_p
_crequest_infer_ctx_result_next_class.argtypes = [c_void_p, c_uint64, POINTER(c_uint64),
                                                  POINTER(c_float), POINTER(c_char_p)]


def _raise_if_error(err):
    if err.value is not None:
        ex = InferenceServerException(err)
        isok = _crequest_error_isok(err)
        _crequest_error_del(err)
        if not isok:
            raise ex

def _raise_error(msg):
    err = c_void_p(_crequest_error_new(msg))
    ex = InferenceServerException(err)
    _crequest_error_del(err)
    raise ex


class InferenceServerException(Exception):
    """Exception indicating non-Success status."""

    def __init__(self, err):
        """Initialize exception from an Error

        err - c_void_p pointer to the Error
        """
        self._msg = None
        self._server_id = None
        if (err is not None) and (err.value is not None):
            self._msg = _crequest_error_msg(err)
            self._server_id = _crequest_error_serverid(err)

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        if self._server_id is not None:
            msg = '[' + self._server_id + '] - ' + msg
        return msg

    def message(self):
        """
        @return the message associated with this exception, or None if no
        message.
        """
        return self._msg

    def server_id(self):
        """
        @return the ID of the server associated with this exception, or
        None if no server is associated.
        """
        return self._server_id

class ServerStatusContext:
    """
    Performs a status request to an inference server. Can get
    status for all models on the server or for a single model.
    """

    def __init__(self, url, model_name=None, verbose=False):
        """Initialize the context.

        url - The inference server URL, e.g. localhost:8000.

        model_name - The name of the model to get status for, or
        None to get status for all models.

        verbose - If True generate verbose output.
        """
        self._ctx = c_void_p()
        _raise_if_error(
            c_void_p(
                _crequest_status_ctx_new(byref(self._ctx), url, model_name, verbose)))

    def __del__(self):
        # when module is unloading may get called after
        # _crequest_status_ctx_del has been released
        if _crequest_status_ctx_del is not None:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """
        Close the context. Any future calls to get_server_status() will
        result in an Error.
        """
        _crequest_status_ctx_del(self._ctx)
        self._ctx = None

    def get_server_status(self):
        """
        Contact the inference server and get status.

        @return ServerStatus protobuf containing the status.

        Raises InferenceServerException if unable to get status.
        """
        if self._ctx is None:
            _raise_error("ServerStatusContext is closed")

        cstatus = c_char_p()
        cstatus_len = c_uint32()
        _raise_if_error(c_void_p(_crequest_status_ctx_get(self._ctx, byref(cstatus), byref(cstatus_len))))

        status_buf = cast(cstatus, POINTER(c_byte * cstatus_len.value))[0]

        status = ServerStatus()
        status.ParseFromString(status_buf)
        return status


class InferContext:
    """
    An InferContext object is used to run inference on an inference
    server for a specific model. Once created an InferContext object
    can be used repeatedly to perform inference using the model.
    """
    class ResultFormat:
        # RAW - All values of the output are returned as an numpy
        # array of the appropriate type.
        RAW = 1,
        # CLASS - Specified as tuple (CLASS, k). Top 'k' results
        # are returned as an array of (index, value, label) tuples.
        CLASS = 2

    def __init__(self, url, model_name, verbose=False):
        """Initialize the context.

        url - The inference server URL, e.g. localhost:8000.

        model_name - The name of the model to use for inference.

        verbose - If True generate verbose output.
        """
        self._ctx = c_void_p()
        _raise_if_error(
            c_void_p(
                _crequest_infer_ctx_new(byref(self._ctx), url, model_name, verbose)))

    def __del__(self):
        # when module is unloading may get called after
        # _crequest_infer_ctx_del has been released
        if _crequest_infer_ctx_del is not None:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _get_result_numpy_dtype(self, result):
        ctype = c_uint32()
        _raise_if_error(c_void_p(_crequest_infer_ctx_result_dtype(result, byref(ctype))))
        if ctype.value == model_config_pb2.TYPE_BOOL:
            return np.bool_
        elif ctype.value == model_config_pb2.TYPE_UINT8:
            return np.uint8
        elif ctype.value == model_config_pb2.TYPE_UINT16:
            return np.uint16
        elif ctype.value == model_config_pb2.TYPE_UINT32:
            return np.uint32
        elif ctype.value == model_config_pb2.TYPE_UINT64:
            return np.uint64
        elif ctype.value == model_config_pb2.TYPE_INT8:
            return np.int8
        elif ctype.value == model_config_pb2.TYPE_INT16:
            return np.int16
        elif ctype.value == model_config_pb2.TYPE_INT32:
            return np.int32
        elif ctype.value == model_config_pb2.TYPE_INT64:
            return np.int64
        elif ctype.value == model_config_pb2.TYPE_FP16:
            return np.float16
        elif ctype.value == model_config_pb2.TYPE_FP32:
            return np.float32
        elif ctype.value == model_config_pb2.TYPE_FP64:
            return np.float64
        _raise_error("unknown result datatype " + ctype.value)

    def close(self):
        """
        Close the context. Any future calls to object will result in an
        Error.
        """
        _crequest_infer_ctx_del(self._ctx)
        self._ctx = None

    def run(self, inputs, outputs, batch_size=1):
        """
        Run inference using the supplied 'inputs' to calculate the outputs
        specified by 'outputs'.

        inputs - Dictionary from input name to the value(s) for that
        input. An input value is specified as a numpy array. Each
        input in the dictionary maps to a list of values (i.e. a list
        of numpy array objects), where the length of the list must
        equal the 'batch_size'.

        outputs - Dictionary from output name to an output format. The
        inference server will use the input values to calculate the value for
        the requested outputs. See return value discussion for how output
        format is used.

        batch_size - The number of batches specified by the inputs.

        Returns a dictionary from output name to the list of values for that
        output (one list element for each batch). The format of a value
        returned for an output depends on the output format specified in
        'outputs'. Supported output formats are:
          RAW   - numpy array of the appropriate type.
          CLASS - Specified as tuple (CLASS, k). Top 'k' output values are
                  returned as an array of (index, value, label) tuples.

        Raises InferenceServerException if all inputs are not specified, if
        the size of input data does not match expectations, if unknown output
        names are specified or if server fails to perform inference.
        """
        # Set run options using formats specified in 'outputs'
        options = c_void_p()
        try:
            _raise_if_error(c_void_p(_crequest_infer_ctx_options_new(byref(options), batch_size)))

            for (output_name, output_format) in iteritems(outputs):
                if output_format == InferContext.ResultFormat.RAW:
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_options_add_raw(self._ctx, options, output_name)))
                elif isinstance(output_format, tuple) and (output_format[0] == InferContext.ResultFormat.CLASS):
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_options_add_class(
                                self._ctx, options, output_name, c_uint64(output_format[1]))))
                else:
                    _raise_error("unrecognized output format")

            _raise_if_error(c_void_p(_crequest_infer_ctx_set_options(self._ctx, options)))

        finally:
            _crequest_infer_ctx_options_del(options)

        # Set the input values
        for (input_name, input_values) in iteritems(inputs):
            input = c_void_p()
            try:
                _raise_if_error(
                    c_void_p(_crequest_infer_ctx_input_new(byref(input), self._ctx, input_name)))

                for input_value in input_values:
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_input_set_raw(
                                input, input_value.ctypes.data_as(c_void_p),
                                c_uint64(input_value.size * input_value.itemsize))))
            finally:
                _crequest_infer_ctx_input_del(input)

        # Run inference...
        _raise_if_error(c_void_p(_crequest_infer_ctx_run(self._ctx)))

        # Create the result map.
        results = dict()
        for (output_name, output_format) in iteritems(outputs):
            result = c_void_p()
            try:
                _raise_if_error(
                    c_void_p(_crequest_infer_ctx_result_new(byref(result), self._ctx, output_name)))
                result_dtype = self._get_result_numpy_dtype(result)
                results[output_name] = list()
                if output_format == InferContext.ResultFormat.RAW:
                    for b in range(batch_size):
                        cval = c_char_p()
                        cval_len = c_uint64()
                        _raise_if_error(
                            c_void_p(
                                _crequest_infer_ctx_result_next_raw(
                                    result, b, byref(cval), byref(cval_len))))
                        val_buf = cast(cval, POINTER(c_byte * cval_len.value))[0]
                        results[output_name].append(np.copy(np.frombuffer(val_buf, dtype=result_dtype)))
                elif isinstance(output_format, tuple) and (output_format[0] == InferContext.ResultFormat.CLASS):
                    for b in range(batch_size):
                        classes = list()
                        ccnt = c_uint64()
                        _raise_if_error(
                           c_void_p(_crequest_infer_ctx_result_class_cnt(result, b, byref(ccnt))))
                        for cc in range(ccnt.value):
                            cidx = c_uint64()
                            cprob = c_float()
                            clabel = c_char_p()
                            _raise_if_error(
                                c_void_p(
                                    _crequest_infer_ctx_result_next_class(
                                        result, b, byref(cidx), byref(cprob), byref(clabel))))
                            classes.append((cidx.value, cprob.value, clabel.value))
                        results[output_name].append(classes)
                else:
                    _raise_error("unrecognized output format")
            finally:
                _crequest_infer_ctx_result_del(result)

        return results
