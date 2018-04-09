// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <curl/curl.h>
#include "src/core/api.pb.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config.h"
#include "src/core/status.pb.h"
#include "src/core/server_status.pb.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================
// Error
//
// Used by client API to report success or failure.
//
class Error {
public:
  // Create an error from a RequestStatus object.
  // @param status - the RequestStatus object
  explicit Error(const RequestStatus& status);

  // Create an error from a code.
  // @param code - The status code for the error
  explicit Error(RequestStatusCode code);

  // Create an error from a code and detailed message.
  // @param code - The status code for the error
  // @param msg - The detailed message for the error
  explicit Error(RequestStatusCode code, const std::string& msg);

  // @return the status code for the error.
  RequestStatusCode Code() const { return code_; }

  // @return the detailed messsage for the error. Empty if no detailed
  // message.
  const std::string& Message() const { return msg_; }

  // @return the ID of the inference server associated with this
  // error, or empty-string if no inference server is associated with
  // the error.
  const std::string& ServerId() const { return server_id_; }

  // @return true if this error is "ok"/"success", false if error
  // indicates a failure.
  bool IsOk() const { return code_ == RequestStatusCode::SUCCESS; }

  // Convenience "success" value can be used as Error::Success to
  // indicate no error.
  static const Error Success;

private:
  friend std::ostream& operator<<(std::ostream&, const Error&);
  RequestStatusCode code_;
  std::string msg_;
  std::string server_id_;
};

//==============================================================================
// ServerStatusContext
//
// A ServerStatusContext object is used to query an inference server
// for status information, including information about the models
// available on that server. Once created a ServerStatusContext object
// can be used repeatedly to get status from the server. For example:
//
//   ServerStatusContext ctx("localhost:8000");
//   ServerStatus status;
//   ctx.GetServerStatus(&status);
//   ...
//   ctx.GetServerStatus(&status);
//   ...
//
// Thread-safety:
//   ServerStatusContext contructors are thread-safe.
//   GetServerStatus() is not thread-safe. For a given
//   ServerStatusContext, calls to GetServerStatus() must be
//   serialized.
//
class ServerStatusContext {
public:
  // Create context that returns information about server and all
  // models on the server.
  // @param server_url - inference server name and port
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  explicit ServerStatusContext(
    const std::string& server_url, bool verbose = false);

  // Create context that returns information about server and one
  // model.
  // @param server_url - inference server name and port
  // @param model-name - get information for this model
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  explicit ServerStatusContext(
    const std::string& server_url, const std::string& model_name,
    bool verbose = false);

  // Contact the inference server and get status
  // @param status - returns the status
  // @return Error object indicating success or failure
  Error GetServerStatus(ServerStatus* status);

private:
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  static size_t ResponseHandler(void*, size_t, size_t, void*);

  // URL for status endpoint on inference server.
  const std::string url_;

  // If true print verbose output
  const bool verbose_;

  // RequestStatus received in server response
  RequestStatus request_status_;

  // Serialized ServerStatus response from server.
  std::string response_;
};

//==============================================================================
// InferContext
//
// An InferContext object is used to run inference on an inference
// server for a specific model. Once created an InferContext object
// can be used repeatedly to perform inference using the
// model. Options that control how inference is performed can be
// changed in between inference runs. For example:
//
//   InferContext ctx("localhost:8000", "mnist");
//   ...
//   Options* options0 = Options::Create();
//   options->SetBatchSize(b);
//   options->AddClassResult(output, topk);
//   ctx.SetRunOptions(*options0);
//   ...
//   ctx.Run(&results0);  // run using options0
//   ctx.Run(&results1);  // run using options0
//   ...
//   Options* options1 = Options::Create();
//   options->AddRawResult(output);
//   ctx.SetRunOptions(*options);
//   ...
//   ctx.Run(&results2);  // run using options1
//   ctx.Run(&results3);  // run using options1
//   ...
//
// Note that the Run() calls are not thread-safe but a new Run() can
// be invoked as soon as the previous completes. The returned result
// objects are owned by the caller and may be retained and accessed
// even after the InferContext object is destroyed.
//
// For more parallelism multiple InferContext objects can access the
// same inference server with not serialization requirements across
// those objects.
//
// Thread-safety:
//   InferContext contructors are thread-safe.
//   All other InferContext methods, and nested class methods are not
//   thread-safe.
//
class InferContext {
public:
  //==============
  // Input
  // An input to the model being used for inference.
  class Input {
  public:
    virtual ~Input() { };

    // @return the name of the input.
    virtual const std::string& Name() const = 0;

    // @return the size in bytes for a single instance of this input
    // (that is, the size when batch-size == 1).
    virtual size_t ByteSize() const = 0;

    // @return the data type of the input.
    virtual DataType DType() const = 0;

    // @return the format of the input.
    virtual ModelInput::Format Format() const = 0;

    // @return the dimensions of the input.
    virtual const DimsList& Dims() const = 0;

    // Prepare this input to receive new tensor values. Forget any
    // existing values that were set by previous calls to
    // Input::SetRaw().
    // @return Error object indicating success or failure
    virtual Error Reset() = 0;

    // Set tensor values for this input from a byte array. The array
    // is not copied and so the it must not be modified or destroyed
    // until this input is no longer needed (that is until the Run()
    // call(s) that use the input have completed). For batched inputs
    // this function must be called batch-size times to provide all
    // tensor values for a batch of this input.
    // @param input - pointer to the array holding tensor value
    // @param input_byte_size - size of the array in bytes, must match
    // the size expected by the input.
    // @return Error object indicating success or failure
    virtual Error SetRaw(
      const uint8_t* input, size_t input_byte_size) = 0;

    // Set tensor values for this input from a byte vector. The vector
    // is not copied and so the it must not be modified or destroyed
    // until this input is no longer needed (that is until the Run()
    // call(s) that use the input have completed). For batched inputs
    // this function must be called batch-size times to provide all
    // tensor values for a batch of this input.
    // @param input - vector holding tensor values
    // @return Error object indicating success or failure
    virtual Error SetRaw(const std::vector<uint8_t>& input) = 0;
  };

  //==============
  // Output
  // An output from the model being used for inference.
  class Output {
  public:
    virtual ~Output() { };

    // @return the name of the output.
    virtual const std::string& Name() const = 0;

    // @return the size in bytes for a single instance of this output
    // (that is, the size when batch-size == 1).
    virtual size_t ByteSize() const = 0;

    // @return the data type of the output.
    virtual DataType DType() const = 0;

    // @return the dimensions of the output.
    virtual const DimsList& Dims() const = 0;
  };

  //==============
  // Result
  // An inference result corresponding to an output.
  class Result {
  public:
    virtual ~Result() { };

    // Format in which result is returned. RAW format is the entire
    // result tensor of values. CLASS format is the top-k highest
    // probability values of the result and the associated class label
    // (if provided by the model).
    enum ResultFormat { RAW = 0, CLASS = 1 };

    // @return the Output object corresponding to this result.
    virtual const std::shared_ptr<Output> GetOutput() const = 0;

    // Get a reference to entire raw result data for a specific batch
    // entry. Returns error if this result is not RAW format.
    // @param batch_idx - return results for this entry the batch
    // @param buf - returns the vector of result bytes
    // @return Error object indicating success or failure
    virtual Error
      GetRaw(size_t batch_idx, const std::vector<uint8_t>** buf) const = 0;

    // Get a reference to raw result data for a specific batch entry
    // at the current "cursor" and advance the cursor by the specified
    // number of bytes. More typically use GetRawAtCursor<T>() method
    // to return the data as a specific type T. Use ResetCursor() to
    // reset the cursor to the beginning of the result. Returns error
    // if this result is not RAW format.
    // @param batch_idx - return results for this entry the batch
    // @param buf - returns pointer to 'adv_byte_size' bytes of data
    // @param adv_byte_size - number of bytes of data to get reference to
    // @return Error object indicating success or failure
    virtual Error GetRawAtCursor(
      size_t batch_idx, const uint8_t** buf, size_t adv_byte_size) = 0;

    // Read a value for a specific batch entry at the current "cursor"
    // from the result tensor as the specified type T and advance the
    // cursor. Use ResetCursor() to reset the cursor to the beginning
    // of the result. Returns error if this result is not RAW format.
    // @param batch_idx - return results for this entry the batch
    // @param out - returns the value
    // @return Error object indicating success or failure
    template<typename T>
    Error GetRawAtCursor(size_t batch_idx, T* out);

    // ClassResult
    // The result value for CLASS format results indicating the index
    // of the class in the result vector, the value of the class (as a
    // float) and the corresponding label (if provided by the model).
    struct ClassResult {
      size_t idx;
      float value;
      std::string label;
    };

    // Get the number of class results for a batch. Returns error if
    // this result is not CLASS format.
    // @param batch_idx - return results for this entry the batch
    // @param cnt - returns the number of ClassResult entries
    // @return Error object indicating success or failure
    virtual Error GetClassCount(size_t batch_idx, size_t* cnt) const = 0;

    // Get the ClassResult result for a specific batch entry at the
    // current cursor. Use ResetCursor() to reset the cursor to the
    // beginning of the result. Returns error if this result is not
    // CLASS format.
    // @param batch_idx - return results for this entry the batch
    // @param result - returns the ClassResult value
    // @return Error object indicating success or failure
    virtual Error GetClassAtCursor(size_t batch_idx, ClassResult* result) = 0;

    // Reset cursor to beginning of result for all batch entries.
    // @return Error object indicating success or failure
    virtual Error ResetCursors() = 0;

    // Reset cursor to beginning of result for specified batch entry.
    // @param batch_idx - result cursor for this entry the batch
    // @return Error object indicating success or failure
    virtual Error ResetCursor(size_t batch_idx) = 0;
  };

  //==============
  // Options
  // Run options to be applied to all subsequent Run() invocations.
  class Options {
  public:
    virtual ~Options() { };

    // @return a new Options object with default values.
    static Options* Create();

    // @return the batch size to use for all subsequent inferences.
    virtual size_t BatchSize() const = 0;

    // Set the batch size to use for all subsequent inferences.
    // @param batch_size - the batch size
    virtual void SetBatchSize(size_t batch_size) = 0;

    // Add 'output' to the list of requested RAW results. Run() will
    // return the output's full tensor as a result.
    // @param output - the output
    // @return Error object indicating success or failure
    virtual Error AddRawResult(
      const std::shared_ptr<InferContext::Output>& output) = 0;

    // Add 'output' to the list of requested CLASS results. Run() will
    // return the highest 'k' values of 'output' as a result.
    // @param output - the output
    // @param k - return 'k' class results for the output
    // @return Error object indicating success or failure
    virtual Error AddClassResult(
      const std::shared_ptr<InferContext::Output>& output, uint64_t k) = 0;
};

public:
  // Create context that performs inference for a model.
  // @param server_url - inference server name and port
  // @param model_name - name of the model to use for inference
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  explicit InferContext(
    const std::string& server_url, const std::string& model_name,
    bool verbose = false);

  // @return the model name.
  const std::string& ModelName() const { return model_name_; }

  // @return the maximum batch size supported by the context.
  uint64_t MaxBatchSize() const { return max_batch_size_; }

  // @return the inputs of the model.
  const std::vector<std::shared_ptr<Input>>& Inputs() const {
    return inputs_;
  }

  // @return the outputs of the model.
  const std::vector<std::shared_ptr<Output>>& Outputs() const {
    return outputs_;
  }

  // Get a named input.
  // @param name - the name of the input
  // @param input - returns the Input object for 'name'
  // @return Error object indicating success or failure
  Error GetInput(
    const std::string& name, std::shared_ptr<Input>* input) const;

  // Get a named output.
  // @param name - the name of the output
  // @param input - returns the Output object for 'name'
  // @return Error object indicating success or failure
  Error GetOutput(
    const std::string& name, std::shared_ptr<Output>* output) const;

  // Set the options to use for all subsequent Run() invocations.
  // @param options - the options
  // @return Error object indicating success or failure
  Error SetRunOptions(const Options& options);

  // Send a request to the inference server to perform an inference to
  // produce a result for the outputs specified in the most recent
  // call to SetRunOptions(). The Result objects holding the output
  // values are returned in the same order as the outputs are
  // specified in the options.
  // @param results - returns Result objects holding inference results.
  // @return Error object indicating success or failure
  Error Run(std::vector<std::unique_ptr<Result>>* results);

private:
  static size_t RequestProvider(void*, size_t, size_t, void*);
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  static size_t ResponseHandler(void*, size_t, size_t, void*);

  // Copy into 'buf' up to 'size' bytes of input data. Return the
  // actual amount copied in 'input_bytes'.
  Error GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

  // Copy into the context 'size' bytes of result data from
  // 'buf'. Return the actual amount copied in 'result_<bytes'.
  Error SetNextRawResult(
    const uint8_t* buf, size_t size, size_t* result_bytes);

  // URL to POST to
  const std::string url_;

  // Model name
  const std::string model_name_;

  // If true print verbose output
  const bool verbose_;

  // Maximum batch size supported by this context.
  uint64_t max_batch_size_;

  // Did object initialize correctly?
  bool initialized_;

  // The inputs and outputs
  std::vector<std::shared_ptr<Input>> inputs_;
  std::vector<std::shared_ptr<Output>> outputs_;

  // Total size of all inputs, in bytes (must be 64-bit integer
  // because used with curl_easy_setopt).
  uint64_t total_input_byte_size_;

  // InferRequestHeader protobuf describing the request
  InferRequestHeader infer_request_;
  std::string infer_request_str_;

  // RequestStatus received in server response
  RequestStatus request_status_;

  // Buffer that accumulates the serialized InferResponseHeader at the
  // end of the body.
  std::string infer_response_buffer_;

  // Requested batch size for inference request
  uint64_t batch_size_;

  // Outputs requested for inference request
  std::vector<std::shared_ptr<Output>> requested_outputs_;

  // Results being collected for the requested outputs from inference
  // server response.
  std::vector<std::unique_ptr<Result>> requested_results_;

  // Current positions within input and output vectors when sending
  // request and receiving response.
  size_t input_pos_idx_;
  size_t result_pos_idx_;
};

//==============================================================================
// ProfileContext
//
// A ProfileContext object is used to control profiling on the
// inference server. Once created a ProfileContext object can be used
// repeatedly. For example:
//
//   ProfileContext ctx("localhost:8000");
//   ctx.StartProfile();
//   ...
//   ctx.StopProfile();
//   ...
//
// Thread-safety:
//   ProfileContext contructors are thread-safe.  StartProfiling() and
//   StopProfiling() are not thread-safe. For a given ProfileContext,
//   calls to these methods must be serialized.
//
class ProfileContext {
public:
  // Create context that controls profiling on a server.
  // @param server_url - inference server name and port
  // @param verbose - if true generate verbose output when contacting
  // the inference server
  explicit ProfileContext(
    const std::string& server_url, bool verbose = false);

  // Start profiling on the inference server
  // @return Error object indicating success or failure
  Error StartProfile();

  // Start profiling on the inference server
  // @return Error object indicating success or failure
  Error StopProfile();

private:
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  Error SendCommand(const std::string& cmd_str);

  // URL for status endpoint on inference server.
  const std::string url_;

  // If true print verbose output
  const bool verbose_;

  // RequestStatus received in server response
  RequestStatus request_status_;
};

//==============================================================================

std::ostream& operator<<(std::ostream&, const Error&);

template<typename T>
Error
InferContext::Result::GetRawAtCursor(size_t batch_idx, T* out)
{
  const uint8_t* buf;
  Error err = GetRawAtCursor(batch_idx, &buf, sizeof(T));
  if (!err.IsOk()) {
    return err;
  }

  std::copy(buf, buf + sizeof(T), reinterpret_cast<uint8_t*>(out));
  return Error::Success;
}

}}} // namespace nvidia::inferenceserver::client
