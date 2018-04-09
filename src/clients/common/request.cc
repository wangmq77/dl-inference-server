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

#include "src/clients/common/request.h"

#include <iostream>
#include <curl/curl.h>
#include <google/protobuf/text_format.h>
#include "src/core/constants.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================

// Global initialization for libcurl. Libcurl requires global
// initialization before any other threads are created and before any
// curl methods are used. The curl_global static object is used to
// perform this initialization.
class CurlGlobal {
public:
  CurlGlobal();
  ~CurlGlobal();

  const Error& Status() const { return err_; }

private:
  Error err_;
};

CurlGlobal::CurlGlobal()
  : err_(RequestStatusCode::SUCCESS)
{
  if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
    err_ = Error(RequestStatusCode::INTERNAL, "global initialization failed");
  }
}

CurlGlobal::~CurlGlobal()
{
  curl_global_cleanup();
}

static CurlGlobal curl_global;

//==============================================================================

const Error Error::Success(RequestStatusCode::SUCCESS);

Error::Error(RequestStatusCode code, const std::string& msg)
  : code_(code), msg_(msg)
{
}

Error::Error(RequestStatusCode code)
  : code_(code)
{
}

Error::Error(const RequestStatus& status)
  : Error(status.code(), status.msg())
{
  server_id_ = status.server_id();
}

std::ostream&
operator<<(std::ostream& out, const Error& err)
{
  out
    << "[" << err.server_id_ << "] "
    << RequestStatusCode_Name(err.code_);
  if (!err.msg_.empty()) {
    out << " - " << err.msg_;
  }
  return out;
}

//==============================================================================
ServerStatusContext::ServerStatusContext(
  const std::string& server_url, bool verbose)
  : url_(server_url + "/" + kStatusRESTEndpoint), verbose_(verbose)
{
}

ServerStatusContext::ServerStatusContext(
  const std::string& server_url, const std::string& model_name, bool verbose)
  : url_(server_url + "/" + kStatusRESTEndpoint + "/" + model_name),
    verbose_(verbose)
{
}

Error
ServerStatusContext::GetServerStatus(ServerStatus* server_status)
{
  server_status->Clear();
  request_status_.Clear();
  response_.clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return
      Error(RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Want binary representation of the status.
  std::string full_url = url_ + "?format=binary";
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  // response data handled by ResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    return
      Error(
        RequestStatusCode::INTERNAL, "HTTP client failed: " +
        std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("status request did not return status");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    return Error(request_status_);
  }

  // Parse the response as a ModelConfigList...
  if (!server_status->ParseFromString(response_)) {
    return Error(RequestStatusCode::INTERNAL, "failed to parse server status");
  }

  if (verbose_) {
    std::cout << server_status->DebugString() << std::endl;
  }

  return Error::Success;
}

size_t
ServerStatusContext::ResponseHeaderHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  ServerStatusContext* ctx = reinterpret_cast<ServerStatusContext*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
          hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

size_t
ServerStatusContext::ResponseHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  ServerStatusContext* ctx = reinterpret_cast<ServerStatusContext*>(userp);
  uint8_t* buf = reinterpret_cast<uint8_t*>(contents);
  size_t result_bytes = size * nmemb;
  std::copy(buf, buf + result_bytes, std::back_inserter(ctx->response_));
  return result_bytes;
}

//==============================================================================

class OptionsImpl : public InferContext::Options {
public:
  OptionsImpl();
  ~OptionsImpl() = default;

  size_t BatchSize() const override { return batch_size_; }
  void SetBatchSize(size_t batch_size) override { batch_size_ = batch_size; }

  Error AddRawResult(
    const std::shared_ptr<InferContext::Output>& output) override;
  Error AddClassResult(
    const std::shared_ptr<InferContext::Output>& output, uint64_t k) override;

  // Options for an output
  struct OutputOptions {
    OutputOptions(InferContext::Result::ResultFormat f, uint64_t n=0)
      : result_format(f), u64(n) { }
    InferContext::Result::ResultFormat result_format;
    uint64_t u64;
  };

  using OutputOptionsPair =
    std::pair<std::shared_ptr<InferContext::Output>, OutputOptions>;

  const std::vector<OutputOptionsPair>& Outputs() const { return outputs_; }

private:
  size_t batch_size_;
  std::vector<OutputOptionsPair> outputs_;
};

OptionsImpl::OptionsImpl()
  : batch_size_(0)
{
}

Error
OptionsImpl::AddRawResult(const std::shared_ptr<InferContext::Output>& output)
{
  outputs_.emplace_back(
    std::make_pair(
      output, OutputOptions(InferContext::Result::ResultFormat::RAW)));
  return Error::Success;
}

Error
OptionsImpl::AddClassResult(
  const std::shared_ptr<InferContext::Output>& output, uint64_t k)
{
  outputs_.emplace_back(
    std::make_pair(
      output, OutputOptions(InferContext::Result::ResultFormat::CLASS, k)));
  return Error::Success;
}

InferContext::Options*
InferContext::Options::Create()
{
  return new OptionsImpl();
}

//==============================================================================

class InputImpl : public InferContext::Input {
public:
  InputImpl(const ModelInput& mio);
  ~InputImpl() = default;

  const std::string& Name() const override { return mio_.name(); }
  size_t ByteSize() const override { return byte_size_; }
  DataType DType() const override { return mio_.data_type(); }
  ModelInput::Format Format() const override { return mio_.format(); }
  const DimsList& Dims() const override { return mio_.dims(); }

  void SetBatchSize(size_t batch_size) { batch_size_ = batch_size; }

  Error Reset() override;
  Error SetRaw(const std::vector<uint8_t>& input) override;
  Error SetRaw(const uint8_t* input, size_t input_byte_size) override;

  // Copy into 'buf' up to 'size' bytes of this input's data. Return
  // the actual amount copied in 'input_bytes'.
  Error GetNext(uint8_t* buf, size_t size, size_t* input_bytes);

  // Prepare to send this input as part of a request.
  Error PrepareForRequest();

private:
  const ModelInput mio_;
  const size_t byte_size_;
  size_t batch_size_;
  std::vector<const uint8_t*> bufs_;
  size_t bufs_idx_, buf_pos_;
};

InputImpl::InputImpl(const ModelInput& mio)
  : mio_(mio), byte_size_(GetSize(mio)),
    batch_size_(0), bufs_idx_(0), buf_pos_(0)
{
}

Error
InputImpl::SetRaw(const uint8_t* input, size_t input_byte_size)
{
  if (input_byte_size != byte_size_) {
    bufs_.clear();
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "invalid size " + std::to_string(input_byte_size) +
        " bytes for input '" + Name() + "', expects " +
        std::to_string(byte_size_) + " bytes");
  }

  if (bufs_.size() >= batch_size_) {
    bufs_.clear();
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "expecting " + std::to_string(batch_size_) +
        " invocations of SetRaw for input '" + Name() +
        "', one per batch entry");
  }

  bufs_.push_back(input);
  return Error::Success;
}

Error
InputImpl::SetRaw(const std::vector<uint8_t>& input)
{
  return SetRaw(&input[0], input.size());
}

Error
InputImpl::GetNext(uint8_t* buf, size_t size, size_t* input_bytes)
{
  size_t total_size = 0;

  while ((bufs_idx_ < bufs_.size()) && (size > 0)) {
    const size_t csz = std::min(byte_size_ - buf_pos_, size);
    if (csz > 0) {
      const uint8_t* input_ptr = bufs_[bufs_idx_] + buf_pos_;
      std::copy(input_ptr, input_ptr + csz, buf);
      buf_pos_ += csz;
      buf += csz;
      size -= csz;
      total_size += csz;
    }

    if (buf_pos_ == byte_size_) {
      bufs_idx_++;
      buf_pos_ = 0;
    }
  }

  *input_bytes = total_size;
  return Error::Success;
}

Error
InputImpl::Reset()
{
  bufs_.clear();
  bufs_idx_ = 0;
  buf_pos_ = 0;
  return Error::Success;
}

Error
InputImpl::PrepareForRequest()
{
  if (bufs_.size() != batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "expecting " + std::to_string(batch_size_) +
        " invocations of SetRaw for input '" + Name() +
        "', have " + std::to_string(bufs_.size()));
  }

  // Reset position so request sends entire input.
  bufs_idx_ = 0;
  buf_pos_ = 0;
  return Error::Success;
}

//==============================================================================

class OutputImpl : public InferContext::Output {
public:
  OutputImpl(const ModelOutput& mio);
  ~OutputImpl() = default;

  const std::string& Name() const override { return mio_.name(); }
  size_t ByteSize() const override { return byte_size_; }
  DataType DType() const override { return mio_.data_type(); }
  const DimsList& Dims() const override { return mio_.dims(); }

  InferContext::Result::ResultFormat ResultFormat() const {
    return result_format_;
  }
  void SetResultFormat(InferContext::Result::ResultFormat result_format) {
    result_format_ = result_format;
  }

  private:
  const ModelOutput mio_;
  const size_t byte_size_;
  InferContext::Result::ResultFormat result_format_;
};

OutputImpl::OutputImpl(const ModelOutput& mio)
  : mio_(mio), byte_size_(GetSize(mio)),
    result_format_(InferContext::Result::ResultFormat::RAW)
{
}

//==============================================================================

class ResultImpl : public InferContext::Result {
public:
  ResultImpl(
    const std::shared_ptr<InferContext::Output>& output, uint64_t batch_size,
    InferContext::Result::ResultFormat result_format);
  ~ResultImpl() = default;

  const std::shared_ptr<InferContext::Output> GetOutput() const override {
    return output_;
  }

  Error GetRaw(
    size_t batch_idx, const std::vector<uint8_t>** buf) const override;
  Error GetRawAtCursor(
    size_t batch_idx, const uint8_t** buf, size_t adv_byte_size) override;
  Error GetClassCount(size_t batch_idx, size_t* cnt) const override;
  Error GetClassAtCursor(size_t batch_idx, ClassResult* result) override;
  Error ResetCursors() override;
  Error ResetCursor(size_t batch_idx) override;

  // Get the result format for this result.
  InferContext::Result::ResultFormat ResultFormat() const {
    return result_format_;
  }

  // Set results for a CLASS format result.
  void SetClassResult(const InferResponseHeader::Output& result) {
    class_result_ = result;
  }

  // For RAW format result, copy into the output up to 'size' bytes of
  // output data from 'buf'. Return the actual amount copied in
  // 'result_bytes'.
  Error SetNextRawResult(
    const uint8_t* buf, size_t size, size_t* result_bytes);

private:
  const std::shared_ptr<InferContext::Output> output_;
  const size_t byte_size_;
  const size_t batch_size_;
  const InferContext::Result::ResultFormat result_format_;

  std::vector<std::vector<uint8_t>> bufs_;
  size_t bufs_idx_;
  std::vector<size_t> bufs_pos_;

  InferResponseHeader::Output class_result_;
  std::vector<size_t> class_pos_;
};

ResultImpl::ResultImpl(
  const std::shared_ptr<InferContext::Output>& output, uint64_t batch_size,
  InferContext::Result::ResultFormat result_format)
  : output_(output), byte_size_(output->ByteSize()),
    batch_size_(batch_size), result_format_(result_format),
    bufs_(batch_size), bufs_idx_(0), bufs_pos_(batch_size),
    class_pos_(batch_size)
{
}

Error
ResultImpl::GetRaw(
  size_t batch_idx, const std::vector<uint8_t>** buf) const
{
  if (result_format_ != InferContext::Result::ResultFormat::RAW) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "raw result not available for non-RAW output '" +
        output_->Name() + "'");
  }

  if (batch_idx >= batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  *buf = &bufs_[batch_idx];
  return Error::Success;
}

Error
ResultImpl::GetRawAtCursor(
  size_t batch_idx, const uint8_t** buf, size_t adv_byte_size)
{
  if (result_format_ != InferContext::Result::ResultFormat::RAW) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "raw result not available for non-RAW output '" +
        output_->Name() + "'");
  }

  if (batch_idx >= batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  if ((bufs_pos_[batch_idx] + adv_byte_size) > byte_size_) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "attempt to read beyond end of result for output output '" +
        output_->Name() + "'");
  }

  *buf = &bufs_[batch_idx][bufs_pos_[batch_idx]];
  bufs_pos_[batch_idx] += adv_byte_size;
  return Error::Success;
}

Error
ResultImpl::GetClassCount(size_t batch_idx, size_t* cnt) const
{
  if (result_format_ != InferContext::Result::ResultFormat::CLASS) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "class result not available for non-CLASS output '" +
        output_->Name() + "'");
  }

  // Number of classifications should equal expected batch size but
  // check both to be careful and to protext class_pos_ accesses.
  if ((batch_idx >= (size_t)class_result_.batch_classes().size()) ||
      (batch_idx >= batch_size_)) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  const InferResponseHeader::Output::Classes& classes =
    class_result_.batch_classes(batch_idx);

  *cnt = classes.cls().size();
  return Error::Success;
}

Error
ResultImpl::GetClassAtCursor(
  size_t batch_idx, InferContext::Result::ClassResult* result)
{
  if (result_format_ != InferContext::Result::ResultFormat::CLASS) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "class result not available for non-CLASS output '" +
        output_->Name() + "'");
  }

  // Number of classifications should equal expected batch size but
  // check both to be careful and to protext class_pos_ accesses.
  if ((batch_idx >= (size_t)class_result_.batch_classes().size()) ||
      (batch_idx >= batch_size_)) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  const InferResponseHeader::Output::Classes& classes =
    class_result_.batch_classes(batch_idx);

  if (class_pos_[batch_idx] >= (size_t)classes.cls().size()) {
    return
      Error(
        RequestStatusCode::UNSUPPORTED,
        "attempt to read beyond end of result for output output '" +
        output_->Name() + "'");
  }

  const InferResponseHeader::Output::Class& cls =
    classes.cls(class_pos_[batch_idx]);

  result->idx = cls.idx();
  result->value = cls.value();
  result->label = cls.label();

  class_pos_[batch_idx]++;
  return Error::Success;
}

Error
ResultImpl::ResetCursors()
{
  std::fill(bufs_pos_.begin(), bufs_pos_.end(), 0);
  std::fill(class_pos_.begin(), class_pos_.end(), 0);
  return Error::Success;
}

Error
ResultImpl::ResetCursor(size_t batch_idx)
{
  if (batch_idx >= batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "unexpected batch entry " + std::to_string(batch_idx) +
        "requested for output '" + output_->Name() +
        "', batch size is " + std::to_string(batch_size_));
  }

  bufs_pos_[batch_idx] = 0;
  class_pos_[batch_idx] = 0;
  return Error::Success;
}

Error
ResultImpl::SetNextRawResult(
  const uint8_t* buf, size_t size, size_t* result_bytes)
{
  size_t total_size = 0;

  while ((bufs_idx_ < bufs_.size()) && (size > 0)) {
    const size_t csz = std::min(byte_size_ - bufs_pos_[bufs_idx_], size);
    if (csz > 0) {
      std::copy(buf, buf + csz, std::back_inserter(bufs_[bufs_idx_]));
      bufs_pos_[bufs_idx_] += csz;
      buf += csz;
      size -= csz;
      total_size += csz;
    }

    if (bufs_pos_[bufs_idx_] == byte_size_) {
      bufs_idx_++;
    }
  }

  *result_bytes = total_size;
  return Error::Success;
}

//==============================================================================

InferContext::InferContext(
  const std::string& server_url, const std::string& model_name, bool verbose)
  : url_(server_url + "/" + kInferRESTEndpoint + "/" + model_name),
    model_name_(model_name), verbose_(verbose), initialized_(false),
    total_input_byte_size_(0), batch_size_(0),
    input_pos_idx_(0), result_pos_idx_(0)
{
  // Get status of the model and create the inputs and outputs.
  ServerStatusContext ctx(server_url, model_name, verbose);

  ServerStatus server_status;
  Error err = ctx.GetServerStatus(&server_status);
  if (err.IsOk()) {
    const auto& itr = server_status.model_status().find(model_name);
    if (itr != server_status.model_status().end()) {
      const ModelConfig& model_info = itr->second.config();

      max_batch_size_ =
        static_cast<uint64_t>(std::max(0, model_info.max_batch_size()));

      // Create inputs and outputs
      for (const auto& io : model_info.input()) {
        inputs_.emplace_back(std::make_shared<InputImpl>(io));
      }
      for (const auto& io : model_info.output()) {
        outputs_.emplace_back(std::make_shared<OutputImpl>(io));
      }

      initialized_ = true;
    }
  }
}

Error
InferContext::GetInput(
  const std::string& name, std::shared_ptr<Input>* input) const
{
  if (!initialized_) {
    return
      Error(
        RequestStatusCode::INTERNAL,
        "failed initializing inference for \"" + model_name_ + "\"");
  }

  for (const auto& io : inputs_) {
    if (io->Name() == name) {
      *input = io;
      return Error::Success;
    }
  }

  return
    Error(
      RequestStatusCode::INVALID_ARG,
      "unknown input '" + name + "' for '" + model_name_ + "'");
}

Error
InferContext::GetOutput(
  const std::string& name, std::shared_ptr<Output>* output) const
{
  if (!initialized_) {
    return
      Error(
        RequestStatusCode::INTERNAL,
        "failed initializing inference for \"" + model_name_ + "\"");
  }

  for (const auto& io : outputs_) {
    if (io->Name() == name) {
      *output = io;
      return Error::Success;
    }
  }

  return
    Error(
      RequestStatusCode::INVALID_ARG,
      "unknown output '" + name + "' for '" + model_name_ + "'");
}

Error
InferContext::SetRunOptions(const InferContext::Options& boptions)
{
  const OptionsImpl& options = reinterpret_cast<const OptionsImpl&>(boptions);

  if (options.BatchSize() > max_batch_size_) {
    return
      Error(
        RequestStatusCode::INVALID_ARG,
        "run batch size " + std::to_string(options.BatchSize()) +
        " exceeds maximum batch size " + std::to_string(max_batch_size_) +
        " allowed for model '" + model_name_ + "'");
  }

  batch_size_ = options.BatchSize();
  total_input_byte_size_ = 0;

  // Create the InferRequestHeader protobuf. This protobuf will be
  // used for all subsequent requests.
  infer_request_.Clear();
  infer_request_str_.clear();

  infer_request_.set_batch_size(batch_size_);

  for (const auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->SetBatchSize(batch_size_);
    total_input_byte_size_ += io->ByteSize() * batch_size_;

    auto rinput = infer_request_.add_input();
    rinput->set_name(io->Name());
    rinput->set_byte_size(io->ByteSize());
  }

  requested_outputs_.clear();
  requested_results_.clear();

  for (const auto& p : options.Outputs()) {
    const std::shared_ptr<Output>& output = p.first;
    const OptionsImpl::OutputOptions& ooptions = p.second;

    reinterpret_cast<OutputImpl*>(output.get())->
      SetResultFormat(ooptions.result_format);
    requested_outputs_.emplace_back(output);

    auto routput = infer_request_.add_output();
    routput->set_name(output->Name());
    routput->set_byte_size(output->ByteSize());
    if (ooptions.result_format == Result::ResultFormat::CLASS) {
      routput->mutable_cls()->set_count(ooptions.u64);
    }
  }

  infer_request_str_ =
    std::string(kInferRequestHTTPHeader) + ":" +
    infer_request_.ShortDebugString();

  return Error::Success;
}

Error
InferContext::Run(std::vector<std::unique_ptr<Result>>* results)
{
  results->clear();

  if (!initialized_) {
    return
      Error(
        RequestStatusCode::INTERNAL,
        "failed initializing inference for \"" + model_name_ + "\"");
  }

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return
      Error(RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  infer_response_buffer_.clear();

  // Reset all the position indicators so that we send all inputs
  // correctly.
  request_status_.Clear();
  input_pos_idx_ = 0;
  result_pos_idx_ = 0;

  for (auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->PrepareForRequest();
  }

  // Initialize the results vector to collect the requested results.
  requested_results_.clear();
  for (const auto& io : requested_outputs_) {
    std::unique_ptr<ResultImpl>
      rp(
        new ResultImpl(
          io, batch_size_,
          reinterpret_cast<OutputImpl*>(io.get())->ResultFormat()));
    requested_results_.emplace_back(std::move(rp));
  }

  curl_easy_setopt(curl, CURLOPT_URL, url_.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // request data provided by RequestProvider()
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, RequestProvider);
  curl_easy_setopt(curl, CURLOPT_READDATA, this);

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  // response data handled by ResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);

  // set the expected POST size. If you want to POST large amounts of
  // data, consider CURLOPT_POSTFIELDSIZE_LARGE
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, total_input_byte_size_);

  // Headers to specify input and output tensors
  struct curl_slist *list = NULL;
  list = curl_slist_append(list, "Expect:");
  list = curl_slist_append(list, "Content-Type: application/octet-stream");
  list = curl_slist_append(list, infer_request_str_.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_slist_free_all(list);
    curl_easy_cleanup(curl);
    requested_results_.clear();
    return
      Error(
        RequestStatusCode::INTERNAL, "HTTP client failed: " +
        std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(list);
  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("infer request did not return status");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    requested_results_.clear();
    return Error(request_status_);
  }

  // The infer response header should be available...
  if (infer_response_buffer_.empty()) {
    requested_results_.clear();
    return
      Error(
        RequestStatusCode::INTERNAL,
        "infer request did not return result header");
  }

  InferResponseHeader infer_response;
  infer_response.ParseFromString(infer_response_buffer_);

  // At this point, the RAW requested results have their result values
  // set. Now need to initialize non-RAW results.
  for (auto& rr : requested_results_) {
    ResultImpl* r = reinterpret_cast<ResultImpl*>(rr.get());
    switch (r->ResultFormat()) {
      case Result::ResultFormat::RAW:
        r->ResetCursors();
        break;

      case Result::ResultFormat::CLASS: {
        for (const auto& ir : infer_response.output()) {
          if (ir.name() == r->GetOutput()->Name()) {
            r->SetClassResult(ir);
            break;
          }
        }
        break;
      }
    }
  }

  results->swap(requested_results_);

  return Error::Success;
}

size_t
InferContext::RequestProvider(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  InferContext* ctx = reinterpret_cast<InferContext*>(userp);

  size_t input_bytes = 0;
  Error err =
    ctx->GetNextInput(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &input_bytes);
  if (!err.IsOk()) {
    std::cerr << "RequestProvider: " << err << std::endl;
    return CURL_READFUNC_ABORT;
  }

  return input_bytes;
}

size_t
InferContext::ResponseHeaderHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  InferContext* ctx = reinterpret_cast<InferContext*>(userp);
  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      if (!google::protobuf::TextFormat::ParseFromString(
          hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

size_t
InferContext::ResponseHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  InferContext* ctx = reinterpret_cast<InferContext*>(userp);
  size_t result_bytes = 0;

  Error err =
    ctx->SetNextRawResult(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &result_bytes);
  if (!err.IsOk()) {
    std::cerr << "ResponseHandler: " << err << std::endl;
    return 0;
  }

  return result_bytes;
}

Error
InferContext::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  while ((size > 0) && (input_pos_idx_ < inputs_.size())) {
    InputImpl* io = reinterpret_cast<InputImpl*>(inputs_[input_pos_idx_].get());
    size_t ib = 0;
    Error err = io->GetNext(buf, size, &ib);
    if (!err.IsOk()) {
      return err;
    }

    // If input didn't have any more bytes then move to the next.
    if (ib == 0) {
      input_pos_idx_++;
    } else {
      *input_bytes += ib;
      size -= ib;
      buf += ib;
    }
  }

  return Error::Success;
}

Error
InferContext::SetNextRawResult(
  const uint8_t* buf, size_t size, size_t* result_bytes)
{
  *result_bytes = 0;

  while ((size > 0) && (result_pos_idx_ < requested_results_.size())) {
    ResultImpl* io =
      reinterpret_cast<ResultImpl*>(requested_results_[result_pos_idx_].get());
    size_t ob = 0;

    // Only try to read raw result for RAW
    if (io->ResultFormat() == Result::ResultFormat::RAW) {
      Error err = io->SetNextRawResult(buf, size, &ob);
      if (!err.IsOk()) {
        return err;
      }
    }

    // If output couldn't accept any more bytes then move to the next.
    if (ob == 0) {
      result_pos_idx_++;
    } else {
      *result_bytes += ob;
      size -= ob;
      buf += ob;
    }
  }

  // If there is any bytes left then they belong to the response
  // header, since all the RAW results have been filled.
  if (size > 0) {
    infer_response_buffer_.append(reinterpret_cast<const char*>(buf), size);
    *result_bytes += size;
  }

  return Error::Success;
}

//==============================================================================
ProfileContext::ProfileContext(
  const std::string& server_url, bool verbose)
  : url_(server_url + "/" + kProfileRESTEndpoint), verbose_(verbose)
{
}

Error
ProfileContext::SendCommand(const std::string& cmd_str)
{
  request_status_.Clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return
      Error(RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Want binary representation of the status.
  std::string full_url = url_ + "?cmd=" + cmd_str;
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    return
      Error(
        RequestStatusCode::INTERNAL, "HTTP client failed: " +
        std::string(curl_easy_strerror(res)));
  }

  // Must use 64-bit integer with curl_easy_getinfo
  int64_t http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("profile request did not return status");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    return Error(request_status_);
  }

  return Error::Success;
}

Error
ProfileContext::StartProfile()
{
  return SendCommand("start");
}

Error
ProfileContext::StopProfile()
{
  return SendCommand("stop");
}

size_t
ProfileContext::ResponseHeaderHandler(
  void* contents, size_t size, size_t nmemb, void* userp)
{
  ProfileContext* ctx = reinterpret_cast<ProfileContext*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
          hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

}}} // namespace nvidia::inferenceserver::client
