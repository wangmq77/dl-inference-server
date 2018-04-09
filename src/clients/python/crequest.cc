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

#include "src/clients/python/crequest.h"

#include <iostream>

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

//==============================================================================
nic::Error*
ErrorNew(const char* msg)
{
  return new nic::Error(ni::RequestStatusCode::INTERNAL, std::string(msg));
}

void
ErrorDelete(nic::Error* ctx)
{
  delete ctx;
}

bool
ErrorIsOk(nic::Error* ctx)
{
  return ctx->IsOk();
}

const char*
ErrorMessage(nic::Error* ctx)
{
  return ctx->Message().c_str();
}

const char*
ErrorServerId(nic::Error* ctx)
{
  return ctx->ServerId().c_str();
}

//==============================================================================
struct ServerStatusContextCtx {
  std::unique_ptr<nic::ServerStatusContext> ctx;
  std::string status_buf;
};

nic::Error*
ServerStatusContextNew(
  ServerStatusContextCtx** ctx, const char* url,
  const char* model_name, bool verbose)
{
  ServerStatusContextCtx* lctx = new ServerStatusContextCtx;
  if (model_name == nullptr) {
    lctx->ctx.reset(new nic::ServerStatusContext(std::string(url), verbose));
  } else {
    lctx->ctx.reset(
      new nic::ServerStatusContext(
        std::string(url), std::string(model_name), verbose));
  }

  *ctx = lctx;
  return nullptr;
}

void
ServerStatusContextDelete(ServerStatusContextCtx* ctx)
{
  delete ctx;
}

nic::Error*
ServerStatusContextGetServerStatus(
  ServerStatusContextCtx* ctx, char** status, uint32_t* status_len)
{
  ctx->status_buf.clear();

  ni::ServerStatus server_status;
  nic::Error err = ctx->ctx->GetServerStatus(&server_status);
  if (err.IsOk()) {
    if (server_status.SerializeToString(&ctx->status_buf)) {
      *status = &ctx->status_buf[0];
      *status_len = ctx->status_buf.size();
    } else {
      err =
        nic::Error(
          ni::RequestStatusCode::INTERNAL, "failed to parse server status");
    }
  }

  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

//==============================================================================
struct InferContextCtx {
  std::unique_ptr<nic::InferContext> ctx;
  std::vector<std::unique_ptr<nic::InferContext::Result>> results;
};

nic::Error*
InferContextNew(
  InferContextCtx** ctx, const char* url, const char* model_name, bool verbose)
{
  InferContextCtx* lctx = new InferContextCtx;
  lctx->ctx.reset(
    new nic::InferContext(std::string(url), std::string(model_name), verbose));

  *ctx = lctx;
  return nullptr;
}

void
InferContextDelete(InferContextCtx* ctx)
{
  delete ctx;
}

nic::Error*
InferContextSetOptions(
  InferContextCtx* ctx, nic::InferContext::Options* options)
{
  nic::Error err = ctx->ctx->SetRunOptions(*options);
  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

nic::Error*
InferContextRun(InferContextCtx* ctx)
{
  ctx->results.clear();
  nic::Error err = ctx->ctx->Run(&ctx->results);
  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

//==============================================================================
nic::Error*
InferContextOptionsNew(
  nic::InferContext::Options** ctx, uint64_t batch_size)
{
  *ctx = nic::InferContext::Options::Create();
  (*ctx)->SetBatchSize(batch_size);
  return nullptr;
}

void
InferContextOptionsDelete(nic::InferContext::Options* ctx)
{
  delete ctx;
}

nic::Error*
InferContextOptionsAddRaw(
  InferContextCtx* infer_ctx, nic::InferContext::Options* ctx,
  const char* output_name)
{
  std::shared_ptr<nic::InferContext::Output> output;
  nic::Error err = infer_ctx->ctx->GetOutput(std::string(output_name), &output);
  if (err.IsOk()) {
    err = ctx->AddRawResult(output);
  }

  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

nic::Error*
InferContextOptionsAddClass(
  InferContextCtx* infer_ctx, nic::InferContext::Options* ctx,
  const char* output_name, uint64_t count)
{
  std::shared_ptr<nic::InferContext::Output> output;
  nic::Error err = infer_ctx->ctx->GetOutput(std::string(output_name), &output);
  if (err.IsOk()) {
    err = ctx->AddClassResult(output, count);
  }

  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

//==============================================================================
struct InferContextInputCtx {
  std::shared_ptr<nic::InferContext::Input> input;
};

nic::Error*
InferContextInputNew(
  InferContextInputCtx** ctx, InferContextCtx* infer_ctx,
  const char* input_name)
{
  InferContextInputCtx* lctx = new InferContextInputCtx;
  nic::Error err =
    infer_ctx->ctx->GetInput(std::string(input_name), &lctx->input);

  *ctx = lctx;
  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

void
InferContextInputDelete(InferContextInputCtx* ctx)
{
  delete ctx;
}

nic::Error*
InferContextInputSetRaw(
  InferContextInputCtx* ctx, const void* data, uint64_t byte_size)
{
  nic::Error err =
    ctx->input->SetRaw(reinterpret_cast<const uint8_t*>(data), byte_size);
  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

//==============================================================================
struct InferContextResultCtx {
  std::unique_ptr<nic::InferContext::Result> result;
  nic::InferContext::Result::ClassResult cr;
};

nic::Error*
InferContextResultNew(
  InferContextResultCtx** ctx, InferContextCtx* infer_ctx,
  const char* result_name)
{
  InferContextResultCtx* lctx = new InferContextResultCtx;
  for (auto& r : infer_ctx->results) {
    if ((r != nullptr) && (r->GetOutput()->Name() == result_name)) {
      lctx->result.swap(r);
    }
  }

  if (lctx->result == nullptr) {
    return
      new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "unable to find result for output '" + std::string(result_name) + "'");
  }

  *ctx = lctx;
  return nullptr;
}

void
InferContextResultDelete(InferContextResultCtx* ctx)
{
  delete ctx;
}

nic::Error*
InferContextResultDataType(InferContextResultCtx* ctx, uint32_t* dtype)
{
  if (ctx->result == nullptr) {
    return
      new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "datatype not available for empty result");
  }

  ni::DataType data_type = ctx->result->GetOutput()->DType();
  *dtype = static_cast<uint32_t>(data_type);

  return nullptr;
}

nic::Error*
InferContextResultNextRaw(
  InferContextResultCtx* ctx, size_t batch_idx,
  const char** val, uint64_t* val_len)
{
  if (ctx->result == nullptr) {
    return
      new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "no raw result available for empty result");
  }

  const std::vector<uint8_t>* buf;
  nic::Error err = ctx->result->GetRaw(batch_idx, &buf);
  if (err.IsOk()) {
    *val = reinterpret_cast<const char*>(&((*buf)[0]));
    *val_len = buf->size();
  }

  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

nic::Error*
InferContextResultClassCount(
  InferContextResultCtx* ctx, size_t batch_idx, uint64_t* count)
{
  if (ctx->result == nullptr) {
    return
      new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "no classes available for empty result");
  }

  nic::Error err = ctx->result->GetClassCount(batch_idx, count);
  return (err.IsOk()) ? nullptr : new nic::Error(err);
}

nic::Error*
InferContextResultNextClass(
  InferContextResultCtx* ctx, size_t batch_idx,
  uint64_t* idx, float* prob, const char** label)
{
  if (ctx->result == nullptr) {
    return
      new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "no classes available for empty result");
  }

  nic::Error err = ctx->result->GetClassAtCursor(batch_idx, &ctx->cr);
  if (err.IsOk()) {
    auto& cr = ctx->cr;
    *idx = cr.idx;
    *prob = cr.value;
    *label = cr.label.c_str();
  }

  return (err.IsOk()) ? nullptr : new nic::Error(err);
}
