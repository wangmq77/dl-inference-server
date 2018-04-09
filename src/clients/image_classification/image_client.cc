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

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "src/core/model_config.pb.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

namespace {

enum ScaleType {
  NONE = 0,
  VGG = 1,
  INCEPTION = 2
};

void
Preprocess(
  const cv::Mat& img, ni::ModelInput::Format format,
  int img_type1, int img_type3,
  size_t img_channels, const cv::Size& img_size,
  const ScaleType scale, std::vector<uint8_t>* input_data)
{
  cv::Mat sample;
  if ((img.channels() == 3) && (img_channels == 1)) {
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  } else if ((img.channels() == 4) && (img_channels == 1)) {
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  } else if ((img.channels() == 4) && (img_channels == 3)) {
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  } else if ((img.channels() == 1) && (img_channels == 3)) {
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  } else {
    sample = img;
  }

  cv::Mat sample_resized;
  if (sample.size() != img_size) {
    cv::resize(sample, sample_resized, img_size);
  } else {
    sample_resized = sample;
  }

  cv::Mat sample_type;
  sample_resized.convertTo(
    sample_type, (img_channels == 3) ? img_type3 : img_type1);

  cv::Mat sample_final;
  if (scale == ScaleType::INCEPTION) {
    sample_final = sample_type.mul(cv::Scalar(1/128.0, 1/128.0, 1/128.0));
    sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);
  } else if (scale == ScaleType::VGG) {
    sample_final = sample_type - cv::Scalar(104, 117, 123);
  } else {
    sample_final = sample_type;
  }

  // Allocate a buffer to hold all image elements.
  size_t img_byte_size = sample_final.total() * sample_final.elemSize();
  size_t pos = 0;
  input_data->resize(img_byte_size);

  // For NHWC format Mat is already in the correct order but need to
  // handle both cases of data being contigious or not.
  if (format == ni::ModelInput::FORMAT_NHWC) {
    if (sample_final.isContinuous()) {
      memcpy(&((*input_data)[0]), sample_final.datastart, img_byte_size);
      pos = img_byte_size;
    } else {
      size_t row_byte_size = sample_final.cols * sample_final.elemSize();
      for (int r = 0; r < sample_final.rows; ++r) {
        memcpy(&((*input_data)[pos]), sample_final.ptr<uint8_t>(r), row_byte_size);
        pos += row_byte_size;
      }
    }
  } else {
    // (format == ni::ModelInput::FORMAT_NCHW)
    //
    // For CHW formats must split out each channel from the matrix and
    // order them as BBBB...GGGG...RRRR. To do this split the channels
    // of the image directly into 'image_data'. The BGR channels are
    // backed by the 'image_data' vector so that ends up with CHW
    // order of the data.
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i) {
      input_bgr_channels.emplace_back(
        img_size.height, img_size.width, img_type1, &((*input_data)[pos]));
      pos +=
        input_bgr_channels.back().total() * input_bgr_channels.back().elemSize();
    }

    cv::split(sample_final, input_bgr_channels);
  }

  if (pos != img_byte_size) {
    std::cerr
      << "unexpected total size of channels " << pos
      << ", expecting " << img_byte_size << std::endl;
    exit(1);
  }
}

void
Infer(
  nic::InferContext& ctx, const size_t batch_size, const size_t topk,
  const std::vector<uint8_t>& input_data,
  std::vector<std::unique_ptr<nic::InferContext::Result>>* results,
  const bool verbose = false)
{
  nic::Error err(ni::RequestStatusCode::SUCCESS);

  // Already verified that there is 1 input and 1 output
  const auto& input = ctx.Inputs()[0];

  // Prepare context for 'batch_size' batches.
  std::unique_ptr<nic::InferContext::Options>
    options(nic::InferContext::Options::Create());
  options->SetBatchSize(batch_size);
  options->AddClassResult(ctx.Outputs()[0], topk);
  err = ctx.SetRunOptions(*options);
  if (!err.IsOk()) {
    std::cerr << "failed initializing batch size: " << err << std::endl;
    exit(1);
  }

  // Forget any previous inputs and set input (i.e. the image) for
  // each batch
  err = input->Reset();
  if (!err.IsOk()) {
    std::cerr << "failed resetting input: " << err << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < batch_size; ++i) {
    nic::Error err = input->SetRaw(input_data);
    if (!err.IsOk()) {
      std::cerr << "failed setting input: " << err << std::endl;
      exit(1);
    }
  }

  err = ctx.Run(results);
  if (!err.IsOk()) {
    std::cerr << "failed sending infer request: " << err << std::endl;
    exit(1);
  }
}

void
Postprocess(
  const std::vector<std::unique_ptr<nic::InferContext::Result>>& results,
  const size_t batch_size, const size_t topk, bool verbose = false)
{
  const bool show_all = verbose || ((batch_size == 1) && (topk > 1));

  if (show_all) {
    std::cout << "Output probabilities:" << std::endl;
  }

  if (results.size() != 1) {
    std::cerr << "expected 1 result, got " << results.size() << std::endl;
    exit(1);
  }

  const std::unique_ptr<nic::InferContext::Result>& result = results[0];

  // For each result in the batch count the top predication. Since we
  // used that same image for every entry in the batch we expect the
  // top predication to be the same for each entry... but this code
  // doesn't assume that.
  std::vector<std::pair<size_t, std::string>> predictions;
  for (size_t b = 0; b < batch_size; ++b) {
    size_t cnt = 0;
    nic::Error err = result->GetClassCount(b, &cnt);
    if (!err.IsOk()) {
      std::cerr
        << "failed reading class count for batch "
        << b << ": " << err << std::endl;
      exit(1);
    }

    // Look at each of the classes returned in the result.
    for (size_t c = 0; c < cnt; ++c) {
      nic::InferContext::Result::ClassResult cls;
      nic::Error err = result->GetClassAtCursor(b, &cls);
      if (!err.IsOk()) {
        std::cerr
          << "failed reading class for batch "
          << b << ": " << err << std::endl;
        exit(1);
      }

      // The first class in the result is the highest probability so
      // record it as the top prediction.
      if (c == 0) {
        if (predictions.size() <= cls.idx) {
          predictions.resize(cls.idx + 1);
        }
        if (predictions[cls.idx].first == 0) {
          predictions[cls.idx].second = cls.label;
        }
        predictions[cls.idx].first++;
      }

      // Keep going if printing all the returned class results,
      // otherwise we've seen the top predication so go to the next
      // entry in the batch
      if (show_all) {
       std::cout
         << "batch " << b << ": " << cls.idx << " (\""
         << cls.label << "\") = " << cls.value << std::endl;
      } else {
        break;
      }
    }
  }

  // Print summary...
  std::cout << "Prediction totals:" << std::endl;
  for (size_t i = 0; i < predictions.size(); i++) {
    if (predictions[i].first > 0) {
      std::cout
        << "\tcnt=" << predictions[i].first << "\t(" << i << ") "
        << predictions[i].second << std::endl;
    }
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr
    << "Usage: " << argv[0] << " [options] <image filename>" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t-c <topk>" << std::endl;
  std::cerr << "\t-s <NONE,|INCEPTION|VGG>" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << std::endl;
  std::cerr
    << "For -b, the image will be replicated and sent in a batch" << std::endl
    << "        of the specified size. Default is 1." << std::endl;
  std::cerr
    << "For -c, the <topk> classes will be returned, default is 1."
    << std::endl;
  std::cerr
    << "For -s, specify the type of pre-processing scaling that" << std::endl
    << "        should be performed on the image, default is NONE." << std::endl
    << "    INCEPTION: scale each pixel RGB value to [-1.0, 1.0)." << std::endl
    << "    VGG: subtract mean BGR value (104, 117, 123) from"
    << std::endl << "         each pixel." << std::endl;
  std::cerr
    << "For -u, the default server URL is localhost:8000." << std::endl;
  std::cerr << std::endl;

  exit(1);
}

ScaleType
ParseScale(const std::string& str)
{
  if (str == "NONE") {
    return ScaleType::NONE;
  } else if (str == "INCEPTION") {
    return ScaleType::INCEPTION;
  } else if (str == "VGG") {
    return ScaleType::VGG;
  }

  std::cerr
    << "unexpected scale type \"" << str
    << "\", expecting NONE, INCEPTION or VGG" << std::endl;
  exit(1);

  return ScaleType::NONE;
}

bool
ParseType(const ni::DataType& dtype, int* type1, int* type3)
{
  if (dtype == ni::DataType::TYPE_UINT8) {
    *type1 = CV_8UC1;
    *type3 = CV_8UC3;
  } else if (dtype == ni::DataType::TYPE_INT8) {
    *type1 = CV_8SC1;
    *type3 = CV_8SC3;
  } else if (dtype == ni::DataType::TYPE_UINT16) {
    *type1 = CV_16UC1;
    *type3 = CV_16UC3;
  } else if (dtype == ni::DataType::TYPE_INT16) {
    *type1 = CV_16SC1;
    *type3 = CV_16SC3;
  } else if (dtype == ni::DataType::TYPE_INT32) {
    *type1 = CV_32SC1;
    *type3 = CV_32SC3;
  } else if (dtype == ni::DataType::TYPE_FP32) {
    *type1 = CV_32FC1;
    *type3 = CV_32FC3;
  } else if (dtype == ni::DataType::TYPE_FP64) {
    *type1 = CV_64FC1;
    *type3 = CV_64FC3;
  } else {
    return false;
  }

  return true;
}

void
ParseModel(
  const nic::InferContext& ctx, const size_t batch_size,
  size_t* c, size_t* h, size_t* w,
  ni::ModelInput::Format* format, int* type1, int* type3,
  bool verbose = false)
{
  if (ctx.Inputs().size() != 1) {
    std::cerr
      << "expecting 1 input, model \"" << ctx.ModelName() << "\" has "
      << ctx.Inputs().size() << std::endl;
    exit(1);
  }

  if (ctx.Outputs().size() != 1) {
    std::cerr
      << "expecting 1 output, model \"" << ctx.ModelName() << "\" has "
      << ctx.Outputs().size() << std::endl;
    exit(1);
  }

  const auto& input = ctx.Inputs()[0];
  const auto& output = ctx.Outputs()[0];

  if (output->DType() != ni::DataType::TYPE_FP32) {
    std::cerr
      << "expecting model output datatype to be TYPE_FP32, model \""
      << ctx.ModelName() << "\" output type is "
      << ni::DataType_Name(output->DType()) << std::endl;
    exit(1);
  }

  // Output is expected to be a vector. But allow any number of
  // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
  // }, { 10, 1, 1 } are all ok).
  size_t non_one_cnt = 0;
  for (const auto dim : output->Dims()) {
    if (dim > 1) {
      non_one_cnt++;
      if (non_one_cnt > 1) {
        std::cerr << "expecting model output to be a vector" << std::endl;
        exit(1);
      }
    }
  }

  *format = input->Format();

  int max_batch_size = ctx.MaxBatchSize();

  // Model specifying maximum batch size of 0 indicates that batching
  // is not supported and so the input tensors do not expect a "N"
  // dimension (and 'batch_size' should be 1 so that only a single
  // image instance is inferred at a time).
  if (max_batch_size == 0) {
    if (batch_size != 1) {
      std::cerr
        << "batching not supported for model \"" << ctx.ModelName() << "\""
        << std::endl;
      exit(1);
    }
  } else {
    // max_batch_size > 0
    if (batch_size > (size_t)max_batch_size) {
      std::cerr
        << "expecting batch size <= " << max_batch_size << " for model \""
        << ctx.ModelName() << "\"" << std::endl;
      exit(1);
    }
  }

  if (input->Dims().size() != 3) {
    std::cerr
      << "expecting model input to have 3 dimensions, model \""
      << ctx.ModelName() << "\" input has " << input->Dims().size()
      << std::endl;
    exit(1);
  }

  // Input must be NHWC or NCHW...
  if ((*format != ni::ModelInput::FORMAT_NCHW) &&
      (*format != ni::ModelInput::FORMAT_NHWC)) {
    std::cerr
      << "unexpected input format " << ni::ModelInput_Format_Name(*format)
      << ", expecting "
      << ni::ModelInput_Format_Name(ni::ModelInput::FORMAT_NHWC) << " or "
      << ni::ModelInput_Format_Name(ni::ModelInput::FORMAT_NCHW)
      << std::endl;
    exit(1);
  }

  if (*format == ni::ModelInput::FORMAT_NHWC) {
    *h = input->Dims()[0];
    *w = input->Dims()[1];
    *c = input->Dims()[2];
  } else if (*format == ni::ModelInput::FORMAT_NCHW) {
    *c = input->Dims()[0];
    *h = input->Dims()[1];
    *w = input->Dims()[2];
  }

  if (!ParseType(input->DType(), type1, type3)) {
    std::cerr
      << "unexpected input datatype \""
      << ni::DataType_Name(input->DType())
      << "\" for model \"" << ctx.ModelName() << std::endl;
    exit(1);
  }
}

} //namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  size_t batch_size = 1;
  size_t topk = 1;
  ScaleType scale = ScaleType::NONE;
  std::string model_name;
  std::string url("localhost:8000");

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:m:b:c:s:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      case 'c':
        topk = atoi(optarg);
        break;
      case 's':
        scale = ParseScale(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (model_name.empty()) { Usage(argv, "-m flag must be specified"); }
  if (batch_size <= 0) { Usage(argv, "batch size must be > 0"); }
  if (topk <= 0) { Usage(argv, "topk must be > 0"); }
  if (optind >= argc) { Usage(argv, "image file must be specified"); }

  std::string filename(argv[optind]);

  // Create the context for inference of the specified model. From it
  // extract and validate that the model meets the requirements for
  // image classification.
  nic::InferContext ctx(url, model_name, verbose);

  size_t c, h, w;
  ni::ModelInput::Format format;
  int type1, type3;
  ParseModel(ctx, batch_size, &c, &h, &w, &format, &type1, &type3, verbose);

  // Load the specified image.
  std::ifstream file(filename);
  std::vector<char> data;
  file >> std::noskipws;
  std::copy(
    std::istream_iterator<char>(file), std::istream_iterator<char>(),
    std::back_inserter(data));
  if (data.empty()) {
    std::cerr << "error: unable to read image file " << filename << std::endl;
    exit(1);
  }

  cv::Mat img = imdecode(cv::Mat(data), 1);
  if (img.empty()) {
    std::cerr << "error: unable to decode image " << filename << std::endl;
    exit(1);
  }

  // Pre-process the image to match input size expected by the model.
  std::vector<uint8_t> input_data;
  Preprocess(img, format, type1, type3, c, cv::Size(w, h), scale, &input_data);

  // Run inference to get output
  std::vector<std::unique_ptr<nic::InferContext::Result>> results;
  Infer(ctx, batch_size, topk, input_data, &results, verbose);

  // Post-process the result to make prediction(s)
  Postprocess(results, batch_size, topk, verbose);

  return 0;
}
