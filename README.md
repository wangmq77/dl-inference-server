# Deep Learning Inference Server Clients

The NVIDIA Inference Server provides a cloud inferencing solution
optimized for NVIDIA GPUs. The server provides an inference service
via an HTTP endpoint, allowing remote clients to request inferencing
for any model being managed by the server.

This repo contains a C++ and Python client libraries that make it easy
to communicate with an inference server. Also included is
*image_client*, an example C++ application that uses the C++ client
library to execute image classification models on the inference
server.

The inference server itself is delivered as a containerized solution
from the NVIDIA Compute Cloud. See the
[Inference Container User Guide](http://docs.nvidia.com/deeplearning/dgx/index.html)
for information on how to install and configure the inference server.

## Building

Before building you must first install some prerequisites. The
following instructions assume Ubuntu 16.04. OpenCV is used by
image_client to preprocess images before sending them to the inference
server for inferencing.

    sudo apt-get update
    sudo apt-get install build-essential libcurl3-dev libopencv-dev libopencv-core-dev software-properties-common

Protobuf3 support is required. For Ubuntu 16.04 this must be installed
from a ppa, but if you are using a more recent distribution this step
might not be necessary.

    sudo add-apt-repository ppa:maarten-fonville/protobuf
    sudo apt-get update
    sudo apt-get install protobuf-compiler libprotobuf-dev

Creating the whl file for the Python client library requires setuptools.

    pip install --no-cache-dir --upgrade setuptools

With those prerequisites installed the C++ and Python client libraries
and example image_cliet application can be built:

    make -f Makefile.clients all pip

Build artifacts are in build/.  The Python whl file is generated in
build/dist/dist/ and can be installed with a command like the following:

    pip install --no-cache-dir --upgrade build/dist/dist/inference_server-0.0.1-cp27-cp27mu-linux_x86_64.whl

## C++ API

The C++ client API exposes a class-based interface for querying server
and model status and for performing inference. The commented interface
is available at src/clients/common/request.h. The image classification
example that uses this API is available at
src/clients/image_classification/image_client.cc.

The following shows an example of the basic steps required for
inferencing (error checking not included to improve clarity, see
image_client.cc for full error checking):

```c++
// Create the context object for inferencing using the 'mnist' model.
InferContext ctx("localhost:8000", "mnist");

// Get handle to model input and output.
const InferContext::Input& input = ctx.GetInput(input_name);
const InferContext::Output& output = ctx.GetOutput(output_name);

// Set options so that subsequent inference runs are for a given batch_size
// and return a result for ‘output’. The ‘output’ result is returned as a
// classification result of the ‘k’ most probable classes.
InferContext::Options* options = InferContext::Options::Create();
options->SetBatchSize(batch_size);
options->AddClassResult(output, k);
ctx.SetRunOptions(*options);

// Provide input data for each batch.
input->Reset();
for (size_t i = 0; i < batch_size; ++i) {
  input->SetRaw(input_data[i]);
}

// Run inference and get the results. When the Run() call returns the ctx
// can be used for another inference run. Results are owned by the caller
// and can be retained as long as necessary.
std::vector<std::unique_ptr<InferContext::Result>> results;
ctx.Run(&results);

// For each entry in the batch print the top prediction.
for (size_t i = 0; i < batch_size; ++i) {
  InferContext::Result::ClassResult cls;
  results[0]->GetClassAtCursor(i, &cls);
  std::cout << "batch " << i << ": " << cls.label << std::endl;
}
```

## Python API

The Python client API provides similar capabilities as the C++
API. The commented interface for StatusContext and InferContext
classes is available at src/clients/python/__init__.py.

The following shows an example of the basic steps required for
inferencing (error checking not included to improve clarity):

```python
from inference_server.api import *

# Create input with random data
input_list = list()
for b in range(batch_size):
    in = np.random.randint(size=input_size, dtype=input_dtype)
    input_list.append(in)

# Run inferencing and get the top-3 classes
ctx = InferContext("localhost:8000", "mnist")
results = ctx.run(
    { "data" : input_list },
    { "prob" : (InferContext.ResultFormat.CLASS, 3) },
    batch_size)


# Print results
for (result_name, result_val) in iteritems(results):
    for b in range(batch_size):
        print("output {}, batch {}: {}".format(result_name, b, result_val[b]))
```
