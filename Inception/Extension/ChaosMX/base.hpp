#pragma once

#include "dnn/tensor.hpp"
#include "dnn/net.hpp"
#include "dnn/optimizer.hpp"

#pragma warning (push, 0)
#include <mxnet/c_api.h>
#include <mxnet/c_predict_api.h>
#pragma warning (pop)