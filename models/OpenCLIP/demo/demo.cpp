#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "memory.h"
#include "bmruntime_interface.h"
#include <getopt.h>
#include <stdio.h>
#include <inttypes.h>
#include <random>
#include <numeric>

namespace py = pybind11;

float bfloat16_to_float32(uint16_t value)
{
    union
    {
        unsigned int u;
        float f;
    } tmp;
    tmp.u = value << 16;
    return tmp.f;
}

uint16_t float32_to_bfloat16(float value)
{
    union
    {
        unsigned int u;
        float f;
    } tmp;
    tmp.f = value;
    return tmp.u >> 16;
}

class OpenClip{
public:
  void init(std::string model_path);
  void deinit();
  py::list forward(std::vector<int> &input, std::vector<int> &mask, std::vector<float> &pixel_values);
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);

private:
  bm_handle_t bm_handle = 0;
  void *p_bmrt;
  const bm_net_info_t *net;
  int batch_size;
};

void OpenClip::net_launch(const bm_net_info_t *net, int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void OpenClip::init(std::string model_path) {
  // request bm_handle
  bm_status_t status = bm_dev_request(&bm_handle, 0);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);

  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);

  // net infos
  net = bmrt_get_network_info(p_bmrt, "OpenClip");
  batch_size = net->stages[0].input_shapes[0].dims[0];
}

void OpenClip::deinit() {
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

py::list OpenClip::forward(std::vector<int> &input, std::vector<int> &mask, 
                                    std::vector<float> &pixel_values) {
  // forward
  auto &in0_mem = net->stages[0].input_mems[0];
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &out_mem = net->stages[0].output_mems[0];

  bm_memcpy_s2d(bm_handle, in0_mem, (void *)input.data());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)mask.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)pixel_values.data());

  net_launch(net);

  std::vector<float> data(batch_size, 0);
  py::list out_list;
  bm_memcpy_d2s_partial(bm_handle, data.data(), out_mem, data.size() * sizeof(float));
  for (int i = 0; i < batch_size; i++) {
    out_list.append(data[i]);
  }
  return out_list;
}

PYBIND11_MODULE(demo, m) {
    pybind11::class_<OpenClip>(m, "OpenClip")
        .def(pybind11::init<>())
        .def("net_launch", &OpenClip::net_launch)
        .def("init", &OpenClip::init)
        .def("deinit", &OpenClip::deinit)
        .def("forward", &OpenClip::forward);
}