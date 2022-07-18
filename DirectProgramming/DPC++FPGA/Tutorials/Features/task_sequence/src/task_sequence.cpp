#include <CL/sycl.hpp>
#include <ext/intel/experimental/task_sequence.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include <iostream>
#include <random>

#include "dpc_common.hpp"

using ValueT = float;

// compute the dot product of 'sz' elements of vector 'v', beginning at index
// 's'
ValueT dotProduct(ValueT* v, size_t s, size_t sz) {
  int result = 1;
  for (size_t i = s; i < s + sz; i++) result *= v[i];

  return result;
}

// return the absolute value of 'x'
template <typename T>
T abs(T x) {
  if (x > 0)
    return x;
  else
    return -x;
}

// Kernel identifiers
class SequentialTask;
class ParallelTask;

int main(int argc, char* argv[]) {
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector selector;
#else
  sycl::ext::intel::fpga_selector selector;
#endif

  size_t count = 16384;
  if (argc > 1) count = atoi(argv[1]);

  if (count <= 0) {
    std::cerr << "ERROR: 'count' must be positive" << std::endl;
    return 1;
  }

  try {
    // create the device queue
    sycl::queue q(selector, dpc_common::exception_handler,
                  sycl::property::queue::enable_profiling{});

    // create input and golden output data
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<ValueT> distr(0, 1);
    std::vector<ValueT> in(count), out(2);
    for (size_t i = 0; i < count; i++) {
      in[i] = distr(eng);
    }

    ValueT golden = dotProduct(in.data(), 0, count);

    // variables for profiling times
    double start, end, sequentialTime, parallelTime;

    // create scope so that buffer destructors are invoked before output
    // is checked
    {
      sycl::buffer in_buf(in);
      sycl::buffer out_buf(out);

      sycl::event e = q.submit([&](sycl::handler& h) {
        sycl::accessor in_acc(in_buf, h, sycl::read_only);
        sycl::accessor out_acc(out_buf, h, sycl::write_only);
        h.single_task<SequentialTask>([=]() {
          sycl::ext::intel::experimental::task_sequence<dotProduct> whole;
          whole.async(in_acc.get_pointer(), 0, count);
          out_acc[0] = whole.get();
        });
      });
      q.wait();

      start =
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

      // unit is nano second, convert to ms
      sequentialTime = (double)(end - start) * 1e-6;

      e = q.submit([&](sycl::handler& h) {
        sycl::accessor in_acc(in_buf, h, sycl::read_only);
        sycl::accessor out_acc(out_buf, h, sycl::write_only);
        h.single_task<ParallelTask>([=]() {
          sycl::ext::intel::experimental::task_sequence<dotProduct>
              firstQuarter;
          sycl::ext::intel::experimental::task_sequence<dotProduct>
              secondQuarter;
          sycl::ext::intel::experimental::task_sequence<dotProduct>
              thirdQuarter;
          sycl::ext::intel::experimental::task_sequence<dotProduct>
              fourthQuarter;
          int quarterCount = count / 4;
          firstQuarter.async(in_acc.get_pointer(), 0, quarterCount);
          secondQuarter.async(in_acc.get_pointer(), quarterCount, quarterCount);
          thirdQuarter.async(in_acc.get_pointer(), 2 * quarterCount,
                             quarterCount);
          fourthQuarter.async(in_acc.get_pointer(), 3 * quarterCount,
                              quarterCount);
          out_acc[1] = firstQuarter.get() + secondQuarter.get() +
                       thirdQuarter.get() + fourthQuarter.get();
        });
      });
      q.wait();

      start =
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

      // unit is nano second, convert to ms
      parallelTime = (double)(end - start) * 1e-6;
    }

    if (abs(out[0] - golden) < (ValueT)0.001)
      std::cout << "PASSED sequential test" << std::endl;
    else
      std::cout << "FAILED" << std::endl;

    if (abs(out[1] - golden) < (ValueT)0.001)
      std::cout << "PASSED parallel test" << std::endl;
    else
      std::cout << "FAILED" << std::endl;

    std::cout << "Sequential time: " << sequentialTime << " ms" << std::endl;
    std::cout << "Parallel time: " << parallelTime << " ms" << std::endl;

  } catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }
}
