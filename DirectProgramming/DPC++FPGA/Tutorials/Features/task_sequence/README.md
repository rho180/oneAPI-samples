

# The `task_sequence` extension
This FPGA tutorial demonstrates how to use the `task_sequence` extension to asynchronously run sub-kernel sets of operations, called tasks, in parallel. The `task_sequence` extension provides a templated class, `task_sequence`, that defines an API for asynchronously launching a parallel task, and for retrieving the results of that task. Objects of this class represent a FIFO queue of tasks matching the order in which these tasks were invoked, as well as an instantiation of the FPGA hardware used to perform the operations of that task.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Basics of task_sequence declaration and usage 
| Time to complete                  | 30 minutes



## Purpose

Use objects of a `task_sequence` class to asynchronously run parallel tasks, and to define the hardware that is instantiated to perform those tasks. An API for invoking tasks and retrieving results of these tasks imposes a FIFO ordering on outstanding tasks and their results. The scope of a `task_sequence` object defines the lifetime in which the hardware represented by that object can be used to perform a task. Users can control hardware reuse and replication by declaring single or multiple objects of the same `task_sequence` class.

### Declaring a `task_sequence`
A `task_sequence` is a templated class that defines a set of operations (tasks), and methods for asynchronously invoking parallel instances of these tasks, and retrieving their results in a FIFO order. The first template parameter is an auto reference to a callable `f` that defines the asynchronous task to be associated with the `task_sequence`. The requirement for an auto reference amounts to a requirement that `f` be statically resolvable at compile time, i.e., not a function pointer. Furthermore, the return type and argument types of `f` must be resolvable and fixed for each definition of `task_sequence`.

The `task_sequence` class optionally takes two additional `unsigned int` template parameters specifying the invocation capacity and response capacity for instantiated `task_sequence` objects. The invocation capacity parameter defines the minimum number of task invocations (see [async](async) in [task_sequence API](task_sequence API) below) that must be supported without any response being collected (see [get](get) in [task_sequence API](task_sequence API) below). This number of `async` invocations without a `get` call is the minimum number that will be supported before a subsequent `async` member function may block. A default value of 1 is assumed if the invocation capacity parameter is not specified.

The response capacity parameter defines the maximum number of outstanding `async` invocations such that all outstanding invocations are guaranteed to make forward progress. Further `async` invocations may block until enough `get` calls are invoked such that the number of outstanding `async` invocations is reduced to the response capacity. A default value of 1 is assumed if the response capacity parameter is not specified.

Object instances of a templated `task_sequence` class represent a specific instantiation of FPGA hardware to perform the operations of the task `f`. Users can control the amount of replication of FPGA hardware by the number of object declarations they use.

```c++
int someTask(int intArg, float floatArg) {
  ...
}

// FPGA code
{
  sycl::ext::intel::experimental::task_sequence<someTask> firstInstance;
  sycl::ext::intel::experimental::task_sequence<someTask> secondInstance; 
  ...
}

```

In this example, `firstInstance` and `secondInstance` are two `task_sequence` objects that implement the task `someTask`, which takes an integer argument and returns and integer result. Since they are two different object instances, they represent two distinct instances of FPGA hardware implementing `someTask`, as well as two separate queues for holding the results of parallel invocations of `someTask`. 

### task_sequence API

`task_sequence` provides two methods for asynchronously invoking and collecting parallel instances of the templated task function. 
  
  - [async](#async)
  - [get](#get)

#### async

The `async` method asynchronously invokes a parallel instance of the templated task function. The `async` method takes the same arguments (the same type and same order) as those defined in the templated task function's signature. 

```c++
int someTask(int intArg, float floatArg);

// FPGA code
{
  sycl::ext::intel::experimental::task_sequence<someTask> firstInstance;
  int argA = ...;
  float argB = ...;
  int argC = ...;
  float argD = ...;

  ...

  firstInstance.async(argA, argB); // first async invocation
  firstInstance.async(argC, argD); // second async invocation
  ...
}

```

In the above example, two asynchronous parallel invocations of `someTask` are invoked on the FPGA hardware represented by the `firstInstance` `task_sequence` object. The first parallel task is invoked with arguments `argA` and `argB`, and the second invocation with `argC` and `argD`. 


#### get
The `get` method collects the result of a `task_sequence` task previously invoked through the `async` method. The `get` method for a particular `task_sequence` object has the same return type as the templated task function for the object. Calling `get` returns results in the same order in which the tasks were invoked.
 
```c++
// FPGA code
{
  sycl::ext::intel::experimental::task_sequence<someTask> firstInstance;
  int argA = ...;
  float argB = ...;
  int argC = ...;
  float argD = ...;

  ...

  firstInstance.async(argA, argB); // first async invocation
  firstInstance.async(argC, argD); // second async invocation
  ...
  auto firstResult = firstInstance.get(); // returns the result of invocation with (argA, argB)
  auto secondResult = firstInstance.get(); // returns the result of invocation with (argC, argD)
  ...
}

```

In this continuation of the `async` example, `firstResult` contains the return value of the `async` invocation using `(argA, argB)`, and `secondResult` contains the return value of the `async` invocation using `(argC, argD)`. 

The `get` method is a blocking call. That is, if no previous `async` invocation has completed, `get` will block until one has.

### Testing the Tutorial
In `task_sequence.cpp`, the dot product of a 16k element vector is calculated twice. The first calculation, performed in the `SequentialTask` kernel, is performed by a single `async` invocation of the `dotProduct` function by a single `task_sequence` object.


```c++
h.single_task<SequentialTask>([=]() {
  sycl::ext::intel::experimental::task_sequence<dotProduct> whole;
  whole.async(in_acc.get_pointer(), 0, count);
  out_acc[0] = whole.get();
});
```
The second calculation, performed by the `ParallelTask` kernel, is performed by 4 `async` invocations of the `dotProduct` function via 4 different `task_sequence` objects. Each `async` invocation operates on one-quarter of the vector. Since each `async` invocation utilizes its own FPGA hardware and operates on a different quarter of the vector, each partial dot product calculation can be done in parallel, speeding up the result.

```c++
h.single_task<ParallelTask>([=]() {
  sycl::ext::intel::experimental::task_sequence<dotProduct> firstQuarter;
  sycl::ext::intel::experimental::task_sequence<dotProduct> secondQuarter;
  sycl::ext::intel::experimental::task_sequence<dotProduct> thirdQuarter;
  sycl::ext::intel::experimental::task_sequence<dotProduct> fourthQuarter;
  int quarterCount = count/4;
  firstQuarter.async(in_acc.get_pointer(), 0, quarterCount);
  secondQuarter.async(in_acc.get_pointer(), quarterCount, quarterCount);
  thirdQuarter.async(in_acc.get_pointer(), 2*quarterCount, quarterCount);
  fourthQuarter.async(in_acc.get_pointer(), 3*quarterCount, quarterCount);
          out_acc[1] = firstQuarter.get() + secondQuarter.get() + thirdQuarter.get() + fourthQuarter.get();
});
```

## Key Concepts
* Basics of declaring `task_sequence` objects
* Using `task_sequence` `async` and `get` API for invoking and collecting parallel tasks

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `task_sequence` Tutorial

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: `. /opt/intel/oneapi/setvars.sh`
>
> Linux User: `. ~/intel/oneapi/setvars.sh`
>
> Windows: `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If you are running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are supported only on `fpga_compile` nodes. Executing programs on FPGA hardware is supported only on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.  You cannot compile or execute programs on FPGA hardware on the `login` nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, increase the job timeout to 12h.

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System

1. Generate the Makefile by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the following command:

   ```
   cmake ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Generate the optimization report:
     ```
     make report
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/task_sequence.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. 
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report:
     ```
     nmake report
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

>Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>

>**Tip**: If you encounter issues with long paths when compiling under Windows*, you might have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`.  You can then run `cmake` from that directory, and provide `cmake` with the full path to your sample directory.

### Troubleshooting

If an error occurs, get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*).
For instructions, refer to [FPGA Workflows on Third-Party IDEs for Intel&reg; oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).

## Examining the Reports

Locate `report.html` in the `task_sequence_report.prj/reports/` directory. Open the report in any of the following web browsers:  Chrome*, Firefox*, Edge*, or Internet Explorer*.

Open the **Views** menu and select **System Viewer**.

In the left-hand pane, select **SequentialTask.B0** under the System hierarchy.

In the main **System Viewer** pane, the `task_sequence` `async` and `get` for the single `whole` `task_sequence` object are highlighted as a `WR` and `RD` node respectively. These represent a write pipe for writing the arguments and start command to the `dotProduct` task function, and a read pipe for returning the results.

Now select **ParallelTask.B0** in the left-hand pane.

In the main **System Viewer(( pane, the four `task_sequence` `async` and `get` commands for the four `task_sequence` objects of the `parallelTask` kernel are highlighted. These represent the four parallel `async` invocations in this kernel. As in the the `sequentialTask` kernel, the `WR` nodes represent pipes for writing the arguments and start command to each instance of the `dotProduct` task function (since there are 4 `task_sequence` objects, there are 4 hardware instances), and the `RD` nodes represent pipes for returning the results.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./task_sequence.fpga_emu     (Linux)
     task_sequence.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./task_sequence.fpga         (Linux)
     task_sequence.fpga.exe     (Windows)
     ```

### Example of Output

```
PASSED sequential test
PASSED parallel test
Sequential time: 29489.7 ms
Parallel time: 12050.7 ms
```
