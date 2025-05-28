
/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "cudaq/algorithm.h"

#include <fstream>
#include <regex>

std::string backendString =
    "iqm;emulate;false;url;"
    "http://localhost:62443";

CUDAQ_TEST(IQMTester, executeOneMeasuredQubitProgram) {
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.y(qubit[0]);
  kernel.z(qubit[0]);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(IQMTester, executeSeveralMeasuredQubitProgram) {
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  kernel.mz(qubit[1]);

  auto counts = cudaq::sample(kernel);
  EXPECT_EQ(counts.size(), 4);
}

CUDAQ_TEST(IQMTester, executeLoopOverQubitsProgram) {
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto N = 5;
  auto kernel = cudaq::make_kernel();

  auto qubit = kernel.qalloc(N);
  kernel.h(qubit[0]);

  kernel.for_loop(
      0, N - 1, [&](auto i) { kernel.x<cudaq::ctrl>(qubit[i], qubit[i + 1]); });

  kernel.mz(qubit[0]);
  auto counts = cudaq::sample(kernel);

  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(IQMTester, executeMultipleMeasuredQubitsProgram) {
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto N = 2;
  auto kernel = cudaq::make_kernel();

  auto qubit = kernel.qalloc(N);
  kernel.h(qubit[0]);
  kernel.x<cudaq::ctrl>(qubit[0], qubit[1]);

  kernel.mz(qubit);

  auto counts = cudaq::sample(kernel);
  EXPECT_EQ(counts.size(), 4);
}

CUDAQ_TEST(IQMTester, iqmServerUrlEnvOverride) {
  EXPECT_THAT(
      []() {
        setenv("IQM_SERVER_URL", "fake-fake-fake", true);
        auto &platform = cudaq::get_platform();
        platform.setTargetBackend(backendString);
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Could not resolve host: fake-fake-fake")));
}

CUDAQ_TEST(IQMTester, tokenFilePathEnvOverride) {
  EXPECT_THAT(
      []() {
        setenv("IQM_TOKENS_FILE", "fake-fake-fake", true);
        auto &platform = cudaq::get_platform();
        platform.setTargetBackend(backendString);
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Unable to open tokens file: fake-fake-fake")));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleMock(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
