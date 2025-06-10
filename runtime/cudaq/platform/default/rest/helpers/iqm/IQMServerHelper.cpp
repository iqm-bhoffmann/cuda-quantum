/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace cudaq {

class IQMServerHelper : public ServerHelper {
protected:
  /// @brief The base URL
  std::string iqmServerUrl = "http://localhost/cocos/";

  /// @brief The default cortex-cli tokens file path
  std::optional<std::string> tokensFilePath = std::nullopt;

  /// @brief Program target QPU architecture
  std::string targetArchitecture = "";

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

  /// @brief Lookup table for translating the qubit names to index numbers
  std::map<std::string, uint> qubitNameMap;

  /// @brief Adjacency map for each qubit
  std::vector<std::set<uint>> qubitAdjacencyMap;

  /// @brief Parse cortex-cli tokens JSON for the API access token
  std::optional<std::string> readApiToken() const {
    if (!tokensFilePath.has_value()) {
      cudaq::info(
          "tokensFilePath is not set, assuming no authentication is required");
      return std::nullopt;
    }

    std::string unwrappedTokensFilePath = tokensFilePath.value();
    std::ifstream tokensFile(unwrappedTokensFilePath);
    if (!tokensFile.is_open()) {
      throw std::runtime_error("Unable to open tokens file: " +
                               unwrappedTokensFilePath);
    }
    nlohmann::json tokens;
    tokensFile >> tokens;
    tokensFile.close();
    return tokens["access_token"].get<std::string>();
  }

  /// @brief Fetch the quantum architecture from server
  void fetchQuantumArchitecture();

  /// @brief Write the dynamic quantum architecture file
  void writeQuantumArchitectureFile(std::string filename);

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "iqm"; }
  RestHeaders getHeaders() override;

  void initialize(BackendConfig config) override {
    backendConfig = config;

    bool emulate = false;
    auto iter = backendConfig.find("emulate");
    if (iter != backendConfig.end()) {
      emulate = iter->second == "true";
    }

    // Set an alternate base URL if provided.
    iter = backendConfig.find("url");
    if (iter != backendConfig.end()) {
      iqmServerUrl = iter->second;
    }

    // Allow overriding IQM Server Url, the compiled program will still work if
    // architecture matches. This is useful in case we're using the same program
    // against different backends, for example simulated and actually connected
    // to the hardware.
    auto envIqmServerUrl = getenv("IQM_SERVER_URL");
    if (envIqmServerUrl) {
      iqmServerUrl = std::string(envIqmServerUrl);
    }

    if (!iqmServerUrl.ends_with("/"))
      iqmServerUrl += "/";
    cudaq::debug("iqmServerUrl = {}", iqmServerUrl);

    if (emulate) {
      cudaq::info(
          "Emulation is enabled, ignore tokens file and IQM Server URL");
      return;
    }

    // Set alternative cortex-cli tokens file path if provided via env var
    auto envTokenFilePath = getenv("IQM_TOKENS_FILE");
    auto defaultTokensFilePath =
        std::string(getenv("HOME")) + "/.cache/iqm-cortex-cli/tokens.json";
    cudaq::debug("defaultTokensFilePath = {}", defaultTokensFilePath);
    if (envTokenFilePath) {
      tokensFilePath = std::string(envTokenFilePath);
    } else if (cudaq::fileExists(defaultTokensFilePath)) {
      tokensFilePath = defaultTokensFilePath;
    }
    cudaq::debug("tokensFilePath = {}", tokensFilePath.value_or("not set"));

    // Fetch the quantum-architecture of the configured IQM server
    fetchQuantumArchitecture();
  }

  /// @brief Create a job payload for the provided quantum codes
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Return the job id from the previous job post
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Return the URL for retrieving job results
  std::string constructGetJobPath(ServerMessage &postResponse) override;
  std::string constructGetJobPath(std::string &jobId) override;

  /// @brief Return next results polling interval
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override;

  /// @brief Return true if the job is done
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override;

  /// @brief Update `passPipeline` with architecture-specific pass options
  void updatePassPipeline(const std::filesystem::path &platformPath,
                          std::string &passPipeline) override;
};

ServerJobPayload
IQMServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> messages;

  // cuda-quantum expects every circuit to be a separate job,
  // so we cannot use the batch mode
  for (auto &circuitCode : circuitCodes) {
    ServerMessage message = ServerMessage::object();
    message["circuits"] = ServerMessage::array();
    message["shots"] = shots;

    ServerMessage yac = nlohmann::json::parse(circuitCode.code);
    yac["name"] = circuitCode.name;
    message["circuits"].push_back(yac);
    messages.push_back(message);
  }

  // Get the headers
  RestHeaders headers = generateRequestHeader();

  // return the payload
  return std::make_tuple(iqmServerUrl + "jobs", headers, messages);
}

std::string IQMServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["id"].get<std::string>();
}

std::string IQMServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return "jobs" + postResponse["id"].get<std::string>() + "/counts";
}

std::string IQMServerHelper::constructGetJobPath(std::string &jobId) {
  return iqmServerUrl + "jobs/" + jobId + "/counts";
}

std::chrono::microseconds
IQMServerHelper::nextResultPollingInterval(ServerMessage &postResponse) {
  return std::chrono::seconds(1); // jobs never take less than few seconds
};

bool IQMServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  cudaq::debug("getJobResponse: {}", getJobResponse.dump());

  auto jobStatus = getJobResponse["status"].get<std::string>();
  std::unordered_set<std::string> terminalStatuses = {"ready", "failed",
                                                      "aborted"};
  return terminalStatuses.find(jobStatus) != terminalStatuses.end();
}

cudaq::sample_result
IQMServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobID) {
  cudaq::info("postJobResponse: {}", postJobResponse.dump());

  // check if the job succeeded
  auto jobStatus = postJobResponse["status"].get<std::string>();
  if (jobStatus != "ready") {
    auto jobMessage = postJobResponse["message"].get<std::string>();
    throw std::runtime_error("Job status: " + jobStatus +
                             ", reason: " + jobMessage);
  }

  auto counts_batch = postJobResponse["counts_batch"];
  if (counts_batch.is_null()) {
    throw std::runtime_error("No counts in the response");
  }

  // assume there is only one measurement and everything goes into the
  // GlobalRegisterName of `sample_results`
  std::vector<ExecutionResult> srs;

  for (auto &counts : counts_batch.get<std::vector<ServerMessage>>()) {
    srs.push_back(ExecutionResult(
        counts["counts"].get<std::unordered_map<std::string, std::size_t>>()));
  }

  sample_result sampleResult(srs);

  // The original sampleResult is ordered by qubit number (FIXME: VERIFY THIS)
  // Now reorder according to reorderIdx[]. This sorts the global bitstring in
  // original user qubit allocation order.
  auto thisJobReorderIdxIt = reorderIdx.find(jobID);
  if (thisJobReorderIdxIt != reorderIdx.end()) {
    auto &thisJobReorderIdx = thisJobReorderIdxIt->second;
    if (!thisJobReorderIdx.empty())
      sampleResult.reorder(thisJobReorderIdx);
  }

  return sampleResult;
}

std::map<std::string, std::string>
IQMServerHelper::generateRequestHeader() const {
  std::map<std::string, std::string> headers{
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"User-Agent", "cudaq/IQMServerHelper"},
      {"Accept", "*/*"}};
  auto apiToken = readApiToken();
  if (apiToken.has_value()) {
    headers["Authorization"] = "Bearer " + apiToken.value();
  };
  return headers;
}

void IQMServerHelper::updatePassPipeline(
    const std::filesystem::path &platformPath, std::string &passPipeline) {
  std::string pathToFile;
  auto iter = backendConfig.find("mapping_file");
  if (iter != backendConfig.end()) {
    // Use provided path to file
    pathToFile = iter->second;
  } else {
    // Construct path to file
    pathToFile =
        std::string(platformPath / std::string("mapping/iqm") /
                    (std::string("qpu-architecture.txt")));
    cudaq::debug("quantum architecture file: {}", pathToFile);

    writeQuantumArchitectureFile(pathToFile);
  }

  // Add leading and trailing single quotes in case there are spaces in the
  // filename.
  pathToFile.insert(0, "'");
  pathToFile.append("'");

  passPipeline =
      std::regex_replace(passPipeline, std::regex("%QPU_ARCH%"), pathToFile);
}

RestHeaders IQMServerHelper::getHeaders() { return generateRequestHeader(); }

/**
 * Fetch the quantum architecture from the configured URL and create a qubit
 * adjacency map. The map contains only qubits which can be measured and can
 * be used in prx-gates as well as cz-gates. As qubits pairs for cz-gates
 * connect only a few qubits the information about neighbors is stored as sets
 * within a vector of all qubits to save memory.
 */
void IQMServerHelper::fetchQuantumArchitecture() {
  try {
    RestClient client;
    auto headers = generateRequestHeader();

    // From the Static Quantum Architecture we need the total number of qubits
    // and the list of qubit designations.
    auto staticQuantumArchitecture =
      client.get(iqmServerUrl, "api/v1/quantum-architecture", headers);
    cudaq::debug("Static QA={}", staticQuantumArchitecture.dump());

    // The number of qubits of this quantum architecture.
    uint qubitCount = staticQuantumArchitecture["qubits"].size();

    // Enumerate the qubit designations.
    uint idx = 0;
    for (auto qubit : staticQuantumArchitecture["qubits"]) {
      qubitNameMap[qubit] = idx++;
    }

    // From the Dynamic Quantum Architecture we need the list of qubit pairs
    // which can form cz-gates and additionally the lists of single qubits
    // which can do prx-gates and support measurement.
    auto dynamicQuantumArchitecture =
      client.get(iqmServerUrl, "api/v1/calibration/default/gates", headers);
    cudaq::debug("Dynamic QA={}", dynamicQuantumArchitecture.dump());

    cudaq::info("Server {} has {} qubits", iqmServerUrl, qubitCount);

    // Initialise the adjacency map with an empty set for each qubit
    std::set<uint> noConnections;
    qubitAdjacencyMap.reserve(qubitCount);
    for (uint i = 0; i < qubitCount; i++) {
      qubitAdjacencyMap.emplace_back(noConnections);
    }

    auto &cz_loci = dynamicQuantumArchitecture["gates"]["cz"]
                        ["implementations"]["crf_crf"]["loci"];
    auto &prx_loci = dynamicQuantumArchitecture["gates"]["prx"]
                        ["implementations"]["drag_crf"]["loci"];
    auto &measure_loci = dynamicQuantumArchitecture["gates"]["measure"]
                            ["implementations"]["constant"]["loci"];

    cudaq::debug("cz_loci={}", cz_loci.dump());
    cudaq::debug("prx_loci={}", prx_loci.dump());
    cudaq::debug("measure_loci={}", measure_loci.dump());

    // Iterate over all cz loci and add only those to the output list for which
    // all qubits have both measure and prx capability.
    for (auto cz : cz_loci) {
      bool lociUsable = true; // assume usable until proven otherwise
      bool found = false;

      // each cz loci connects 2 qubits - check each of these individually
      for (auto qubit : cz) {   // cz is an array of strings

        // Check whether this qubit has prx capability.
        found = false;
        for (auto it = prx_loci.begin(); it != prx_loci.end() ; it++) {
          if ((*it)[0] == qubit) {
            found = true;
            break;
          }
        }
        if (!found) {
          lociUsable = false;
          break;
        }

        // Check whether this qubit has measurement capability.
        found = false;
        for (auto it = measure_loci.begin(); it != measure_loci.end() ; it++) {
          if ((*it)[0] == qubit) {
            found = true;
            break;
          }
        }
        if (!found) {
          lociUsable = false;
          break;
        }
      }

      if (lociUsable) {
        // This cz_loci has passed all the tests so add it to the list.
        cudaq::debug("usable cz_loci {}", cz.dump());
        qubitAdjacencyMap[qubitNameMap[cz[0]]].insert(qubitNameMap[cz[1]]);
        qubitAdjacencyMap[qubitNameMap[cz[1]]].insert(qubitNameMap[cz[0]]);
      }
    } // for all cz loci
  }
  catch (const std::exception &e) {
    throw std::runtime_error("Unable to get quantum architecture from \"" +
                            iqmServerUrl + "\": " + std::string(e.what()));
  }
} // IQMServerHelper::fetchQuantumArchitecture()

/**
 * Write the dynamic quantum architecture to the specified filename. If the
 * file cannot be opened for writing an exception is thrown.
 * @param filename String with path+filename to write to.
 * @throws std::runtime_error Thrown when file cannot be opened for writing.
 */
void IQMServerHelper::writeQuantumArchitectureFile(std::string filename) {
  uint qubitCount = qubitAdjacencyMap.size();

  FILE* file = fopen(filename.c_str(), "w");
  if (!file) {
    throw std::runtime_error("cannot write QPU architecture file" + filename);
  }

  // Header information
  fprintf(file, "NOTE: automatically generated by " __FILE__ "\n"
                "      for server at URL: %s\n\n", iqmServerUrl.c_str());
  fprintf(file, "Number of nodes: %u\n", qubitCount);
  fprintf(file, "Number of edges: ?\n\n");

  // Write one line for each qubit listing the adjacent qubits.
  for (uint i = 0; i < qubitCount; i++) {
    bool first = true;

    std::string outputLine = std::to_string(i) + " --> {";
    for (uint node : qubitAdjacencyMap[i]) {
      if (first)
        first=false;
      else
        outputLine += ", ";
      outputLine += std::to_string(node);
    }
    outputLine += "}\n";

    fwrite(outputLine.c_str(), outputLine.length(), 1, file);
  }

  fclose(file);
} // IQMServerHelper::writeQuantumArchitectureFile()

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IQMServerHelper, iqm)
