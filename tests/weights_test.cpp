// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Checks that packed weight files parse correctly, that the embedded copies are identical to the
// files, and that corrupted weights are rejected.

#include "weights.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace vkoidn;

static std::vector<char> readFile(const char* path)
{
  std::ifstream file(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

static std::string getName(const std::string& path)
{
  size_t begin = path.find_last_of("/\\");
  begin = (begin == std::string::npos) ? 0 : begin + 1;
  const size_t end = path.find_last_of('.');
  return path.substr(begin, end - begin);
}

int main(int argc, char* argv[])
{
  int numErrors = 0;

  for (int i = 1; i < argc; ++i)
  {
    const std::vector<char> blob = readFile(argv[i]);
    const std::string name = getName(argv[i]);

    Weights weights;
    std::string error;
    if (!parseWeights(blob.data(), blob.size(), weights, &error))
    {
      printf("FAILED %s: %s\n", name.c_str(), error.c_str());
      ++numErrors;
      continue;
    }

    // The embedded copy must be identical
    size_t embeddedSize = 0;
    const void* embedded = getEmbeddedWeights(name.c_str(), &embeddedSize);
    const char* embeddedStatus = "not embedded";
    if (embedded)
    {
      if (embeddedSize == blob.size() && memcmp(embedded, blob.data(), blob.size()) == 0)
        embeddedStatus = "embedded copy identical";
      else
      {
        printf("FAILED %s: embedded copy differs\n", name.c_str());
        ++numErrors;
        continue;
      }
    }

    // Corrupting the graph must be detected
    std::vector<char> corrupted = blob;
    WeightsLayer* layers = reinterpret_cast<WeightsLayer*>(corrupted.data() + sizeof(WeightsHeader));
    layers[3].inChannels0 += 16;
    Weights dummy;
    if (parseWeights(corrupted.data(), corrupted.size(), dummy))
    {
      printf("FAILED %s: corrupted weights accepted\n", name.c_str());
      ++numErrors;
      continue;
    }
    if (parseWeights(blob.data(), blob.size() / 2, dummy))
    {
      printf("FAILED %s: truncated weights accepted\n", name.c_str());
      ++numErrors;
      continue;
    }

    printf("OK     %-26s %2u layers, %u -> %u channels, %6.2f MB GPU data, %s\n",
           name.c_str(), weights.header->numLayers, weights.header->inputChannels,
           weights.header->outputChannels, weights.dataByteSize / (1024. * 1024.), embeddedStatus);
  }

  printf(numErrors ? "%d error(s)\n" : "All weights OK\n", numErrors);
  return numErrors ? 1 : 0;
}
