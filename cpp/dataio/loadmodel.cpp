#include <ctime>
#include "../dataio/loadmodel.h"

#include <ghc/filesystem.hpp>

//------------------------
#include "../core/using.h"
//------------------------

template <typename TP>
std::time_t to_time_t(TP tp)
{
  using namespace std::chrono;
  auto sctp = time_point_cast<system_clock::duration>(
    tp - TP::clock::now() + system_clock::now()
  );
  return system_clock::to_time_t(sctp);
}

static const vector<string> ACCEPTABLE_MODEL_SUFFIXES {
  ".bin.gz",
  ".bin",
  "model.txt.gz",
  "model.txt"
};
static const vector<string> GENERIC_MODEL_NAMES {
  "model.bin.gz",
  "model.bin",
  "model.txt.gz",
  "model.txt"
  "Model.bin.gz",
  "Model.bin",
  "Model.txt.gz",
  "Model.txt"
  "MODEL.bin.gz",
  "MODEL.bin",
  "MODEL.txt.gz",
  "MODEL.txt"
  "model.ckpt",
  "Model.ckpt"
  "MODEL.ckpt",
  "model.checkpoint",
  "Model.checkpoint"
  "MODEL.checkpoint",
  "model",
  "Model"
  "MODEL",
};

static bool endsWithAnySuffix(const string& path, const vector<string>& suffixes) {
  for(const string& suffix: suffixes) {
    if(Global::isSuffix(path,suffix))
      return true;
  }
  return false;
}

// Extract step number from model path like "prefix-s123456-d789012"
// Returns -1 if no step number found
static int64_t extractStepNumber(const string& pathStr) {
  size_t sPos = pathStr.rfind("-s");
  if(sPos == string::npos)
    return -1;
  
  size_t startPos = sPos + 2; // Skip "-s"
  size_t endPos = pathStr.find('-', startPos);
  if(endPos == string::npos)
    endPos = pathStr.find('/', startPos);
  if(endPos == string::npos)
    endPos = pathStr.find('\\', startPos);
  if(endPos == string::npos)
    endPos = pathStr.length();
  
  try {
    string numStr = pathStr.substr(startPos, endPos - startPos);
    return std::stoll(numStr);
  }
  catch(...) {
    return -1;
  }
}

bool LoadModel::findLatestModel(const string& modelsDir, Logger& logger, string& modelName, string& modelFile, string& modelDir, time_t& modelTime) {
  namespace gfs = ghc::filesystem;
  (void)logger;

  // For blob storage compatibility, use numeric comparison on step numbers in directory names
  // Model directories are typically named like: prefix-s123456-d789012 where s{step} is the training step
  bool hasLatestPath = false;
  string latestPathStr;
  gfs::path latestPath;
  gfs::file_time_type latestTime;
  int64_t latestStepNumber = -1;
  
  for(const auto& dirEntry: gfs::recursive_directory_iterator(gfs::u8path(modelsDir))) {
    gfs::path filePath = dirEntry.path();
    if(gfs::is_regular_file(filePath) && endsWithAnySuffix(filePath.filename().u8string(), ACCEPTABLE_MODEL_SUFFIXES)) {
      string pathStr = filePath.u8string();
      int64_t stepNumber = extractStepNumber(pathStr);
      
      // Compare by step number first (numeric), then lexicographic, then mtime as fallback
      bool isNewer = false;
      if(!hasLatestPath) {
        isNewer = true;
      }
      else if(stepNumber >= 0 && latestStepNumber >= 0) {
        // Both have step numbers, compare numerically
        isNewer = (stepNumber > latestStepNumber);
      }
      else if(stepNumber >= 0 && latestStepNumber < 0) {
        // New path has step number, old doesn't - prefer new
        isNewer = true;
      }
      else if(stepNumber < 0 && latestStepNumber >= 0) {
        // Old path has step number, new doesn't - keep old
        isNewer = false;
      }
      else {
        // Neither has step number, use lexicographic + mtime
        isNewer = (pathStr > latestPathStr || (pathStr == latestPathStr && gfs::last_write_time(filePath) > latestTime));
      }
      
      if(isNewer) {
        hasLatestPath = true;
        latestPathStr = pathStr;
        latestPath = filePath;
        latestTime = gfs::last_write_time(filePath);
        latestStepNumber = stepNumber;
      }
    }
  }

  modelName = "random";
  modelFile = "/dev/null";
  modelDir = "/dev/null";
  modelTime = (std::time_t)(0);
  if(hasLatestPath) {
    modelFile = latestPath.u8string();
    modelDir = latestPath.parent_path().u8string();
    if(contains(GENERIC_MODEL_NAMES, latestPath.filename().u8string())) {
      modelName = latestPath.parent_path().filename().u8string();
    }
    else {
      modelName = latestPath.filename().u8string();
    }
    modelTime = to_time_t(latestTime);
  }

  return true;
}

void LoadModel::setLastModifiedTimeToNow(const string& filePath, Logger& logger) {
  namespace gfs = ghc::filesystem;
  gfs::path path(gfs::u8path(filePath));
  try {
    gfs::last_write_time(path, gfs::file_time_type::clock::now());
  }
  catch(gfs::filesystem_error& e) {
    logger.write("Warning: could not set last modified time for " + filePath + ": " + e.what());
  }
}

void LoadModel::deleteModelsOlderThan(const string& modelsDir, Logger& logger, const time_t& time) {
  namespace gfs = ghc::filesystem;
  vector<gfs::path> pathsToRemove;
  for(gfs::directory_iterator iter(gfs::u8path(modelsDir)); iter != gfs::directory_iterator(); ++iter) {
    gfs::path filePath = iter->path();
    if(gfs::is_directory(filePath))
      continue;
    string filePathStr = filePath.u8string();
    if(Global::isSuffix(filePathStr,".bin.gz") ||
       Global::isSuffix(filePathStr,".txt.gz") ||
       Global::isSuffix(filePathStr,".bin") ||
       Global::isSuffix(filePathStr,".txt")) {
      time_t thisTime = to_time_t(gfs::last_write_time(filePath));
      if(thisTime < time) {
        pathsToRemove.push_back(filePath);
      }
    }
  }

  for(size_t i = 0; i<pathsToRemove.size(); i++) {
    logger.write("Deleting old unused model file: " + pathsToRemove[i].u8string());
    try {
      gfs::remove(pathsToRemove[i]);
    }
    catch(gfs::filesystem_error& e) {
      logger.write("Warning: could not delete " + pathsToRemove[i].u8string() + ": " + e.what());
    }
  }

}
