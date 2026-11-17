## Configuration
```json
{
  "cache_dir" : "./cache-local-r2c/SMOS-133742243",
  "gold_standard_configuration" : {
    "hasHeader" : false,
    "path" : "./datasets/req2code/SMOS/answer.csv"
  },
  "source_artifact_provider" : {
    "name" : "text",
    "args" : {
      "artifact_type" : "requirement",
      "path" : "./datasets/req2code/SMOS/UC"
    }
  },
  "target_artifact_provider" : {
    "name" : "text",
    "args" : {
      "artifact_type" : "source code",
      "path" : "./datasets/req2code/SMOS/CC"
    }
  },
  "source_preprocessor" : {
    "name" : "artifact",
    "args" : { }
  },
  "target_preprocessor" : {
    "name" : "artifact",
    "args" : { }
  },
  "embedding_creator" : {
    "name" : "ollama",
    "args" : {
      "model" : "bge-m3",
      "num_ctx" : "8192"
    }
  },
  "source_store" : {
    "name" : "custom",
    "args" : { }
  },
  "target_store" : {
    "name" : "custom",
    "args" : {
      "max_results" : "20"
    }
  },
  "classifier" : {
    "name" : "simple_ollama",
    "args" : {
      "model" : "qwen2.5-coder:14b",
      "seed" : "133742243",
      "num_ctx" : "16384",
      "num_predict" : "4096",
      "template" : "Question: Here are two parts of software development artifacts.\n\n{source_type}: '''{source_content}'''\n\n{target_type}: '''{target_content}'''\nAre they related?\n\nAnswer with 'yes' or 'no'.\n"
    }
  },
  "result_aggregator" : {
    "name" : "any_connection",
    "args" : {
      "source_granularity" : "0",
      "target_granularity" : "0"
    }
  },
  "tracelinkid_postprocessor" : {
    "name" : "req2code",
    "args" : { }
  }
}
```

## Stats
* # TraceLinks (GS): 1044
* # Source Artifacts: 67
* # Target Artifacts: 100
## Results
* True Positives: 197
* False Positives: 80
* False Negatives: 847
* Precision: 0.7111913357400722
* Recall: 0.18869731800766285
* F1: 0.2982588947766844
