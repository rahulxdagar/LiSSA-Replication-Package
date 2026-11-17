## Configuration
```json
{
  "cache_dir" : "./cache-local-r2c/iTrust-133742243",
  "gold_standard_configuration" : {
    "hasHeader" : false,
    "path" : "./datasets/req2code/iTrust/answer.csv"
  },
  "source_artifact_provider" : {
    "name" : "text",
    "args" : {
      "artifact_type" : "requirement",
      "path" : "./datasets/req2code/iTrust/UC"
    }
  },
  "target_artifact_provider" : {
    "name" : "text",
    "args" : {
      "artifact_type" : "source code",
      "path" : "./datasets/req2code/iTrust/CC"
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
      "model" : "llama3.1:8b",
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
* # TraceLinks (GS): 286
* # Source Artifacts: 131
* # Target Artifacts: 226
## Results
* True Positives: 115
* False Positives: 1852
* False Negatives: 171
* Precision: 0.058464667005592275
* Recall: 0.4020979020979021
* F1: 0.1020861074123391
