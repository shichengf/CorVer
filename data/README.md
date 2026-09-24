# Selected training questions

`train.json` contains 6,621 selected question/reference records for the main method, in a fixed order. `manifest.json` records the count and SHA256. Each row has an `id`, a `question` string, an `answers` string, and a `title` string. The `title` field may contain multiple titles separated by semicolons. Evidence is never inserted into the policy prompt.

Source projects: [Natural Questions / NQ-Open](https://github.com/google-research-datasets/natural-questions) and [WebQuestions](https://nlp.stanford.edu/software/sempre/). The repository's MIT code license does not replace source data or Wikipedia terms.
