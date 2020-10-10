# Python machine learning backend for Moodle #
=======================================================

This package is used by Moodle's mlbackend_python plugin.

Info about packaging a new version in _readme_moodle.md: https://github.com/moodlehq/moodle-mlbackend-python/blob/master/readme_moodle.md

## How Moodle Integrates and uses this library ##

The core of the code in the Moodle PHP backend is located here: https://github.com/moodle/moodle/tree/master/lib/mlbackend/python

Which has two possible configiruations:
1) Moodle talks to a backend REST server that hosts this library/models
2) Moodle calls a python process directly to get/generates models

See - https://github.com/moodle/moodle/blob/master/lib/mlbackend/python/classes/processor.php#L97

1) As a result as interactions occur and models are generated they need to be stored and this is currently done via the TensorFlow Saver (https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver) which is similar to Pickle.
2) If using the REST API, there is an included Flask service however it should be noted that these are synchronous calls so long running HTTP requests can/will timeout. It is recommended to use the direct python process call for large datasets.

## Flow of calls ##

1) REST or Python process call to `training`, at the conclusion the model is "saved"
2) REST or Python process call to `predict` to get actual predictions
3) [Optional] REST or Python call to `evaluation` to get models scores

## Python Process Calls ##
When the python process is called directly the moduls read directly from the system arguments for their inputs

## Flask REST Calls ##
1) All  REST calls supply a Basic Authorization header which the Flask apps looks in the `MOODLE_MLBACKEND_PYTHON_USERS` env vars for logins to use
2) Flask app can be configured to be backed by local storage or Amazon S3
3) Flask app REST calls can then have data pulled from either source, implies data is saved there by the caller before this app being called
