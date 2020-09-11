## filmbase

### Description
Simple Movie Recommendation API with Demographic Filtering, Content-Based Filtering, and Collaborative Filtering

### Chalice
This application uses the [Chalice Serverless Microframework](https://chalice.readthedocs.io/en/latest/) for routing.

### Requirements

* Python3.7+

### Local development

* Create a virtual env: In root directory, `python3.7 -mvenv env`
* Activate virtual env: Run `source env/bin/activate`
* Install dependencies: Run `pip install -r requirements.txt`
* Run local API:        cd into `filmbase-api` and run `chalice local`

### Example curl command run locally:
* Get estimated prediction with userId ratings: `curl -X GET http://localhost:8000/v1/filmbase/results/ratings`
