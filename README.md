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
* Install dependencies: cd into `filmbase-api` Run `pip install -r requirements.txt`
* Run local API:        cd into `filmbase-api` and run `chalice local`

### Example curl command run locally:
* Get estimated prediction with userId ratings: `curl -X GET http://localhost:8000/v1/filmbase/results/ratings`
* Get recommendations of movies with user input ratings: 
`curl -d '{
   "1":{
      "title":"Breakfast Club, The",
      "rating":5
   },
   "2":{
      "title":"Toy Story",
      "rating":3.5
   },
   "3":{
      "title":"Jumanji",
      "rating":2
   },
   "4":{
      "title":"Pulp Fiction",
      "rating":5
   },
   "5":{
      "title":"Akira",
      "rating":4.5
   }
}' -H 'Content-Type: application/json' http://localhost:8000/v1/filmbase/results/recommendations`

### See WIKI page for more details
* https://github.com/rosakng/filmbase/wiki
