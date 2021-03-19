
### Running the demo 
To install the python dependencies, simply do: 
```python
$ cd demo
$ pip install -r requirements.txt
```

- Check if Django is installed:
 ```
 $ python -m django --version
 ```
 
 - Run the app: 
```
$ python3.6 manage.py runserver
e.g.  python manage.py runserver 0.0.0.0:4003
```

### notes 
no models in ./models/demo-models/squad2-better-bert-base, the only file in that directory lists the required files for running a prediction model.
