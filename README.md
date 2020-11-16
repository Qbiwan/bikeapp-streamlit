# Streamlit Bike-App deployed on Heroku

## Directory

```
bikeapp-streamlit
├── model                                         
│     └── xgbmodel.h5                            <- trained xgboost model  
├── Demo                                         
│     └── Demo.gif 
├── streamlit_app.py                             <- streamlit                                    
├── train.py                                     <- train keras model
├── Procfile                                     <- commands to be executed by heroku on app startup 
├── requirements.txt                             <- heroku app dependencies  
├── setup.sh                                     <- setup for heroku  
├── runtime.txt                                  <- specify python version for heroku  
└── README.md
```




Bike App makes prediction about the number of bikes on the road based on temperature, humidity, and windspeed.

## Demo

![Demo](Demo/demo.gif)

Bike App is deployed on Heroku live at [HERE](https://bikeapp-streamlit.herokuapp.com/)

## Libraries

- streamlit
- pandas
- xgboost
- pickle
- sklearn
