# EnergyEfficiency

A web app that predicts the energy efficiency of a building given its characteristics.


## Dataset Citation

Tsanas, A., Xifara, A., 2012. Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. Energy Build. 48, 560-567.
  
Dataset accessible through UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/energy+efficiency#

## Inspiration
We were inspired by current problems with energy efficiency, and wanted to address this in a concrete way. Upon finding the dataset for predicting energy efficiency in homes we knew we would be able to develop a meaningful and useful web application that residential planners and energy companies can use.

## What it does
The web app contains a tool for inputting user specifications for various attributes of a residential building. This includes dimensions (`length`, `width`, `height`), as well as the number of windows, how close the windows are together, and the `relative insulation` of the building. We then construct a 3D visualization of the user's structure and provide them with an estimate on the heating and cooling load for their house, as predicted by the machine learning model we developed. 

The visualization of the user's structure provides a unique representation of the underlying dataset itself. Rather than simply predicting the heating and cooling loss for a set of parameters, we let the user visualize the dataset on their own by creating a building and getting accurate efficiency results.

We also have a section in our web app dedicated to various visualizations of each of the predictors against both heating and cooling load. Seeing the individual scatters along with a `LOWESS trendline` provides users with some intuition on what makes an efficient residential home based on the training data.

## How we built it

The machine learning model is a DecisionTreeRegressor trained on the entire energy efficiency dataset. The loss function is calculated using mean squared error and rigorous evaluation procedures were performed. See each of the 'png's in our github repository.

The main web app consists of a Javascript and CSS frontend, using the backend framework Flask to get our Python scripts containing important model information to run. Our univariate visualizations were done using Plotly Express and our 3D model renders on our main user input page were rendered using Three.js.

## Challenges we ran into

We ran into two main challenges during the hackathon. First, none of us had worked on projects that connected python code with a html/Javascript/CSS front end before. Second, we had to engage in a lot of trial and error to get the Three.js model to render correctly. We stumbled at first by trying to install the Three.js library directly and load it in from our repository, but ended up find a clean solution involving the three.js main src file being available online.

## Accomplishments that we're proud of

We're quite proud of the 3D visualization that we managed to get working. It's exactly how we envisioned the user input visualization to work. We also figured out to successfully retrieve our model predictions from python to be displayed in html. In addition, our ML model actually performs really well on the dataset, and does not suffer from overfitting as per our cross validation tests; this can be an issue when working with decision trees.

## What we learned

We learned how to properly integrate python code with a frontend web app setup, render 3D models with animations in javascript embedded in html, and learned some important statistical concepts about machine learning model evaluation. We also found some cool tools for visualizing the dataset, and were able to integrate those nicely into our html/CSS code as well.

## What's next for Construct: An Energy Efficiency Estimation Platform

Right now, the 3D model is a rectangular prism which uses the user-input length, width, and height from our efficiency estimator web app. However, we'd like to improve upon this model in the future and add a roof and windows to the model that accurately scale to the user's specifications, so that the 3D model more closely line up with the dataset and model predictions
