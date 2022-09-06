# gasverbruikmonitor
 
Small project in which I was interested in the effect of weather on gas usage for heating. I made two 'interventions', changes in the way I heat my home, and was curious what the effect was on gas usage. I wanted to eliminate the variation of the usage as effect of weather difference.

## Data

I got my gas usage numbers from my energy provider. The data are integers in whole cubic meters. In the summer period gas is only used for hot water, so I'd see a 1m3 spike every few days. To correct for that I applied a simple smoothing algorithm: nullify the usage in any 10 day period with less than 2m3 usage.

Weather data was obtained from the KNMI. I selected the weather station in Voorschoten as the most applicable to my situation. The data contains various measurements on a daily basis.

## Model
I trained a couple models using the same principle: given the weather as input, what is the predicted gas usage? The idea is that the model learns how I heat my house under different weather conditions. I suspect this is a quite simple relationship.

I selected minimum temperature, maximum temperature, (sun) radation and wind speed as input variables, since I assume those should be sufficient to describe the various ways a building cools down or heats up: by conduction (temperature difference), convection (wind) and radation (sun).

### 1. Bayesian Regression

I tried Bayesian Regression first, which wasn't too bad, but was unable to deal with usage being near zero in the summer period.

![afbeelding](https://user-images.githubusercontent.com/33519926/188610736-f68b7257-8996-48ef-a896-e814a8806407.png)

### 2. Support Vector Machine

Then I tried a Support Vector Machine model, which is better able to deal with the summer period better:

![afbeelding](https://user-images.githubusercontent.com/33519926/188610918-19b5ae18-7a00-4f84-a2cd-f4eb769fee52.png)

### 3. Multilayer perceptron
For fun I also tried a multilayer perceptron using Pytorch, even though I suspected it wouldn't be necessary. And indeed, though prediction quality was roughly similar to the SVM model, it took much longer to train, not even considering the time spent on model architecture and hyperparameter tuning.

## Result

Zooming in on the winter of 2022, where I made an intervention on January 16th, we can now see the intervention had a positive effect, since gas usage dropped compared to the months before.

![afbeelding](https://user-images.githubusercontent.com/33519926/188611996-7268a0ec-5d70-424a-8a3b-f9df20dfcf7c.png)

## Further

Some remaining questions:
- Sensitivity analysis: are the used variables all important? Maybe wind or sun doesn't contribute that much? In 'industry' they commonly only use temperature to correct gas usage for.
- Are the prediction errors acceptable?
