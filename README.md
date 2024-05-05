# binomial-options-pricing-model
Binomial options pricing model, by Rox, Ross and Rubinstein.

BOPM is a numerical method of obtaining option prices. This model assumes
discrete steps of time and discrete changes in asset price. It's a huge
simplification of the market, but as you decrease the step size it's 
getting closer to Black-Scholes-Merton model of option pricing.

Another assumtion is constant volatility of an underlying asset. Changes in
prices are symmetrical (they go up or down with the same factor).

The model is implemented in Python in binomial-options-pricing-model/bopm.py. 
In jupyter notebook you can see interactive plots of price of an option
vs one parameter.

Testing is done through pytest.
