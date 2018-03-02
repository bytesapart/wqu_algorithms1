from math import log, sqrt, exp
from scipy.stats import norm


def implied_volatility_using_newton_raphson(current_price, exercise_price,
                                            time_to_expiry, risk_free_rate,
                                            call_option_price, sigma):
    """

    :param current_price:       s0      The current underlier stock price
    :param exercise_price:      E       The Exercise Price
    :param time_to_expiry:      t       The Time to Expiry
    :param risk_free_rate:      r       The Risk-Free Rate of Return
    :param call_option_price:   c       The call option price, using Black-Scholes Model
    :param sigma:               sigma   The value of Sigma

    :type current_price:        float
    :type exercise_price:       float
    :type time_to_expiry:       int
    :type risk_free_rate:       float
    :type call_option_price:    float
    :type sigma:                float

    :return: Float containing the Implied Volatility

    """

    # Print all the values
    print('Current Price    :    %f' % current_price)
    print('Exercise Price   :    %f' % exercise_price)
    print('Time To Expiry   :    %d' % time_to_expiry)
    print('Risk Free Rate   :    %f' % risk_free_rate)
    print('Call Option Price:    %f' % call_option_price)
    print('Sigma            :    %f' % sigma)

    # Create an empty list to store sigma values
    sig = list()

    # Set the first value of sigma to the function as sig[1]
    sig.append(sigma)

    # Loop for calculating Implied Volatility using Newton-Raphson Method
    # This loop iterates over 100 iterations for calculation

    for i in range(1, 101):
        d1 = (log(current_price / exercise_price) + (risk_free_rate + sigma ** 2 / 2) * time_to_expiry) / (sigma * sqrt(time_to_expiry))
        d2 = d1 - sigma * sqrt(time_to_expiry)
        f = current_price * norm.cdf(d1) - exercise_price * exp(- risk_free_rate * time_to_expiry) * norm.cdf(d2) - call_option_price

        # Calculate Derivative of d1 w.r.t. sigma
        d_d1 = (sigma ** 2 * time_to_expiry * sqrt(time_to_expiry)
                - (log(current_price / exercise_price) +
                   (risk_free_rate + sigma **2 / 2) * time_to_expiry)
                * sqrt(time_to_expiry)) / (sigma ** 2 * time_to_expiry)

        # Calculate the Derivative of d2 w.r.t. sigma
        d_d2 = d_d1 - sqrt(time_to_expiry)

        # Calculate the Derivative of f(sigma)
        f1 = current_price * norm.pdf(d1) * d_d1 - exercise_price * exp(- risk_free_rate * time_to_expiry) * norm.pdf(d2) * d_d2

        # Update Sigma
        sigma = sigma - f / f1
        sig.append(sigma)

        if abs(sig[i] - sig[i-1]) < 0.00000001:
            sig = sig[0:i+1]
            break

    return sig[-1]

if __name__ == '__main__':
    # Call the Newton Raphson Calculating Method
    s0 = 34
    e = 34
    t = 1
    r = 0.01
    c = 2.7240
    sigma_value = 0.1

    implied_volatility = implied_volatility_using_newton_raphson(s0, e, t, r, c, sigma_value)
    print('The Implied Volatility using Newton-Raphson is: %f' % implied_volatility)

