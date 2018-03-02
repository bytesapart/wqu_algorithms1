import pandas as pd
from pylab import *
import numpy as np
import scipy.signal as sc
import matplotlib.pyplot as plt

# Read the data from the CSV, which was downloaded from Quandl. A direct Quandl connection API was not used
# because that would expose my API key. Secondly, Yahoo! has decommissioned their ichart API, which causes all
# direct connections to Yahoo! Finance (such as matplotlib.finance and pd.datareader) to fail. I could have
# connected to Google to fetch the Data, but opted for Quandl, as it had the best aggregation of commodity collection
# which is freely available. Therefore, the .csv contains data from Quandl, and the datasource should be irrelevant
# for this mini-project.

# Read the Qaundl Data into a DataFrame
data = pd.read_csv('COM-COPPER.csv', index_col=0)

# Slice the DataFrame and get only the 'Value', that is, the actual prices of commodity (here, Copper)
# Also note that the 'Value' pd.Series() has been cast into a numpy array. This is to remove the time series
# because Fourier Transform is continuous, while a Time Series is discrete.

value = np.array(data['Value'])
np.count_nonzero(value)
plt.plot(value, color='r')
plt.title('COM Copper price movement (in USD)')
plt.show()

# Detrend the values
# Detrending is necessary for creating a null hypothesis, as pointed out by
# Prof. Douglas on piazza forums discussion for Unit 3
detrend = sc.detrend(value)
plt.plot(detrend, color='b')
plt.title('COM Copper detrended prices (in USD)')
plt.show()


# Form a Balckman window
# The Blackman window is a taper formed by using the first three terms of a summation of cosines.
w = np.blackman(20)  # Here, our window size or number of points is 20, similar to that in the Labs
y = np.convolve(w/w.sum(), detrend, mode='same')
plt.plot(y, color='g')
plt.title('Blackman window function for detrended COM Copper prices (in USD)')
plt.show()

# Perform Fast Fourier Transform
fft = abs(rfft(y))
plt.plot(fft, color='orange')
plt.title('FFT Algorithm applied to COM Copper Prices')
plt.show()

# print all the peak values
print('All the peak values are %s' % str(fft))

# Print the Highest Magnitude and it's frequency
print('The largest magnitude (y axis) of the Fast Fourier Transform is %f:' % max(fft))
print('The largest magnitude of the Fast Fourier Transform is located at frequency (x axis) %f:' % float(fft.argmax(axis=0)))


