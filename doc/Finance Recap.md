# Finance Recap
Here you can find a small recap on portfolios and metrics used to compare portfolios.

## Portfolio
A portfolio is a combination of assets with respective shares. In order to ease our process, we will represent shares as a percentage of the portfolio. We will represent a portfolio $P$ as follows :

* $P$ is of size $N$ if it has $N$ different asset
* list of assets: $(a_i)_{i \in [[1, N]]}$ 
* list of weights: $(w_i)_{i \in [[1, N]]}$ 

To optimize a portfolio, we have to find the best combination of weights and assets. Suppose we have $M$ assets and we want to create the best portfolio of size $N$. It can be represented by the following function: 
$$F: (a_i)_{i \in [[1, M]]} \mapsto \begin{pmatrix} a_i \\ w_i \end{pmatrix}_{i \in [[1, N]]}$$
Then again, this function is supposed to give us the optimal list of assets ande their respective weights. We don't know yet what optimal means and how to compute it. Here comes the metrics.

## Metrics
In order to compare portfolios, we need to pass by some metrics. These metrics are defined for one asset and can be generalized to a portfolio. In this section we will define the following metrics:

* Return (Rendement en français)
* Volatility (Volatilité en français)
* Sharpe 

### Return -- Rendement (Fr)
**Return of an asset:**
The Return value of an asset is an evaluation of the performance of that asset on a certain period. Let $a$ be an asset, and $V_i$ be the Value of $a$. The return value $R$ of $a$ on the period $T = [t_{i-1}, t_i]$ is defined as:
$$ R (a, T) = \frac{V_i - V_{i - 1} + Cash \space flow \space received}{V_{i - 1}} $$
**Return of a portfolio:**
The Return value of a portfolio is an evaluation of the performance of that portfolio on a certain period. It is actually the weighted average of its assets. Let $P$ a portfolio of size $N$. Let $(a_i)_{i \in [[1, N]]}$ the list of assets in $P$ and their respective weights $(w_i)_{i \in [[1, N]]}$. The return value $R$ of $P$ is defined as:
$$ R(P, T) = \sum_{i = 1}^N{w_i R(a_i, T)} $$

### Volatility -- Volatilité (Fr)
**Volatility of an asset:**
volatility $\sigma$ is the degree of variation of a trading price series over time as measured by the standard deviation of logarithmic returns _(Wikipedia)_. In _JUMP_'s slides, it is the standard deviation of the value. We will follow this definition since we are using their API. Let $a$ an asset, $\sigma_a$ its volatility and $V_a$ its Variance on a period of size $n$. Let $\overline a$ the mean value of $a$ on that period and $a_i$ the value of $a$ at an instant $i$. The Volatility of $a$ is defined as:
$$ \sigma_a = \sqrt{V_a} \\
\sigma_a = \sqrt{\frac{\sum_{i = 1}^n (a_i-\overline a)^2}{n}}
$$
**Volatility of a portfolio:**
The volatility of a portfolio is standard deviation of the value (or return) of that portfolio. Since any portfolio is a weighted sum of assets and we can compute the variance-covariance matrix of all assets, it is easy to compute the volatility of a portfolio. 
Let $P$ a porfolio of size $N$ sush as :

* $(a_i)_{i \in [[1, N]]}$ is the list assets composing $P$
* $(w_i)_{i \in [[1, N]]}$ is the list of respective weights
* The values of the assets are taken over a period $T$ of size $n$
* $a_{i,j}$ is the value of asset $a_i$ at the instant $j$ in $T$
* $\overline a_i$ is the mean value of the asset $a_i$ over $T$
* $Cov(a_i, a_j)$ the covariance between assets $a_i$ and $a_j$ over $T$

The mean and covariance over $T$ are defined as:

* $\overline a_i = \frac{1}{n}\sum_{j = 1}^n a_{i, j}$
* $Cov(a_i, a_j) = \frac{1}{n}\sum_{k = 1}^n (a_{i,k} - \overline a_i)(a_{j, k} - \overline a_j)$

Let $p$ the value of $P$, we have $p = \sum_{i=1}^N w_i a_i$. The volatility $\sigma_P$ of the portfolio is simply the standard deviation of $p$, there for $\sigma_P = \sqrt{Var(p)}$ with $Var(p)$ the variance of $p$ over the period $T$. The volatility of a the portfolio is defined as:

$$ \sigma_P = \sqrt{\sum_{i = 1}^N \sum_{j = 1}^N w_i w_j Cov(a_i, a_j)} $$

### Sharpe ratio
In finance, the Sharpe ratio is a way to examine the performance of an investment by adjusting for its risk. The ratio measures the excess return per unit of deviation in an investment asset, typically referred to as risk. 

Let $a$ an asset, $R_a$ the asset return and $Var(a)$ the asset variance. Let $b$ a benchmark asset, $R_b$ the benchmark return and $Var(b)$ the benchmark variance. The sharpe of $a$ is defined as:
$$S_a = \frac{E[R_a - R_b]}{\sqrt{Var(a) - Var(b)}}$$
If we take the asset benchmark as $R_b$ as the *constant* risk-free return, noted $R_f$, we have $Var(R_f) = 0$ since $R_f$ is a constant. The volatility of $a$ is $\sigma_a = \sqrt{Var(R_a)}$
the sharpe becomes: 
$$S_a = \frac{E[R_a - R_f]}{\sigma_a}$$


## Conclusion
In order to optimize a portfolio $P$ of size $N$ (already specified assets), we need to find the weights $(w_i)_{i \in [[1, N]]}$ that maximizes $R(P)$ and minimizes $\sigma_P$


#Implementation
The first idea was to directly resolve the convexe optimization with all assets.
However, our convexe optimization algorithm must take into account that
we can't take more than 20 assets at a time, and that is really hard
to implement.

To resolve this issue, we chose to compute the best weight for a given portfolio.
We will then launch this computation on a significant number of portfolio.
The number of different portfolio we can create from N assets being 20 among N,
it is in our best interest to launch every computation on a new thread.

To implement this algorithm, we had to clean the data retrived from the API.
Some information are missing and we had to complete them.
Also, because we want to reduce the computational time, we chose to compute
every ratio ourselves, such as the return, the sharp etc...
That's why we started our work by trying to find the good formula in order
to compute our ratios the same way the API does.
This step has been difficult and some formula still need to be found.

We then started to implement the first step of our portfolio optimization.
First, we had to create a parser in order to use the data stored in our file.
This parser takes two arguments, a start and an end date. Those dates allow
us to choose the period of time in which we want to work.
If any information during this period of time is missing, we simply ignore the
asset. The parser return a vector of asset containing all data for each valid asset.

## Next step
We now have to create different portfolio randomly and call our
optimization computation on them.
The main challenge here will be to parallelize the execution of each
optimization.