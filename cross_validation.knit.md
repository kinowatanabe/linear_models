---
title: "cross_validation"
author: "Kino Watanabe"
date: "2025-11-28"
output: html_document
---



Load key packages.


``` r
library(tidyverse)
library(p8105.datasets)
library(modelr)
```

Load the LIDAR


``` r
data("lidar")
```

Look at the data


``` r
lidar
```

```
## # A tibble: 221 × 2
##    range logratio
##    <dbl>    <dbl>
##  1   390  -0.0504
##  2   391  -0.0601
##  3   393  -0.0419
##  4   394  -0.0510
##  5   396  -0.0599
##  6   397  -0.0284
##  7   399  -0.0596
##  8   400  -0.0399
##  9   402  -0.0294
## 10   403  -0.0395
## # ℹ 211 more rows
```

``` r
lidar_df = 
  lidar |> 
  mutate(id = row_number())

lidar_df |> 
  ggplot(aes(x = range, y = logratio)) + 
  geom_point()
```

<img src="cross_validation_files/figure-html/unnamed-chunk-4-1.png" width="90%" />

## Create dataframes


``` r
train_df = 
  sample_frac(lidar_df, size = .8) |> 
  arrange(id)

test_df = anti_join(lidar_df, train_df, by = "id")
```


Look at these


``` r
ggplot(train_df, aes(x = range, y = logratio)) + 
  geom_point() + 
  geom_point(data = test_df, color = "red")
```

<img src="cross_validation_files/figure-html/unnamed-chunk-6-1.png" width="90%" />

Fit a few models to `train_df`.


``` r
linear_mod = lm(logratio ~ range, data = train_df)
smooth_mod = mgcv::gam(logratio ~ s(range), data = train_df)
wiggly_mod = mgcv::gam(logratio ~ s(range, k = 50), sp = 10e-8, data = train_df)
```


Look at this!!


``` r
train_df |> 
  add_predictions(wiggly_mod) |> 
  ggplot(aes(x = range, y = logratio)) + 
  geom_point() + 
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-html/unnamed-chunk-8-1.png" width="90%" />


* A quick visual inspection suggests that the `linear model` is too simple, the `standard gam fit` is pretty good, and the `wiggly gam` fit is too complex. Put differently, the `linear model` is too simple and, no matter what training data we use, will never capture the true relationship between variables – it will be consistently wrong due to its simplicity, and is therefore biased. The `wiggly fit`, on the other hand, is chasing data points and will change a lot from one training dataset to the the next – it will be consistently wrong due to its complexity, and is therefore highly variable. Both are bad!


Try computing our RMSEs


``` r
rmse(linear_mod, test_df)
```

```
## [1] 0.129855
```

``` r
rmse(smooth_mod, test_df)
```

```
## [1] 0.05850651
```

``` r
rmse(wiggly_mod, test_df)
```

```
## [1] 0.07286931
```

* The `modelr` has other outcome measures – `RMSE` is the most common, but median absolute deviation is pretty common as well.

* The RMSEs are suggestive that both nonlinear models work better than the linear model, and that the smooth fit is better than the wiggly fit. 


## ITERATE!!

* r, to get a sense of model stability we really need to iterate this whole process. Of course, this could be done using `loops` but that’s a hassle 

*  `crossv_mc` preforms the training / testing split multiple times, a stores the datasets using list columns.


``` r
cv_df = 
  crossv_mc(lidar_df, n = 100) |> 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

* `crossv_mc` tries to be smart about memory – rather than repeating the dataset a bunch of times, it saves the data once and stores the indexes for each training / testing split using a resample object. This can be coerced to a dataframe, and can often be treated exactly like a dataframe.
Did this work? Yes!


``` r
cv_df |> pull(train) |> nth(3)
```

```
## # A tibble: 176 × 3
##    range logratio    id
##    <dbl>    <dbl> <int>
##  1   390  -0.0504     1
##  2   391  -0.0601     2
##  3   394  -0.0510     4
##  4   397  -0.0284     6
##  5   400  -0.0399     8
##  6   402  -0.0294     9
##  7   403  -0.0395    10
##  8   409  -0.0382    14
##  9   412  -0.0500    16
## 10   414  -0.0457    17
## # ℹ 166 more rows
```

Let's fit models over and over.


``` r
lidar_lm = function(df) {
  
  lm(logratio ~ range, data = df)
  
}
```



``` r
cv_df =
  cv_df |> 
  mutate(
    linear_fits = map(train, \(df) lm(logratio ~ range, data = df)),
    smooth_fits = map(train, \(df) mgcv::gam(logratio ~ s(range), data = df)),
    wiggly_fits = map(train, \(df) mgcv::gam(logratio ~ s(range, k = 50), sp = 10e-8, data = df))
  ) |> 
  mutate(
    rmse_linear = map2_dbl(linear_fits, test, rmse),
    rmse_smooth = map2_dbl(smooth_fits, test, rmse),
    rmse_wiggly = map2_dbl(wiggly_fits, test, rmse)
  )
```

*  \(df) lm(logratio ~ range, data = df) --> anonymous function is the same thing as making standalone function

* I now have many training and testing datasets, and I’d like to fit my candidate models above and assess prediction accuracy as I did for the single training / testing split. To do this, I’ll fit models and obtain RMSEs using `mutate` + `map` & `map2.`


Let's try to look at this better. 


``` r
cv_df |> 
  select(starts_with("rmse")) |> 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) |> 
  ggplot(aes(x = model, y = rmse)) + 
  geom_violin()
```

<img src="cross_validation_files/figure-html/unnamed-chunk-14-1.png" width="90%" />

## Child growth




``` r
growth_df = 
  read_csv("nepalese_children.csv")
```

```
## Rows: 2705 Columns: 5
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## dbl (5): age, sex, weight, height, armc
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

Weight v arm_c


``` r
growth_df |> 
  ggplot(aes(x = weight, y = armc)) + 
  geom_point(alpha = .5)
```

<img src="cross_validation_files/figure-html/unnamed-chunk-16-1.png" width="90%" />


Let's show the models we might use.


``` r
growth_df =
  growth_df |> 
  mutate(
    weight_cp7 = (weight > 7) * (weight - 7)
  )
```

* piece-wise linear fit, change point at weight ~7
* For the piecewise linear fit, we need to add a “change point term” to our dataframe.

Let's fit three models


``` r
linear_mod = lm(armc ~ weight, data = growth_df)
pwl_mod    = lm(armc ~ weight + weight_cp7, data = growth_df)
smooth_mod = mgcv::gam(armc ~ s(weight), data = growth_df)
```


``` r
growth_df |> 
  add_predictions(smooth_mod) |> 
  ggplot(aes(x = weight, y = armc)) +
  geom_point(alpha = .5) + 
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-html/unnamed-chunk-19-1.png" width="90%" />

Now cross validate!


``` r
cv_df = 
  crossv_mc(growth_df, n = 100) |> 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

* 80/20 split by default


``` r
cv_df = 
  cv_df |> 
  mutate(
    linear_mod = map(train, \(df) lm(armc ~ weight, data = df)),
    pwl_mod    = map(train, \(df) lm(armc ~ weight + weight_cp7, data = df)),
    smooth_mod = map(train, \(df) mgcv::gam(armc ~ s(weight), data = df))
  ) |> 
  mutate(
    rmse_linear = map2_dbl(linear_mod, test, rmse),
    rmse_pwl    = map2_dbl(pwl_mod, test, rmse),
    rmse_smooth = map2_dbl(smooth_mod, test, rmse)
  )
```

Create my boxplots!


``` r
cv_df |> 
  select(starts_with("rmse")) |> 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) |> 
  ggplot(aes(x = model, y = rmse)) + 
  geom_violin()
```

<img src="cross_validation_files/figure-html/unnamed-chunk-22-1.png" width="90%" />
