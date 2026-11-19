---
layout: default
title: What Makes a Genre? Spotify Audio Features
---

# What Makes a Genre? Spotify Audio Features

**by Joowon Jang** · a final project for DSC 80 at UC San Diego

---

## Introduction

Whenever I hosted a party in my freshman days, Spotify would almost exclusively suggest explicit songs when I searched for party-appropriate playlists, and it also chose mostly hip-hop. That made me wonder whether a song's genre and whether it's explicit actually show up in how it *sounds*.

This project uses a Spotify dataset of **114,000 tracks across 114 genres** (1,000 per genre). Each track comes with Spotify's audio features (how danceable, energetic, acoustic, … it sounds) plus metadata like whether it is explicit. I focus on 5 genres that sound very different: **classical, jazz, hip-hop, edm and metal**. After filtering to these genres and cleaning, **3,615 tracks** remain.

**My question:** *How well do a track's audio features reveal its genre, and do explicit tracks sound different (e.g. more danceable)?*

**Relevant columns:**

| Column | Description |
|---|---|
| `track_genre` | Spotify's genre label (what I predict later) |
| `explicit` | whether the track has explicit content (True/False) |
| `danceability` | how suitable the track is for dancing (0–1) |
| `energy` | perceived intensity and activity (0–1) |
| `acousticness` | confidence the track is acoustic (0–1) |
| `instrumentalness` | likelihood the track has no vocals (0–1) |
| `valence` | how positive/happy the track sounds (0–1) |
| `tempo` | beats per minute (has missing values) |

---

## Data Cleaning and Exploratory Data Analysis

### Data cleaning

First I only kept the 5 genres I care about, which took the data from 114,000 rows down to 5,000. Then I found that the same song shows up more than once in a lot of ways:

- Some track_ids were listed twice under the same genre, so I just kept one.
- Some track_ids were labeled as two of my genres at once (like classical and jazz). Since I'm trying to predict genre later, a song with two answers would just confuse the model, so I dropped those.
- 1,255 rows were literally the same recording (same artist, same title, same audio numbers) just re-released on a different album with a new id. If I left them in, the same song could end up in both my training and test data, which would make my model look better than it really is.

Spotify also gave some songs a tempo of 0 BPM or a time signature of 0. A song can't actually have 0 beats, so I took this as the measurement failing and changed those to NaN. Lastly I made `release_year` and `duration_min` columns. `release_year` looks off for some songs (for example "Vaseegara" shows 1972), probably because it's the date of a re-release, so I don't rely on it.

Here are the first few rows of the cleaned data:

| track_name             | track_genre   |   popularity | explicit   |   danceability |   energy |   tempo |
|:-----------------------|:--------------|-------------:|:-----------|---------------:|---------:|--------:|
| Zara Zara              | classical     |           58 | False      |          0.643 |    0.268 | nan     |
| Zara Zara - Lofi       | classical     |           54 | False      |          0.608 |    0.638 | 140.109 |
| Vaseegara              | classical     |           68 | False      |          0.695 |    0.293 | nan     |
| Zara Zara - LoFi Chill | classical     |           59 | False      |          0.583 |    0.308 | 118.226 |
| Ondra Renda            | classical     |           59 | False      |          0.642 |    0.562 | 149.82  |

### Univariate analysis

<iframe src="assets/danceability_hist.html" width="800" height="500" frameborder="0"></iframe>

The distribution of danceability is pretty symmetric and bell-shaped, centered around 0.55 (most songs fall between 0.42 and 0.68). There's a long tail on the low end, which makes sense because classical songs (median 0.38) are much harder to dance to than hip-hop (median 0.73).

### Bivariate analysis

<iframe src="assets/energy_by_genre.html" width="800" height="500" frameborder="0"></iframe>

The genres split into two clear camps. Classical (median 0.14) and jazz (0.34) are calm, while edm (0.76), hip-hop (0.70) and metal (0.88) are high-energy, with metal being the most intense. This already hints that energy alone could tell some genres apart.

<iframe src="assets/energy_vs_acousticness.html" width="800" height="550" frameborder="0"></iframe>

There's a strong negative relationship (correlation −0.87): the more acoustic a song is, the lower its energy. The genres form their own clusters. Classical and jazz sit in the bottom-right (high acousticness, low energy), and metal, edm and hip-hop are in the top-left.

### Interesting aggregates

Average audio features and the share of explicit songs in each genre:

| track_genre   |   danceability |   energy |   acousticness |   valence |   explicit_share |
|:--------------|---------------:|---------:|---------------:|----------:|-----------------:|
| classical     |           0.38 |     0.19 |           0.92 |      0.38 |             0    |
| edm           |           0.65 |     0.75 |           0.12 |      0.45 |             0.11 |
| hip-hop       |           0.72 |     0.69 |           0.22 |      0.54 |             0.25 |
| jazz          |           0.53 |     0.37 |           0.67 |      0.45 |             0.01 |
| metal         |           0.47 |     0.84 |           0.04 |      0.42 |             0.15 |

Average danceability of explicit vs non-explicit songs **within each genre**:

| track_genre   |   non-explicit |   explicit |
|:--------------|---------------:|-----------:|
| classical     |           0.38 |     nan    |
| edm           |           0.64 |       0.7  |
| hip-hop       |           0.7  |       0.76 |
| jazz          |           0.53 |       0.7  |
| metal         |           0.46 |       0.49 |

In edm, hip-hop and metal, explicit songs are still a bit more danceable even when I compare them to songs from the same genre. This matters because hip-hop is both the most danceable genre and the one with the most explicit songs, so the overall gap could just be "hip-hop vs everything else." Since the gap shows up inside each genre too, genre alone doesn't seem to explain it. Classical has no explicit songs at all, and jazz only has 3, so its 0.70 isn't really reliable.

---

## Assessment of Missingness

### NMAR analysis

I think `tempo` could be **NMAR**. Tempo isn't something a person types in; Spotify's algorithm estimates it from the audio. If a song doesn't have a steady beat (like a lot of classical pieces that speed up and slow down, or free-flowing jazz), the algorithm probably can't lock onto a tempo and leaves it blank. In that case, whether tempo is missing depends on the song's actual rhythm, which is the very value that's missing. To make this closer to MAR, I'd want more data about each song's rhythm, like a "beat strength" score or how confident Spotify's algorithm was in its estimate.

### Missingness dependency

About 21% of tracks are missing `tempo`. I ran permutation tests comparing songs where tempo is missing vs not missing:

| column      | statistic   |   observed |   p_value |
|:------------|:------------|-----------:|----------:|
| track_genre | TVD         |      0.274 |     0.000 |
| energy      | K-S         |      0.322 |     0.000 |
| key         | TVD         |      0.066 |     0.162 |

- **Genre:** Tempo's missingness clearly depends on genre. Classical (34.5% missing) and jazz (31.0%) are missing tempo way more often than metal (9.3%), so tempo is **MAR** on genre.
- **Energy:** It also depends on energy. Songs with missing tempo have much lower energy (average 0.42 vs 0.62), which fits the idea that calm songs without a steady beat are harder to measure.
- **Key:** I fail to reject here, so I don't have evidence that tempo's missingness depends on the song's key.

<iframe src="assets/tempo_missing_energy.html" width="800" height="500" frameborder="0"></iframe>

---

## Hypothesis Testing

- **Null hypothesis:** Among tracks in the 5 genres, explicit and non-explicit tracks have the same distribution of danceability; the observed difference is due to chance.
- **Alternative hypothesis:** Explicit tracks are more danceable on average than non-explicit tracks.
- **Test statistic:** mean danceability (explicit) − mean danceability (non-explicit)
- **Significance level:** 0.05

I used the difference in means because danceability is numerical and I only care about which group is higher on average, and I made it one-sided because my question is specifically whether explicit songs are *more* danceable.

<iframe src="assets/hypothesis_test.html" width="800" height="500" frameborder="0"></iframe>

The observed difference was **0.129**, and none of my 5,000 shuffled differences were that big, so the **p-value is about 0**. I **reject the null hypothesis**: the data suggests explicit songs tend to be more danceable than non-explicit ones. Genre could be a confounder (hip-hop has the most explicit songs, 188 of the 395), but the gap still shows up inside edm, hip-hop and metal. This test only shows an association, not that being explicit *causes* a song to be more danceable.

---

## Framing a Prediction Problem

**Prediction problem:** predict a track's `track_genre` (classical, edm, hip-hop, jazz, metal) from its audio features. This is a **multiclass classification** problem with 5 classes.

**Metric:** I use **macro-averaged F1-score** instead of accuracy. My classes aren't balanced (jazz is only about 14% of the training data while metal is about 24%), and macro-F1 computes the F1-score for each genre separately and averages them, so every genre counts equally. I still report accuracy since it's easy to understand.

**Time of prediction:** I'm imagining predicting a song's genre right when it gets uploaded. At that point Spotify can already compute the audio features from the sound itself, and whether the track is explicit is known too. I left out `popularity` (based on plays after release), the artist/album/track names (they identify the song instead of describing its sound), and `release_year` (unreliable).

---

## Baseline Model

My baseline is a multiclass **logistic regression** using just 2 features: `danceability` and `energy`. Both are quantitative (0 ordinal, 0 nominal), so the only preprocessing was standardizing them with `StandardScaler`, all inside one sklearn Pipeline. I trained it on 75% of the data and evaluated on the 25% test set it never saw.

On the test set it got a **macro-F1 of 0.574** and **accuracy of 0.618**.

| genre | F1 |
|---|---|
| metal | 0.78 |
| classical | 0.74 |
| hip-hop | 0.61 |
| edm | 0.45 |
| jazz | 0.29 |

**Is it good?** Not really. It's much better than always guessing the most common genre (right about 24% of the time), and it does well on the "extreme" genres like classical and metal. But it struggles with jazz and edm, which sit in the middle on danceability and energy. Two features just aren't enough to describe how a genre sounds.

---

## Final Model

For my final model I switched to a **random forest classifier** using 14 audio features, and engineered these new features inside one Pipeline:

1. **Tempo imputation + "tempo was missing" flag.** Tempo is missing much more often for classical and jazz, so the fact that it couldn't be measured is itself a hint about the genre. I filled missing tempos with the median and added a flag for whether it was missing.
2. **Quantile transform on speechiness, liveness and instrumentalness.** These are very skewed (median speechiness is only 0.05 but it goes up to 0.9), so turning them into percentiles spreads out the values near 0.
3. **"Instrumental" flag (instrumentalness > 0.5).** Spotify says above 0.5 suggests no vocals. 66% of classical songs are instrumental compared to only 1% of hip-hop.
4. **One-hot encoding** of `explicit` and `mode`.

I tuned `n_estimators` (100, 300), `max_depth` (None, 10, 20) and `min_samples_split` (2, 10) with 5-fold GridSearchCV on the training set, scoring by macro-F1. The best combination was **300 trees, no max depth, min_samples_split = 2** (average CV macro-F1 0.792).

| model | test macro-F1 | test accuracy |
|---|---|---|
| Baseline | 0.574 | 0.618 |
| **Final** | **0.767** | **0.783** |

Every genre improved: classical 0.86, metal 0.88, hip-hop 0.76, edm 0.71 and jazz 0.63 (still the hardest, but way up from 0.29).

<iframe src="assets/confusion_matrix.html" width="700" height="600" frameborder="0"></iframe>

---

## Fairness Analysis

Does my final model work worse for songs whose tempo couldn't be measured?

- **Group X:** tracks with missing tempo · **Group Y:** tracks with a measured tempo
- **Metric:** accuracy
- **Null hypothesis:** the model is fair; its accuracy is the same for both groups, and any difference is due to chance.
- **Alternative hypothesis:** the model is unfair; its accuracy is lower for tracks with missing tempo.
- **Test statistic:** accuracy(tempo present) − accuracy(tempo missing) · **Significance level:** 0.05

<iframe src="assets/fairness_test.html" width="800" height="500" frameborder="0"></iframe>

The model was right 79.3% of the time on songs with a measured tempo and 74.7% on songs with missing tempo, a gap of about 0.045. The **p-value is about 0.106**, which is above 0.05, so I **fail to reject the null hypothesis**. I don't have strong evidence that the model is unfair to songs with missing tempo, though with only 190 such songs in the test set this doesn't prove it's perfectly fair.
