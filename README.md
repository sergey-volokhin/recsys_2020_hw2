# recsys_2020_hw2
Surya told me to not use sklearns' Nearest Neighbors, so I tried making my own similarity matrix, which takes quite a long time (1.5 hours for user-to-user) to compute.<br/>
So i decided, in addition to my own version, to write the second, faster CF, using sklearn (so that TA doesn't need to wait for hours to grade my HW).<br/>
As a result, there are two files:
  * CF.py (my own)
  * fasterCF.py (sklearn)

They both use pearson similarity (correlation) metric from scipy.

## Running
To run, you need to specify the type of CF (positional argument "user" or "item"):
```
python3 CF.py user
python3 fasterCF.py item
```

## Data
The data I am using is critics reviews from rottentomatoes which I personally scraped (file reviews.tsv).<br/>
From initial dataset of 714337 rating points, I removed movies, which received less than 50 reviews, and critics, who reviewed less than 50 movies.<br/>
Resulting dataset has 427677 rating points, for 4038 distinct movies and 1253 critics.

## Output
For a given user (who should already exist in the dataset), the model outputs top-k (k is variable you can specify) recommendations for the movies that user has not seen yet.

## Results
To check the quality of the algorithm, I created and added to the dataframe a "test_user", who loved Marvel movies (check the code for detailed profile).<br/>
Here are top-10 results on all systems and versions:

* fasterCF, item-to-item:

  * mad_max_fury_road w score 4.833244733392808
  * the_dark_knight w score 4.826900094143733
  * avengers_endgame w score 4.590863011475688
  * interstellar_2014 w score 4.146119877692023
  * avengers_infinity_war w score 4.132211926349221
  * ant_man_and_the_wasp w score 4.126526283978871
  * captain_marvel w score 4.078982245417843
  * indiana_jones_and_the_kingdom_of_the_crystal_skull w score 4.055704759816217
  * marvels_the_avengers w score 4.01556420233463
  * godzilla_2014 w score 3.8349388674222338


* fasterCF, user-to-user:

  * spider_man_into_the_spider_verse w score 5.671013281435961
  * marvels_the_avengers w score 5.568935744530494
  * the_dark_knight w score 5.538268477630433
  * mission_impossible_fallout w score 5.529882483322089
  * mad_max_fury_road w score 5.514411941620574
  * black_panther_2018 w score 5.495419424517688
  * war_for_the_planet_of_the_apes w score 5.446767041038895
  * the_jungle_book_2016 w score 5.4392351636042715
  * thor_ragnarok_2017 w score 5.407300172665458
  * guardians_of_the_galaxy w score 5.380702715960557


* CF, item-to-item

  * the_souvenir w score 4.125
  * custody_2018 w score 3.9696969696969697
  * the_big_lebowski w score 3.925925925925926
  * roger_dodger w score 3.894867625953717
  * my_architect w score 3.8793103448275863
  * revenge_2018 w score 3.8333333333333335
  * train_to_busan w score 3.7962962962962963
  * comedian w score 3.6964179481865984
  * frida w score 3.574468085106383
  * belle_2014 w score 3.550984457835005



* CF, user-to-user

  * inception w score 5.008272528437352
  * 1220551_bounty_hunter w score 5.006577884411386
  * over_the_hedge w core 5.0061926183377095
  * pirates_of_the_caribbean_dead_mans_chest w score 5.006135620190487
  * 10009460_the_road w score 5.005964628029316
  * nick_and_norahs_infinite_playlist w score 5.005817220805755
  * there_will_be_blood w score 5.005800432856403
  * green_lantern w score 5.005702940782325
  * whatever_works w score 5.0056655167421145
  * larry_crowne w score 5.00548967871344

As you can see, sklearn version gives very good results. My version, for some reason, not only runs significantly longer, but also works much worse. I am not sure why.