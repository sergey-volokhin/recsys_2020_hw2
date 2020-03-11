# recsys_2020_hw2
Surya told me to not use sklearns' Nearest Neighbors, so I tried making my own similarity matrix, which takes quite a long time (3+ hours) to compute. 
So i decided, in addition to my own version, to write the second, faster CF, using sklearn (so that TA doesn't need to wait for hours to grade my HW).
As a result, there are two files:
  CF.py (my own)
  fasterCF.py (sklearn)

They both use pearson similarity metric from scipy.

## Running
To run, you need to specify the type (positional argument "user" or "item").

## Data
The data I am using is critics reviews from rottentomatoes. 

## Results
To check the quality of the algorithm, I created and added to the dataframe a "test_user", who loved Marvel movies (check the code for detailed profile).
Here are top-10 results on all systems and versions:

fasterCF, item-to-item:

Movie 1: mad_max_fury_road w score 4.833244733392808
Movie 2: the_dark_knight w score 4.826900094143733
Movie 3: avengers_endgame w score 4.590863011475688
Movie 4: interstellar_2014 w score 4.146119877692023
Movie 5: avengers_infinity_war w score 4.132211926349221
Movie 6: ant_man_and_the_wasp w score 4.126526283978871
Movie 7: captain_marvel w score 4.078982245417843
Movie 8: indiana_jones_and_the_kingdom_of_the_crystal_skull w score 4.055704759816217
Movie 9: marvels_the_avengers w score 4.01556420233463
Movie 10: godzilla_2014 w score 3.8349388674222338


fasterCF, user-to-user:

Movie 1: spider_man_into_the_spider_verse w score 5.671013281435961
Movie 2: marvels_the_avengers w score 5.568935744530494
Movie 3: the_dark_knight w score 5.538268477630433
Movie 4: mission_impossible_fallout w score 5.529882483322089
Movie 5: mad_max_fury_road w score 5.514411941620574
Movie 6: black_panther_2018 w score 5.495419424517688
Movie 7: war_for_the_planet_of_the_apes w score 5.446767041038895
Movie 8: the_jungle_book_2016 w score 5.4392351636042715
Movie 9: thor_ragnarok_2017 w score 5.407300172665458
Movie 10: guardians_of_the_galaxy w score 5.380702715960557



CF, item-to-item




CF, user-to-user

Movie 1: inception w score 5.008272528437352
Movie 2: 1220551_bounty_hunter w score 5.006577884411386
Movie 3: over_the_hedge w score 5.0061926183377095
Movie 4: pirates_of_the_caribbean_dead_mans_chest w score 5.006135620190487
Movie 5: 10009460_the_road w score 5.005964628029316
Movie 6: nick_and_norahs_infinite_playlist w score 5.005817220805755
Movie 7: there_will_be_blood w score 5.005800432856403
Movie 8: green_lantern w score 5.005702940782325
Movie 9: whatever_works w score 5.0056655167421145
Movie 10: larry_crowne w score 5.00548967871344
