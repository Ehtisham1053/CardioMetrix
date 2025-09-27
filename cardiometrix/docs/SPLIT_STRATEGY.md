# SPLIT_STRATEGY.md

- Stratify on joint label: "00","01","10","11".
- Train 70% / Val 15% / Test 15%, fixed seed.
- Persist row indices to /indices/{train,val,test}.csv